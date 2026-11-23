"""
Big Face Detector - 大人脸专用检测管线 v5
==========================================
问题：DSCN9691.JPG（4000×3000），人脸占图 25.6%（1767px高），
原 v3 的 scale 公式写反了，导致漏检。

v5 改进：
  - 新增 OpenCV DNN ResNet-SSD 作为首选检测器（大脸特写最佳）
  - YuNet + expand 作为备选（小脸/多脸场景）
  - 自动选择策略：根据图像特征智能切换

检测器优先级（fallback 链）：
  1. OpenCV DNN ResNet-SSD  ← 首选（大脸特写，置信度高）
  2. YuNet + expand_bbox     ← 备选（小脸/多脸）
  3. YOLOv8n-face            ← 兜底（已有）

关键参数（实测校准）：
  scale = target_h / orig_h
  orig_h = 3000（图像高）
  目标：把人脸缩放到 200-700px（YuNet 最佳区间）

  target_h=500  → scale=500/3000≈0.167  → 人脸1767px→294px ✓ 刚好检测
  target_h=700  → scale=700/3000≈0.233  → 人脸1767px→412px ✓
  target_h=900  → scale=900/3000=0.3     → 人脸1767px→530px ✓ 最佳

核心策略：
  1. 多尺度检测（让YuNet在最佳尺度上工作）
  2. NMS 合并
  3. 按面积占比 × YuNet分数 综合排序
  4. 只返回 Top-N（消除小误检，默认 top_n_by_area=1）
  5. ArcFace 编码质量验证（辅助，不作主要过滤）
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# 引用主 detector 的编码器
sys.path.insert(0, os.path.dirname(__file__))

# 尝试导入 OpenCV DNN 检测器
try:
    from opencv_dnn_detector import detect_opencv_dnn_face
    _HAS_OPENCV_DNN = True
except Exception:
    _HAS_OPENCV_DNN = False


def _load_yunet(model_path, scale_w, scale_h, score_threshold=0.25):
    detector = cv2.FaceDetectorYN.create(
        model_path, '', (scale_w, scale_h),
        score_threshold=score_threshold,
        nms_threshold=0.35,
        top_k=30)
    return detector


def _arcface_encode(bgr_img, bbox, face_info=None):
    """提取 ArcFace 512d 特征向量。Returns: (encoding, norm)"""
    from face_detector import _get_onnx_session

    x, y, w, h = [int(v) for v in bbox]
    x = max(0, x); y = max(0, y)
    if w <= 0 or h <= 0:
        return None, 0

    arcface_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'arcface')
    arcface_path = None
    for p in ['w600k_mbf.onnx', 'w600k_r50.onnx', 'buffalo_l.onnx']:
        p2 = os.path.join(arcface_dir, p)
        if os.path.exists(p2):
            arcface_path = p2
            break
    if arcface_path is None:
        return None, 0

    session = _get_onnx_session(arcface_path)
    if session is None:
        return None, 0

    face_crop = bgr_img[y:y+h, x:x+w]
    if face_crop.size == 0:
        return None, 0

    # 对齐裁剪（使用 landmarks）
    if face_info is not None and len(face_info) >= 14:
        kps = face_info[4:14].reshape(5, 2).astype(np.float32)
        dst = np.array([
            [30.2946, 51.6968], [65.5318, 51.5014],
            [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041],
        ], dtype=np.float32)
        tform, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
        if tform is not None:
            aligned = cv2.warpAffine(bgr_img, tform, (112, 112), borderValue=0.0)
        else:
            aligned = cv2.resize(face_crop, (112, 112))
    else:
        aligned = cv2.resize(face_crop, (112, 112))

    # ArcFace 预处理（/255 → /0.5 - 1 = ×2 - 1 → [-1, 1]）
    blob = aligned.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    blob = (blob - mean) / std
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    try:
        input_name = session.get_inputs()[0].name
        feature = session.run(None, {input_name: blob})[0]
        feat = feature.flatten()
        return feat.tolist(), float(np.linalg.norm(feat))
    except Exception:
        return None, 0


def _nms(boxes, iou_threshold=0.35):
    """在原图尺度上做 NMS，返回保留的 bbox 元组列表"""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = x1 + arr[:, 2], y1 + arr[:, 3]
    areas = arr[:, 2] * arr[:, 3]
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return [tuple(boxes[i]) for i in keep]


def _expand_bbox_to_full_face(bbox, orig_w, orig_h, expand_ratio=1.5, min_margin=50):
    """
    将 YuNet 检测的"面部核心区域"扩展到完整脸部边界。
    
    策略：
    1. 按比例扩展（默认 1.5x）
    2. 如果扩展后接近图像边界，直接贴边（适用于"脸占满照片"场景）
    """
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2
    
    # 计算扩展后的尺寸
    new_w = w * expand_ratio
    new_h = h * expand_ratio
    
    # 计算新边界
    new_x = cx - new_w/2
    new_y = cy - new_h/2
    
    # 如果接近边界，贴边（适用于大脸场景）
    if new_x < min_margin:
        new_x = 0
    if new_y < min_margin:
        new_y = 0
    if new_x + new_w > orig_w - min_margin:
        new_w = orig_w - new_x
    if new_y + new_h > orig_h - min_margin:
        new_h = orig_h - new_y
    
    # 确保整数坐标
    new_x = max(0, int(new_x))
    new_y = max(0, int(new_y))
    new_w = min(int(new_w), orig_w - new_x)
    new_h = min(int(new_h), orig_h - new_y)
    
    return (new_x, new_y, new_w, new_h)


def detect_face_auto(
    image_path: str,
    prefer_detector: str = 'auto',  # 'auto', 'opencv_dnn', 'yunet'
    **kwargs
) -> list:
    """
    自动选择检测器的人脸检测。
    
    策略：
    - auto: 优先 OpenCV DNN，失败则 fallback 到 YuNet
    - opencv_dnn: 强制使用 OpenCV DNN
    - yunet: 强制使用 YuNet
    
    Returns: list of dict (统一格式)
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    total_area = orig_w * orig_h
    
    results = []
    
    # 策略 1: OpenCV DNN (首选，大脸效果好)
    if prefer_detector in ('auto', 'opencv_dnn') and _HAS_OPENCV_DNN:
        try:
            dnn_results = detect_opencv_dnn_face(
                image_path,
                conf_threshold=kwargs.get('conf_threshold', 0.3),
                min_face_height=kwargs.get('min_face_height', 60),
                max_face_height=kwargs.get('max_face_height', 3000),
                expand_ratio=kwargs.get('expand_ratio', 1.0),
                multiscale=kwargs.get('multiscale', True),
                do_encode=kwargs.get('do_encode', True),
                norm_min=kwargs.get('norm_min', 0.1),
                norm_max=kwargs.get('norm_max', 9999.0),
            )
            if dnn_results:
                # 转换格式统一
                for r in dnn_results:
                    x, y, w, h = r['bbox']
                    results.append({
                        "bbox": r['bbox'],
                        "score": r['confidence'],
                        "area_pct": r['area_pct'],
                        "arcface_norm": r.get('arcface_norm', 0),
                        "encoding": r.get('encoding'),
                        "detector": "opencv_dnn",
                        "scale": r.get('scale', 1.0),
                    })
                print(f"[AutoDetect] OpenCV DNN: {len(results)} face(s)")
                return results
            elif prefer_detector == 'opencv_dnn':
                print("[AutoDetect] OpenCV DNN: no faces, returning empty")
                return []
        except Exception as e:
            print(f"[AutoDetect] OpenCV DNN failed: {e}")
            if prefer_detector == 'opencv_dnn':
                return []
    
    # 策略 2: YuNet (备选)
    if prefer_detector in ('auto', 'yunet'):
        print("[AutoDetect] Falling back to YuNet...")
        yunet_results = detect_big_face(image_path, **kwargs)
        for r in yunet_results:
            r['detector'] = 'yunet'
        return yunet_results
    
    return results


def detect_big_face(
    image_path: str,
    min_face_height: int = 200,
    max_face_height: int = 1800,
    # ── 面积占比优先模式 ─────────────────────────────────────
    # top_n_by_area > 0 时，只返回面积最大的 N 个（消除小误检）
    top_n_by_area: int = 1,
    # min_area_pct = 人脸最小占图面积百分比（%），0=不限制
    min_area_pct: float = 0,
    # min_yunet_score = YuNet 最低置信度
    min_yunet_score: float = 0.20,
    # norm 过滤（w600k_mbf 实测真脸≈5-15；辅助过滤，不作主要依据）
    norm_min: float = 0.1,
    norm_max: float = 9999.0,
    # ── 扩展检测框到完整脸部 ─────────────────────────────────
    expand_bbox: bool = True,
    expand_ratio: float = 1.5,
) -> list:
    """
    检测大尺寸人脸（高分辨率照片专用）。

    Returns:
        List[dict] — 每个 dict:
            bbox (x,y,w,h), score (YuNet), arcface_norm,
            area_pct (占图%), combined_score, encoding (ArcFace 512d)
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    orig_arr = np.array(img)
    bgr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2BGR)
    total_area = orig_w * orig_h

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'face_detection_yunet_2023mar.onnx'
    )

    # ── 核心：多尺度检测 ────────────────────────────────────
    # scale = target_h / orig_h  （原图高度 → 缩放后高度）
    # YuNet 最佳检测区间：人脸在缩放后图中 200-700px
    # 人脸占图越大 → 需要越小的 scale 才能把人脸压到 200-700px
    #
    # 公式：orig_face_h (px) * scale = scaled_face_h (px)
    #       scale = scaled_face_h / orig_face_h = target_h / orig_h
    #
    # 示例（orig_h=3000）：
    #   target_h=300 → scale=0.1  → 1767px人脸→177px（太小，检测不到）
    #   target_h=500 → scale=0.167 → 1767px人脸→295px（边缘，能检测）
    #   target_h=700 → scale=0.233 → 1767px人脸→412px（最佳）
    #   target_h=900 → scale=0.3   → 1767px人脸→530px（最佳）
    #   target_h=1200→ scale=0.4   → 1767px人脸→707px（边缘）

    scales = [
        (0.167, "tight (target=500px)"),   # 核心尺度：能检1767px人脸
        (0.233, "mid   (target=700px)"),   # 核心尺度
        (0.300, "std   (target=900px)"),   # 核心尺度
        (0.400, "large (target=1200px)"),  # 覆盖更大的脸
        (0.133, "small (target=400px)"),   # 覆盖较小的脸
        (0.500, "full  (target=1500px)"),  # 覆盖最大人脸
    ]

    all_candidates = []

    for scale, desc in scales:
        new_w = max(320, int(orig_w * scale))
        new_h = max(240, int(orig_h * scale))
        # 不放大（只缩小）
        if new_w > orig_w or new_h > orig_h:
            continue

        try:
            detector = _load_yunet(model_path, new_w, new_h, score_threshold=min_yunet_score)
        except Exception:
            continue

        resized = cv2.resize(orig_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, faces = detector.detect(resized)

        if faces is None:
            continue

        scale_x = orig_w / new_w
        scale_y = orig_h / new_h

        for f in faces:
            if f is None:
                continue
            fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            score = float(f[-1]) if len(f) >= 7 else 0.5

            ox = max(0, int(fx * scale_x))
            oy = max(0, int(fy * scale_y))
            ow = min(int(fw * scale_x), orig_w - ox)
            oh = min(int(fh * scale_y), orig_h - oy)

            if ow < 30 or oh < 30:
                continue

            area_pct = (ow * oh) / total_area * 100

            all_candidates.append({
                "bbox": (ox, oy, ow, oh),
                "score": score,
                "face_info": f,
                "scale": scale,
                "scale_desc": desc,
                "area_pct": area_pct,
                "orig_face_h": oh,  # 映射回原图的高度
            })

    if not all_candidates:
        print(f"[BigFace] No detections for {os.path.basename(image_path)}")
        return []

    print(f"[BigFace] {os.path.basename(image_path)}: "
          f"{len(all_candidates)} raw detections")

    # ── 高度 + 面积过滤 ──
    filtered = [
        c for c in all_candidates
        if min_face_height <= c["orig_face_h"] <= max_face_height
        and c["area_pct"] >= min_area_pct
    ]
    if not filtered:
        filtered = all_candidates
    print(f"[BigFace] after height/area filter: {len(filtered)}")

    # ── 扩展检测框到完整脸部 ──
    if expand_bbox:
        for c in filtered:
            orig_bbox = c["bbox"]
            c["bbox"] = _expand_bbox_to_full_face(
                orig_bbox, orig_w, orig_h, expand_ratio
            )
            c["orig_face_h"] = c["bbox"][3]  # 更新高度
            c["area_pct"] = (c["bbox"][2] * c["bbox"][3]) / total_area * 100
            c["expanded"] = True

    # ── NMS ──
    bboxes = [c["bbox"] for c in filtered]
    kept_boxes = _nms(bboxes, iou_threshold=0.35)
    kept_set = set(kept_boxes)
    nmsed = [c for c in filtered if c["bbox"] in kept_set]
    print(f"[BigFace] after NMS: {len(nmsed)}")

    # ── 关键改进：按 (面积占比 × 10 + YuNet分数) 综合排序 ───
    # 面积越大 → 越可能是真实大脸 → 综合分数越高
    # 误检通常面积小，分数低，被排到后面
    for c in nmsed:
        c["combined_score"] = c["area_pct"] * 10 + c["score"] * 5
    nmsed.sort(key=lambda c: c["combined_score"], reverse=True)

    # ── Top-N（消除小误检）──
    if top_n_by_area > 0 and len(nmsed) > top_n_by_area:
        print(f"[BigFace] Limited to top {top_n_by_area} by area")
        nmsed = nmsed[:top_n_by_area]

    # ── ArcFace 编码 ──
    results = []
    for c in nmsed:
        enc, norm = _arcface_encode(bgr, c["bbox"], c.get("face_info"))
        c["arcface_norm"] = norm
        c["encoding"] = enc

        norm_ok = norm_min <= norm <= norm_max
        print(f"  bbox={c['bbox']} h={c['orig_face_h']}px "
              f"area={c['area_pct']:.1f}% combined={c['combined_score']:.2f} "
              f"norm={norm:.2f} yunet={c['score']:.3f} [{c['scale_desc']}] "
              f"[{'PASS' if norm_ok else 'FAIL'}]")

        if norm_ok and enc is not None:
            results.append(c)

    # Fallback：全部被过滤，取面积最大的
    if not results and nmsed:
        best = nmsed[0]
        print(f"[BigFace] WARNING: all filtered. Using top-1 by area as fallback.")
        enc, norm = _arcface_encode(bgr, best["bbox"], best.get("face_info"))
        best["arcface_norm"] = norm
        best["encoding"] = enc
        results = [best]

    print(f"[BigFace] FINAL: {len(results)} face(s) returned")
    return results


def detect_and_encode(
    image_path: str,
    prefer_detector: str = 'auto',
    expand_bbox: bool = True,
    expand_ratio: float = 1.5,
    **kwargs
) -> list:
    """检测 + 编码，返回 FaceMatcher 格式"""
    detections = detect_face_auto(
        image_path,
        prefer_detector=prefer_detector,
        expand_bbox=expand_bbox,
        expand_ratio=expand_ratio,
        **kwargs
    )
    results = []
    for item in detections:
        ox, oy, ow, oh = item["bbox"]
        location = (oy, ox + ow, oy + oh, ox)
        results.append({
            "location": location,
            "bbox": (ox, oy, ow, oh),
            "encoding": item["encoding"],
            "image_path": image_path,
            "score": item.get("score", 0),
            "arcface_norm": item.get("arcface_norm", 0),
            "area_pct": item.get("area_pct", 0),
            "detector": item.get("detector", 'unknown'),
        })
    return results


if __name__ == "__main__":
    import glob, sys

    base = (
        r'D:\photoes\Nikon相机照片\比特恢复的文件\02 本地磁盘 (E ) 删除文件'
        r'*\Dir00310742\DSCN9691.JPG'
    )
    matches = glob.glob(base, recursive=True)
    test_path = sys.argv[1] if len(sys.argv) > 1 else (matches[0] if matches else None)

    if test_path:
        print(f"\n{'='*60}")
        print(f"Testing: {test_path}")
        print(f"{'='*60}\n")

        print("=== Auto Detect (OpenCV DNN → YuNet fallback) ===")
        r1 = detect_and_encode(test_path, prefer_detector='auto', top_n_by_area=1)
        print(f"\nResult: {len(r1)} face(s)")
        for d in r1:
            print(f"  detector={d.get('detector','unknown')}, "
                  f"bbox={d['bbox']}, area={d.get('area_pct',0):.1f}%, "
                  f"norm={d.get('arcface_norm',0):.2f}")

        print()

        print("=== Force YuNet (for comparison) ===")
        r2 = detect_and_encode(test_path, prefer_detector='yunet', top_n_by_area=1,
                                expand_bbox=True, expand_ratio=1.5)
        print(f"\nResult: {len(r2)} face(s)")
        for d in r2:
            print(f"  detector={d.get('detector','unknown')}, "
                  f"bbox={d['bbox']}, area={d.get('area_pct',0):.1f}%, "
                  f"norm={d.get('arcface_norm',0):.2f}")
    else:
        print("Usage: python big_face_detector.py <image_path>")
