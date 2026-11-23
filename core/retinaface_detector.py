"""
RetinaFace Detector — 集成到 face-photo-sort 管线
===================================================
支持两种后端：
  1. ONNX RetinaFace（推荐）：从 models/retinaface/ 目录加载
     - 需要下载 .onnx 模型（见下方）
  2. torchvision RetinaNet（COCO 预训练）：零配置备选

模型下载（ONNX 格式）：
  InsightFace RetinaFace-R50:
    https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/
    文件: retinaface_r50_v2.onnx (~16MB)
  放到: models/retinaface/retinaface_r50_v2.onnx

  InsightFace RetinaFace-MobileNet0.25:
    https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/
    文件: retinaface_mobilenet025_v2.onnx (~2MB)
  放到: models/retinaface/retinaface_mobilenet025_v2.onnx
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.transforms import functional as TF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))


# ── ONNX RetinaFace ─────────────────────────────────────────

def _load_onnx_retinaface():
    """查找并加载 ONNX RetinaFace 模型"""
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'retinaface')
    candidates = [
        os.path.join(model_dir, 'retinaface_r50_v2.onnx'),
        os.path.join(model_dir, 'retinaface_mobilenet025_v2.onnx'),
    ]
    for p in candidates:
        if os.path.exists(p):
            net = cv2.dnn.readNet(p)
            print(f"[RetinaFace ONNX] Loaded: {os.path.basename(p)}")
            return net
    return None


def _run_onnx_retinaface(net, bgr_img, score_thresh=0.5, nms_thresh=0.4):
    """
    运行 ONNX RetinaFace。
    InsightFace ONNX 输出：shape (N, 15)
      [0:4]   = bbox (x1,y1,x2,y2)
      [4:14]  = 5 landmarks (x,y each)
      [14]    = confidence
    """
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr_img, 1.0, (640, 640),
                                  mean=(104, 117, 123), swapRB=False, crop=False)
    net.setInput(blob)
    raw = net.forward()  # list of outputs or ndarray

    # 解析输出（InsightFace ONNX 格式）
    if isinstance(raw, (list, tuple)):
        dets = raw[0] if len(raw) > 0 else None
    else:
        dets = raw

    results = []
    if dets is None or dets.ndim != 2 or dets.shape[1] < 5:
        return results

    for row in dets:
        conf = float(row[14]) if row.shape[0] > 14 else float(row[4])
        if conf < score_thresh:
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        # Clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue
        # Landmarks
        lm = None
        if row.shape[0] >= 14:
            lm = row[4:14].reshape(5, 2).astype(np.float32)
            lm[:, 0] = np.clip(lm[:, 0], 0, w - 1)
            lm[:, 1] = np.clip(lm[:, 1], 0, h - 1)
        results.append({
            'bbox': (int(x1), int(y1), int(bw), int(bh)),
            'confidence': conf,
            'landmarks': lm,
        })

    # NMS
    if results:
        boxes = [(r['bbox'][0], r['bbox'][1],
                  r['bbox'][0] + r['bbox'][2], r['bbox'][1] + r['bbox'][3])
                 for r in results]
        scores = [r['confidence'] for r in results]
        keep = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, nms_thresh)
        if len(keep) > 0:
            keep = keep.flatten() if keep.ndim > 1 else [keep.item()] if isinstance(keep[0], (np.ndarray,)) else keep
            results = [results[i] for i in keep if i < len(results)]
    return results


# ── torchvision RetinaNet (COCO, 备选) ──────────────────────

_RETINANET_CACHE = {}

def _get_retinanet_coco(score_thresh=0.3, nms_thresh=0.4):
    key = (score_thresh, nms_thresh)
    if key not in _RETINANET_CACHE:
        model = retinanet_resnet50_fpn_v2(weights='DEFAULT')
        model.eval()
        _RETINANET_CACHE[key] = (model, score_thresh, nms_thresh)
        print("[RetinaFace] Loaded torchvision RetinaNet (COCO, ResNet50-FPN v2)")
    return _RETINANET_CACHE[key][0]


def _run_retinanet(model, rgb_img, orig_w, orig_h,
                   score_thresh=0.3, nms_thresh=0.4):
    """运行 torchvision RetinaNet，返回 (bbox_list, conf_list, landmarks_list=None)"""
    pil = Image.fromarray(rgb_img)
    t = TF.to_tensor(pil)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        preds = model([t])[0]

    boxes = preds['boxes'].numpy()
    scores = preds['scores'].numpy()

    results = []
    for box, score in zip(boxes, scores):
        if score < score_thresh:
            continue
        box = np.asarray(box, dtype=np.float32).flatten()
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        results.append({
            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            'confidence': float(score),
            'landmarks': None,
        })

    if results:
        boxes_xyxy = [(r['bbox'][0], r['bbox'][1],
                       r['bbox'][0] + r['bbox'][2],
                       r['bbox'][1] + r['bbox'][3])
                      for r in results]
        scores_list = [r['confidence'] for r in results]
        keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores_list, score_thresh, nms_thresh)
        if len(keep) > 0:
            keep_flat = [x.item() if hasattr(x, 'item') else int(x) for x in keep.flatten()]
            results = [results[i] for i in keep_flat if i < len(results)]
    return results


# ── 主检测函数 ──────────────────────────────────────────────

def detect_retinaface(
    image_path: str,
    mode: str = 'auto',        # 'onnx', 'retinanet', 'auto'
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    min_face_height: int = 200,
    max_face_height: int = 3000,
    expand_ratio: float = 1.0,
    multiscale: bool = True,
    # ArcFace 编码参数（复用 big_face_detector 的实现）
    do_encode: bool = True,
    norm_min: float = 0.1,
    norm_max: float = 9999.0,
) -> list:
    """
    使用 RetinaFace/RetinaNet 检测人脸。

    Args:
        image_path: 图像路径
        mode: 'onnx'（ONNX RetinaFace） / 'retinanet'（COCO RetinaNet） / 'auto'（优先 ONNX）
        score_threshold: 置信度阈值
        nms_threshold: NMS IoU 阈值
        min_face_height / max_face_height: 人脸高度过滤
        expand_ratio: 检测框扩展比例（RetinaFace 本身已覆盖完整脸，
                      但大脸特写可能需要轻微扩展）
        multiscale: 是否多尺度检测
        do_encode: 是否做 ArcFace 编码
    Returns:
        List[dict]: bbox, confidence, landmarks, area_pct, encoding, arcface_norm
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    orig_arr = np.array(img)
    bgr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2BGR)
    total_area = orig_w * orig_h

    # ── 选择后端 ──
    if mode == 'onnx':
        net = _load_onnx_retinaface()
        if net is None:
            print("[RetinaFace] ONNX model not found, falling back to RetinaNet")
            mode = 'retinanet'
    elif mode == 'auto':
        net = _load_onnx_retinaface()
        mode = 'onnx' if net is not None else 'retinanet'

    print(f"[RetinaFace] Mode: {mode}")

    # ── 检测尺度 ──
    if multiscale:
        scales = [1.0, 0.75, 0.5]   # 全尺度 + 缩小（RetinaFace FPN 多尺度强）
    else:
        scales = [1.0]

    all_results = []
    for scale in scales:
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        scaled = cv2.resize(bgr, (sw, sh))
        sx, sy = orig_w / sw, orig_h / sh

        if mode == 'onnx':
            detections = _run_onnx_retinaface(net, scaled, score_threshold, nms_threshold)
        else:
            rgb_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
            detections = _run_retinanet(
                _get_retinanet_coco(score_threshold, nms_threshold),
                rgb_scaled, orig_w, orig_h, score_threshold, nms_threshold
            )

        for det in detections:
            x, y, bw, bh = det['bbox']
            ox, oy = int(x * sx), int(y * sy)
            ow, oh = int(bw * sx), int(bh * sy)
            oh = min(oh, orig_h - oy)

            if oh < min_face_height or oh > max_face_height:
                continue

            # 扩展检测框
            if expand_ratio != 1.0:
                cx, cy = ox + ow / 2, oy + oh / 2
                new_ow = min(int(ow * expand_ratio), orig_w)
                new_oh = min(int(oh * expand_ratio), orig_h)
                ox = max(0, int(cx - new_ow / 2))
                oy = max(0, int(cy - new_oh / 2))
                ow = min(new_ow, orig_w - ox)
                oh = min(new_oh, orig_h - oy)

            area_pct = ow * oh / total_area * 100
            all_results.append({
                'bbox': (ox, oy, ow, oh),
                'confidence': det['confidence'],
                'landmarks': det.get('landmarks'),
                'area_pct': area_pct,
                'scale': scale,
            })

    if not all_results:
        print(f"[RetinaFace] No faces detected: {os.path.basename(image_path)}")
        return []

    # ── NMS 去重 ──
    boxes = [r['bbox'] for r in all_results]
    boxes_xyxy = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boxes]
    scores_all = [r['confidence'] for r in all_results]
    keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores_all, score_threshold, nms_threshold)
    keep_flat = [x.item() if hasattr(x, 'item') else int(x) for x in keep.flatten()]
    nmsed = [all_results[i] for i in keep_flat if i < len(all_results)]

    # ── 按面积排序 ──
    nmsed.sort(key=lambda r: r['area_pct'], reverse=True)

    print(f"[RetinaFace] {len(nmsed)} face(s) after NMS (from {len(all_results)} raw)")

    # ── ArcFace 编码 ──
    if do_encode:
        from big_face_detector import _arcface_encode
        for r in nmsed:
            enc, norm = _arcface_encode(bgr, r['bbox'], r.get('landmarks'))
            r['encoding'] = enc
            r['arcface_norm'] = norm
            norm_ok = norm_min <= norm <= norm_max
            status = '[PASS]' if norm_ok else '[FAIL]'
            x, y, w, h = r['bbox']
            print(f"  bbox=({x},{y},{w},{h}) conf={r['confidence']:.3f} "
                  f"area={r['area_pct']:.1f}% norm={norm:.2f} {status}")

    return nmsed


def detect_and_encode_retinaface(image_path: str, **kwargs) -> list:
    """检测 + 编码，返回 FaceMatcher 格式"""
    detections = detect_retinaface(image_path, **kwargs)
    results = []
    for item in detections:
        x, y, w, h = item['bbox']
        location = (y, x + w, y + h, x)  # (top, right, bottom, left)
        results.append({
            'location': location,
            'bbox': (x, y, w, h),
            'encoding': item.get('encoding'),
            'image_path': image_path,
            'score': item.get('confidence', 0),
            'arcface_norm': item.get('arcface_norm', 0),
            'area_pct': item.get('area_pct', 0),
            'landmarks': item.get('landmarks'),
        })
    return results


if __name__ == '__main__':
    import glob

    base = (
        r'D:\photoes\Nikon相机照片\比特恢复的文件\02 本地磁盘 (E ) 删除文件'
        r' (5222 个文件)\Dir00310742\DSCN9691.JPG'
    )
    matches = glob.glob(base)
    img_path = matches[0] if matches else None

    if img_path:
        print(f"Image: {img_path}\n")
        print("=" * 60)

        # ONNX RetinaFace（需要下载模型）
        print("\n--- ONNX RetinaFace (auto) ---")
        r = detect_retinaface(img_path, mode='auto', score_threshold=0.4,
                              expand_ratio=1.0, min_face_height=150)
        if not r:
            # Fallback to RetinaNet
            print("\n--- torchvision RetinaNet (fallback) ---")
            r = detect_retinaface(img_path, mode='retinanet',
                                  score_threshold=0.2,
                                  expand_ratio=1.0, min_face_height=150)

        print(f"\nTotal: {len(r)} face(s)")
    else:
        print("Image not found")
