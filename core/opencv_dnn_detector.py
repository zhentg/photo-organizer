"""
OpenCV DNN Face Detector — 零依赖、轻量级
==========================================
OpenCV 内置了 ResNet-SSD 人脸检测器，来自 OpenCV 官方模型仓库：
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/

模型文件：
  - deploy.prototxt    (caffe 风格的网络定义)
  - res10_300x300_ssd_iter_140000.caffemodel  (预训练权重)

会自动检测并下载（如果不存在）。

对于 DSCN9691 这类大脸照片，DNN SSD 相比 YuNet 的优势：
  - 对遮挡和极端角度更鲁棒
  - 边界框通常更贴合面部外轮廓
"""

import os, sys
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_opencv_dnn_model():
    """加载 OpenCV DNN 人脸检测器（ResNet-SSD）"""
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'opencv_dnn_face')
    os.makedirs(model_dir, exist_ok=True)

    prototxt = os.path.join(model_dir, 'deploy.prototxt')
    weights  = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

    # 自动下载
    base = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
    for fname, url in [('deploy.prototxt', base + 'deploy.prototxt'),
                        ('res10_300x300_ssd_iter_140000.caffemodel', base + 'res10_300x300_ssd_iter_140000.caffemodel')]:
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"[OpenCV DNN] Downloading {fname}...")
            try:
                import ssl, urllib.request
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                    data = resp.read()
                with open(path, 'wb') as f:
                    f.write(data)
                print(f"[OpenCV DNN] Downloaded: {fname} ({len(data)/1024/1024:.1f} MB)")
            except Exception as e:
                print(f"[OpenCV DNN] Download failed: {e}")
                return None

    net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return net


def detect_opencv_dnn(bgr_img, conf_threshold=0.5):
    """
    运行 OpenCV DNN ResNet-SSD 人脸检测。

    Returns: list of dicts {bbox, confidence, landmarks=None}
    """
    h, w = bgr_img.shape[:2]

    # 推理尺寸：保持宽高比
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr_img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    if not hasattr(detect_opencv_dnn, '_net') or detect_opencv_dnn._net is None:
        detect_opencv_dnn._net = _get_opencv_dnn_model()

    if detect_opencv_dnn._net is None:
        return []

    net = detect_opencv_dnn._net
    net.setInput(blob)
    detections = net.forward()

    results = []
    for det in detections[0, 0]:
        conf = float(det[2])
        if conf < conf_threshold:
            continue
        x1 = int(det[3] * w)
        y1 = int(det[4] * h)
        x2 = int(det[5] * w)
        y2 = int(det[6] * h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            continue
        results.append({
            'bbox': (x1, y1, bw, bh),
            'confidence': conf,
            'landmarks': None,
        })

    # NMS — 多脸场景用 IOU=0.5（避免抑制相邻人脸）；
    #      单人/大脸场景下抑制误检用 IOU=0.4
    iou_threshold = 0.5 if len(results) > 2 else 0.4
    if results:
        boxes_xyxy = [(r['bbox'][0], r['bbox'][1],
                       r['bbox'][0] + r['bbox'][2],
                       r['bbox'][1] + r['bbox'][3]) for r in results]
        scores_list = [r['confidence'] for r in results]
        keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores_list, conf_threshold, iou_threshold)
        keep_flat = [int(x) for x in keep.flatten()] if len(keep) > 0 else []
        results = [results[i] for i in keep_flat if i < len(results)]

    return results


# ── 主检测函数（复用 retinaface_detector 的结构）─────────────

def detect_opencv_dnn_face(
    image_path: str,
    conf_threshold: float = 0.3,
    min_face_height: int = 60,
    max_face_height: int = 3000,
    expand_ratio: float = 1.0,
    multiscale: bool = True,
    do_encode: bool = True,
    norm_min: float = 0.1,
    norm_max: float = 9999.0,
) -> list:
    """
    使用 OpenCV DNN ResNet-SSD 检测人脸。
    自动多尺度，集成 ArcFace 编码。
    """
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    orig_arr = np.array(img)
    bgr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2BGR)
    total_area = orig_w * orig_h

    print(f"[OpenCV DNN Face] Image: {orig_w}x{orig_h}, expand={expand_ratio}x")

    if multiscale:
        scales = [1.0, 0.75, 0.5]
    else:
        scales = [1.0]

    all_results = []
    for scale in scales:
        sw, sh = int(orig_w * scale), int(orig_h * scale)
        scaled = cv2.resize(bgr, (sw, sh))
        sx, sy = orig_w / sw, orig_h / sh
        detections = detect_opencv_dnn(scaled, conf_threshold)

        for det in detections:
            x, y, bw, bh = det['bbox']
            ox, oy = int(x * sx), int(y * sy)
            ow, oh = int(bw * sx), int(bh * sy)
            oh = min(oh, orig_h - oy)
            if oh < min_face_height or oh > max_face_height:
                continue
            if expand_ratio != 1.0:
                cx, cy = ox + ow / 2.0, oy + oh / 2.0
                new_ow = min(int(ow * expand_ratio), orig_w)
                new_oh = min(int(oh * expand_ratio), orig_h)
                ox = max(0, int(cx - new_ow / 2.0))
                oy = max(0, int(cy - new_oh / 2.0))
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
        print(f"[OpenCV DNN Face] No faces detected")
        return []

    # NMS
    boxes_xyxy = [(r['bbox'][0], r['bbox'][1],
                   r['bbox'][0] + r['bbox'][2],
                   r['bbox'][1] + r['bbox'][3]) for r in all_results]
    scores_all = [r['confidence'] for r in all_results]
    keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores_all, conf_threshold, 0.4)
    keep_flat = [int(x) for x in keep.flatten()] if len(keep) > 0 else []
    nmsed = [all_results[i] for i in keep_flat if i < len(all_results)]
    nmsed.sort(key=lambda r: r['area_pct'], reverse=True)

    print(f"[OpenCV DNN Face] {len(nmsed)} face(s) after NMS")

    # ArcFace 编码
    if do_encode:
        from big_face_detector import _arcface_encode
        for r in nmsed:
            enc, norm = _arcface_encode(bgr, r['bbox'], r.get('landmarks'))
            r['encoding'] = enc
            r['arcface_norm'] = norm
            norm_ok = norm_min <= norm <= norm_max
            x, y, w, h = r['bbox']
            print(f"  bbox=({x},{y},{w},{h}) conf={r['confidence']:.3f} "
                  f"area={r['area_pct']:.1f}% norm={norm:.2f} {'[PASS]' if norm_ok else '[FAIL]'}")

    return nmsed


if __name__ == '__main__':
    import sys
    img = (
        r"D:\photoes\Nikon相机照片\比特恢复的文件\02 本地磁盘 (E ) 删除文件"
        r"  (5222 个文件)\Dir00310742\DSCN9691.JPG"
    )
    print("=" * 60)
    results = detect_opencv_dnn_face(img, conf_threshold=0.3,
                                      expand_ratio=1.0,
                                      min_face_height=150,
                                      multiscale=True)
    print(f"\nTotal: {len(results)} face(s)")
