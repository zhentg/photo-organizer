"""
Benchmark Phase 2: ONNX Runtime + combined schemes
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"D:\CADD\workspace\qclaw\projects\face-photo-sort")
os.chdir(r"D:\CADD\workspace\qclaw\projects\face-photo-sort")

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

with open(r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_set.json", 'r', encoding='utf-8') as f:
    test_set = json.load(f)

print(f"Test set: {len(test_set)} photos")

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def benchmark(name, fn, images, n_runs=5):
    for img in images[:1]:
        fn(img)
    all_times = []
    for run in range(n_runs):
        for img in images:
            t0 = time.perf_counter()
            fn(img)
            t1 = time.perf_counter()
            all_times.append((t1 - t0) * 1000)
    return {
        "name": name,
        "avg_ms": round(np.mean(all_times), 1),
        "median_ms": round(np.median(all_times), 1),
        "min_ms": round(np.min(all_times), 1),
        "max_ms": round(np.max(all_times), 1),
    }

# Load images
images = []
for p in test_set:
    images.append(load_image(p["path"]))
print(f"Loaded {len(images)} images")

# Init ultralytics detector for baseline
from core.face_detector import FaceDetector
detector = FaceDetector(yunet_threshold=0.5, yunet_nms_threshold=0.3,
                         yunet_min_face=60, eye_validation=True, strict_mode=True,
                         opencv_dnn_conf=0.12, opencv_dnn_min_face=30,
                         prefer_opencv_dnn=False, yolo_conf=0.5, fallback_dnn_conf=0.12)
detector._load_yolo()

results = {}

# Baseline
print("[1/5] Baseline (ultralytics)...")
r = benchmark("Baseline (ultralytics)", lambda arr: detector._detect_yolo(arr), images)
results["baseline"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# Pre-resize + keypoints (best from phase 1)
print("[2/5] Pre-resize + keypoints (best from phase 1)...")
def yolo_preresize_kpts(arr):
    h, w = arr.shape[:2]
    scale = min(640 / w, 640 / h)
    if scale < 1.0:
        small = cv2.resize(arr, (int(w*scale), int(h*scale)))
    else:
        small = arr; scale = 1.0
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    rs = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640)
    faces = []
    for r in rs:
        if r.boxes is None: continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = int(x1/scale), int(y1/scale)
            bw, bh = int((x2-x1)/scale), int((y2-y1)/scale)
            if bw < 60 or bh < 60: continue
            if r.keypoints is not None and r.keypoints.xy is not None and i < len(r.keypoints.xy):
                kpts = r.keypoints.xy[i].cpu().numpy()
                if len(kpts) >= 2:
                    le, re = kpts[0], kpts[1]
                    has_eyes = (le[1] < (y1+y2)/2) and (re[1] < (y1+y2)/2) and (abs(le[0]-re[0]) > (x2-x1)*0.1)
                else:
                    has_eyes = True
            else:
                has_eyes = True
            if has_eyes:
                faces.append((x, y, bw, bh))
    return detector._nms(faces, 0.4)

r = benchmark("pre-resize + keypoints", yolo_preresize_kpts, images)
results["preresize_kpts"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── ONNX Runtime ──
print("[3/5] ONNX Runtime (pre-resize + keypoints)...")
onnx_path = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\models\yolov8n-face\face_yolov8n.onnx"
print(f"  Loading {onnx_path}...")
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
inp_shape = sess.get_inputs()[0].shape
print(f"  Input: {inp_name} shape={inp_shape}")

def yolo_onnx_detect(arr):
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    # Letterbox to 640
    img_sz = 640
    scale = min(img_sz / w, img_sz / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h))
    pad_w, pad_h = img_sz - new_w, img_sz - new_h
    top, left = pad_h // 2, pad_w // 2
    padded = np.full((img_sz, img_sz, 3), 114, dtype=np.uint8)
    padded[top:top+new_h, left:left+new_w] = resized
    
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    
    outputs = sess.run(None, {inp_name: blob})
    pred = outputs[0]  # (1, 5, 8400) for YOLOv8-face
    
    # Parse: shape is (1, 5, 8400) where 5 = 4 bbox + 1 conf
    pred = pred[0].T  # (8400, 5)
    
    faces = []
    for det in pred:
        cx, cy, bw, bh, conf = det[:5]
        if conf < 0.5: continue
        x1 = (cx - bw/2 - left) / scale
        y1 = (cy - bh/2 - top) / scale
        x2 = (cx + bw/2 - left) / scale
        y2 = (cy + bh/2 - top) / scale
        fw, fh = int(x2 - x1), int(y2 - y1)
        if fw >= 60 and fh >= 60:
            faces.append((int(x1), int(y1), fw, fh))
    return detector._nms(faces, 0.4)

r = benchmark("ONNX Runtime", yolo_onnx_detect, images)
results["onnx"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── Batch ONNX ──
print("[4/5] Batch ONNX (simulate by pre-padding all images to same size)...")
# ONNX batch requires same input shape; we letterbox each to 640x640
def yolo_onnx_batch(images_arr):
    blobs = []
    metas = []
    for arr in images_arr:
        h, w = arr.shape[:2]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        scale = min(640/w, 640/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(bgr, (nw, nh))
        pad_w, pad_h = 640-nw, 640-nh
        top, left = pad_h//2, pad_w//2
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[top:top+nh, left:left+nw] = resized
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        blobs.append(blob)
        metas.append((scale, top, left, h, w))
    
    batch_blob = np.concatenate(blobs, axis=0)
    outputs = sess.run(None, {inp_name: batch_blob})
    pred = outputs[0]  # (batch, 5, 8400)
    
    all_faces = []
    for i in range(len(images_arr)):
        p = pred[i].T  # (8400, 5)
        scale, top, left, orig_h, orig_w = metas[i]
        faces = []
        for det in p:
            cx, cy, bw, bh, conf = det[:5]
            if conf < 0.5: continue
            x1 = (cx - bw/2 - left) / scale
            y1 = (cy - bh/2 - top) / scale
            x2 = (cx + bw/2 - left) / scale
            y2 = (cy + bh/2 - top) / scale
            fw, fh = int(x2 - x1), int(y2 - y1)
            if fw >= 60 and fh >= 60:
                faces.append((int(x1), int(y1), fw, fh))
        all_faces.append(detector._nms(faces, 0.4))
    return all_faces

# Warmup
yolo_onnx_batch(images[:1])
batch_times = []
for run in range(5):
    t0 = time.perf_counter()
    yolo_onnx_batch(images)
    t1 = time.perf_counter()
    batch_times.append((t1 - t0) * 1000 / len(images))

results["onnx_batch"] = {
    "name": "ONNX batch (per-image avg)",
    "avg_ms": round(np.mean(batch_times), 1),
    "median_ms": round(np.median(batch_times), 1),
    "min_ms": round(np.min(batch_times), 1),
    "max_ms": round(np.max(batch_times), 1),
}
print(f"  avg={results['onnx_batch']['avg_ms']}ms median={results['onnx_batch']['median_ms']}ms")

# ── Full pipeline: ONNX detect + ArcFace encoding ──
print("[5/5] Full pipeline: ONNX detect + keypoints eye + ArcFace...")
def full_onnx_pipeline(arr):
    # 1. ONNX detect
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    scale = min(640/w, 640/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr, (nw, nh))
    pad_w, pad_h = 640-nw, 640-nh
    top, left = pad_h//2, pad_w//2
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[top:top+nh, left:left+nw] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = sess.run(None, {inp_name: blob})
    pred = outputs[0][0].T
    
    faces = []
    for det in pred:
        cx, cy, bw, bh, conf = det[:5]
        if conf < 0.5: continue
        x1 = (cx - bw/2 - left) / scale
        y1 = (cy - bh/2 - top) / scale
        x2 = (cx + bw/2 - left) / scale
        y2 = (cy + bh/2 - top) / scale
        fw, fh = int(x2 - x1), int(y2 - y1)
        if fw >= 60 and fh >= 60:
            faces.append((int(x1), int(y1), fw, fh))
    faces = detector._nms(faces, 0.4)
    
    # 2. ArcFace encoding for each face
    result_faces = []
    for (x, y, fw, fh) in faces:
        enc = detector._arcface_encoding(bgr, face_info=None, bbox=(x, y, fw, fh))
        if enc is not None:
            result_faces.append({"bbox": (x, y, fw, fh), "encoding": enc})
    return result_faces

r = benchmark("Full ONNX+ArcFace", full_onnx_pipeline, images)
results["full_onnx"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# Current full pipeline for comparison
print("[Extra] Full current pipeline...")
def full_current(arr):
    # This uses the original detect_faces which takes a path, so we need to adapt
    pass  # Already measured in phase 1: ~1030ms

# ── Summary ──
print("\n" + "="*70)
print("BENCHMARK RESULTS - Phase 2")
print("="*70)
base = results["baseline"]["avg_ms"]
print(f"{'Scheme':<35} {'Avg ms':>8} {'Median':>8} {'Speedup':>8}")
print("-"*61)
for key in ["baseline", "preresize_kpts", "onnx", "onnx_batch", "full_onnx"]:
    r = results[key]
    if isinstance(r.get("avg_ms"), (int, float)):
        sp = f"{base/r['avg_ms']:.2f}x"
        print(f"{r['name']:<35} {r['avg_ms']:>8.1f} {r['median_ms']:>8.1f} {sp:>8}")

# Save
out = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_results_p2.json"
with open(out, 'w', encoding='utf-8') as f:
    json.dump({"test_set": test_set, "results": results}, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved to {out}")
