"""
Benchmark Phase 3: Clean comparison with proper warmup
Focus: ONNX with cv2.dnn (which is already optimized for CPU) vs ultralytics vs onnxruntime
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

# Load images
images = []
bgr_images = []
for p in test_set:
    arr = np.array(Image.open(p["path"]).convert('RGB'))
    images.append(arr)
    bgr_images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
print(f"Loaded {len(images)} images")

def benchmark(name, fn, inputs, n_runs=5, warmup=3):
    for inp in inputs[:1]:
        fn(inp)
    # Extra warmup
    for _ in range(warmup):
        for inp in inputs:
            fn(inp)
    
    all_times = []
    for run in range(n_runs):
        for inp in inputs:
            t0 = time.perf_counter()
            fn(inp)
            t1 = time.perf_counter()
            all_times.append((t1 - t0) * 1000)
    return {
        "name": name,
        "avg_ms": round(np.mean(all_times), 1),
        "median_ms": round(np.median(all_times), 1),
        "min_ms": round(np.min(all_times), 1),
        "max_ms": round(np.max(all_times), 1),
    }

results = {}

# ── 1. Baseline: ultralytics _detect_yolo ──
from core.face_detector import FaceDetector
detector = FaceDetector(yunet_threshold=0.5, yunet_nms_threshold=0.3,
                         yunet_min_face=60, eye_validation=True, strict_mode=True,
                         opencv_dnn_conf=0.12, opencv_dnn_min_face=30,
                         prefer_opencv_dnn=False, yolo_conf=0.5, fallback_dnn_conf=0.12)
detector._load_yolo()

print("[1/7] Baseline: ultralytics YOLO...")
r = benchmark("Baseline ultralytics", lambda arr: detector._detect_yolo(arr), images)
results["baseline"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── 2. Ultralytics + imgsz=640 + pre-resize ──
print("[2/7] Ultralytics + pre-resize + imgsz=640...")
def yolo_preresize(arr):
    h, w = arr.shape[:2]
    scale = min(640/w, 640/h)
    if scale < 1.0:
        small = cv2.resize(arr, (int(w*scale), int(h*scale)))
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        scale = 1.0
    rs = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640)
    faces = []
    for r in rs:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = int(x1/scale), int(y1/scale)
            bw, bh = int((x2-x1)/scale), int((y2-y1)/scale)
            if bw >= 60 and bh >= 60:
                faces.append((x, y, bw, bh))
    return detector._nms(faces, 0.4)

r = benchmark("Ultralytics pre-resize", yolo_preresize, images)
results["ultra_preresize"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── 3. ONNX Runtime (letterbox with numpy) ──
print("[3/7] ONNX Runtime (letterbox numpy)...")
onnx_path = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\models\yolov8n-face\face_yolov8n.onnx"
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

def yolo_onnx_letterbox(arr):
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
    pred = outputs[0][0].T  # (8400, 5)
    
    faces = []
    for det in pred:
        cx, cy, bw, bh, conf = det[:5]
        if conf < 0.5: continue
        x1 = (cx - bw/2 - left) / scale
        y1 = (cy - bh/2 - top) / scale
        x2 = (cx + bw/2 - left) / scale
        y2 = (cy + bh/2 - top) / scale
        fw, fh = int(x2-x1), int(y2-y1)
        if fw >= 60 and fh >= 60:
            faces.append((int(x1), int(y1), fw, fh))
    return detector._nms(faces, 0.4)

r = benchmark("ONNX Runtime letterbox", yolo_onnx_letterbox, images)
results["onnx_letterbox"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── 4. ONNX Runtime (pre-resize, simpler) ──
print("[4/7] ONNX Runtime (simple resize, no letterbox)...")
def yolo_onnx_simple(arr):
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    scale = min(640/w, 640/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr, (nw, nh))
    # Just pad to square
    if nw != 640 or nh != 640:
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
    else:
        padded = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    
    outputs = sess.run(None, {inp_name: blob})
    pred = outputs[0][0].T
    
    faces = []
    for det in pred:
        cx, cy, bw, bh, conf = det[:5]
        if conf < 0.5: continue
        # Coordinates in padded space, convert back
        x1 = (cx - bw/2) / scale
        y1 = (cy - bh/2) / scale
        x2 = (cx + bw/2) / scale
        y2 = (cy + bh/2) / scale
        fw, fh = int(x2-x1), int(y2-y1)
        if fw >= 60 and fh >= 60:
            faces.append((int(x1), int(y1), fw, fh))
    return detector._nms(faces, 0.4)

r = benchmark("ONNX simple resize", yolo_onnx_simple, images)
results["onnx_simple"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── 5. cv2.dnn with ONNX model ──
print("[5/7] cv2.dnn with ONNX model...")
net = cv2.dnn.readNetFromONNX(onnx_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def yolo_cv2dnn(arr):
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    scale = min(640/w, 640/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr, (nw, nh))
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    
    blob = cv2.dnn.blobFromImage(padded, 1.0/255.0, (640, 640), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()  # (1, 5, 8400)
    
    pred = output[0].T
    
    faces = []
    for det in pred:
        cx, cy, bw, bh, conf = det[:5]
        if conf < 0.5: continue
        x1 = (cx - bw/2) / scale
        y1 = (cy - bh/2) / scale
        x2 = (cx + bw/2) / scale
        y2 = (cy + bh/2) / scale
        fw, fh = int(x2-x1), int(y2-y1)
        if fw >= 60 and fh >= 60:
            faces.append((int(x1), int(y1), fw, fh))
    return detector._nms(faces, 0.4)

r = benchmark("cv2.dnn ONNX", yolo_cv2dnn, images)
results["cv2dnn_onnx"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── 6. Eye validation: Haar vs YOLO keypoints ──
print("[6/7] Eye validation comparison...")

# Haar eye validation
def eye_haar_test(arr):
    faces = detector._detect_yolo(arr)
    if faces:
        return detector._eye_validate(arr, faces, [])
    return [], []
r = benchmark("Eye Haar", eye_haar_test, images)
results["eye_haar"] = r
print(f"  Haar: avg={r['avg_ms']}ms")

# YOLO keypoints (using ultralytics result)
def eye_yolo_kpts(arr):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rs = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640)
    faces = []
    for r in rs:
        if r.boxes is None: continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = int(x2-x1), int(y2-y1)
            if w < 60 or h < 60: continue
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
                faces.append((int(x1), int(y1), w, h))
    return detector._nms(faces, 0.4)
r = benchmark("Eye YOLO kpts", eye_yolo_kpts, images)
results["eye_yolo_kpts"] = r
print(f"  YOLO kpts: avg={r['avg_ms']}ms")

# ── 7. ArcFace encoding per face ──
print("[7/7] ArcFace encoding timing...")
def arcface_test(arr):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    faces = detector._detect_yolo(arr)
    encodings = []
    for (x, y, w, h) in faces:
        enc = detector._arcface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
        if enc: encodings.append(enc)
    return encodings

r = benchmark("YOLO + ArcFace", arcface_test, images)
results["yolo_arcface"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ArcFace only (faces already detected)
print("  ArcFace only (per-face)...")
face_per_img = []
arcface_times = []
for arr in images:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    faces = detector._detect_yolo(arr)
    face_per_img.append(len(faces))
    for (x, y, w, h) in faces:
        t0 = time.perf_counter()
        detector._arcface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
        arcface_times.append((time.perf_counter() - t0) * 1000)

if arcface_times:
    print(f"  ArcFace per face: avg={np.mean(arcface_times):.1f}ms, total faces={len(arcface_times)}")
    results["arcface_per_face"] = {
        "name": "ArcFace per face",
        "avg_ms": round(np.mean(arcface_times), 1),
        "faces_tested": len(arcface_times),
    }

# ── Summary ──
print("\n" + "="*70)
print("FINAL BENCHMARK RESULTS")
print("="*70)
base = results["baseline"]["avg_ms"]
print(f"{'Scheme':<35} {'Avg ms':>8} {'Median':>8} {'Speedup':>8}")
print("-"*61)
for key in ["baseline", "ultra_preresize", "onnx_letterbox", "onnx_simple", "cv2dnn_onnx",
            "eye_haar", "eye_yolo_kpts", "yolo_arcface"]:
    r = results[key]
    if isinstance(r.get("avg_ms"), (int, float)):
        sp = f"{base/r['avg_ms']:.2f}x"
        print(f"{r['name']:<35} {r['avg_ms']:>8.1f} {r['median_ms']:>8.1f} {sp:>8}")

if "arcface_per_face" in results:
    print(f"\nArcFace per face: {results['arcface_per_face']['avg_ms']:.1f}ms")

out = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_results_p3.json"
with open(out, 'w', encoding='utf-8') as f:
    json.dump({"test_set": test_set, "results": results}, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved to {out}")
