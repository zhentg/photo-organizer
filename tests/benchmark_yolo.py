"""
Benchmark YOLO optimization schemes - Phase 1
Schemes that don't require new model exports
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"D:\CADD\workspace\qclaw\projects\face-photo-sort")
os.chdir(r"D:\CADD\workspace\qclaw\projects\face-photo-sort")

import cv2
import numpy as np
from PIL import Image

with open(r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_set.json", 'r', encoding='utf-8') as f:
    test_set = json.load(f)

print(f"Test set: {len(test_set)} photos")
for i, p in enumerate(test_set):
    print(f"  {i+1}. {os.path.basename(p['path'])[:20]:20s} | {p['size_kb']:>6}KB | {p['width']}x{p['height']}")

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def benchmark(name, fn, images, n_runs=5):
    # Warmup
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
        "per_size": {},  # filled below
    }

# Load images
print("\nLoading images...")
images = []
image_sizes = []
for p in test_set:
    arr = load_image(p["path"])
    images.append(arr)
    image_sizes.append(f"{p['width']}x{p['height']}")
print(f"Loaded {len(images)} images")

# Init detector
from core.face_detector import FaceDetector
detector = FaceDetector(yunet_threshold=0.5, yunet_nms_threshold=0.3,
                         yunet_min_face=60, eye_validation=True, strict_mode=True,
                         opencv_dnn_conf=0.12, opencv_dnn_min_face=30,
                         prefer_opencv_dnn=False, yolo_conf=0.5, fallback_dnn_conf=0.12)
detector._load_yolo()

results = {}

# ── Baseline ──
print("\n[1/6] Baseline: ultralytics default...")
r = benchmark("Baseline", lambda arr: detector._detect_yolo(arr), images)
results["baseline"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── A: imgsz=640 ──
print("[2/6] Scheme A: imgsz=640 explicit...")
def yolo_640(arr):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    results = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640)
    faces = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = int(x2-x1), int(y2-y1)
            if w >= 60 and h >= 60:
                faces.append((int(x1), int(y1), w, h))
    return detector._nms(faces, 0.4)
r = benchmark("imgsz=640", yolo_640, images)
results["imgsz640"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── A2: Pre-resize ──
print("[3/6] Scheme A2: pre-resize to 640...")
def yolo_preresize(arr):
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
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x, y = int(x1/scale), int(y1/scale)
            bw, bh = int((x2-x1)/scale), int((y2-y1)/scale)
            if bw >= 60 and bh >= 60:
                faces.append((x, y, bw, bh))
    return detector._nms(faces, 0.4)
r = benchmark("pre-resize 640", yolo_preresize, images)
results["preresize"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── A3: half precision (FP16) ──
print("[4/6] Scheme A3: YOLO FP16 on CPU...")
try:
    # On CPU, half() may not help much but worth testing
    original_model = detector._yolo_model
    detector._yolo_model = original_model  # keep reference
    
    def yolo_fp16(arr):
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        rs = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640, half=True)
        faces = []
        for r in rs:
            if r.boxes is None: continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = int(x2-x1), int(y2-y1)
                if w >= 60 and h >= 60:
                    faces.append((int(x1), int(y1), w, h))
        return detector._nms(faces, 0.4)
    r = benchmark("FP16", yolo_fp16, images)
    results["fp16"] = r
    print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")
except Exception as e:
    print(f"  Skipped: {e}")
    results["fp16"] = {"name": "FP16", "avg_ms": "N/A", "median_ms": "N/A"}

# ── E: YOLO keypoints eye validation ──
print("[5/6] Scheme E: YOLO keypoints for eye validation...")
def yolo_keypoint_eye(arr):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rs = detector._yolo_model(bgr, verbose=False, conf=0.5, imgsz=640)
    faces = []
    for r in rs:
        if r.boxes is None: continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = int(x2-x1), int(y2-y1)
            if w < 60 or h < 60: continue
            # Check keypoints for eyes
            if r.keypoints is not None and r.keypoints.xy is not None:
                kpts = r.keypoints.xy[i].cpu().numpy() if i < len(r.keypoints.xy) else None
                if kpts is not None and len(kpts) >= 2:
                    # keypoints 0=left_eye, 1=right_eye for YOLO-face
                    le, re = kpts[0], kpts[1]
                    # Eyes should be above center of face and separated
                    has_eyes = (le[1] < (y1 + y2)/2) and (re[1] < (y1 + y2)/2) and (abs(le[0]-re[0]) > w*0.1)
                else:
                    has_eyes = True
            else:
                has_eyes = True
            if has_eyes:
                faces.append((int(x1), int(y1), w, h))
    return detector._nms(faces, 0.4)
r = benchmark("YOLO keypoints eye", yolo_keypoint_eye, images)
results["keypoints_eye"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── E2: Combined best: pre-resize + keypoints ──
print("[6/6] Scheme E2: pre-resize + YOLO keypoints eye validation...")
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
            if r.keypoints is not None and r.keypoints.xy is not None:
                kpts = r.keypoints.xy[i].cpu().numpy() if i < len(r.keypoints.xy) else None
                if kpts is not None and len(kpts) >= 2:
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

# ── Eye validation timing ──
print("\n[Extra] Eye validation (Haar) timing...")
def eye_haar(arr):
    faces = detector._detect_yolo(arr)
    if faces:
        return detector._eye_validate(arr, faces, [])
    return [], []
r = benchmark("eye Haar", eye_haar, images)
results["eye_haar"] = r
print(f"  avg={r['avg_ms']}ms median={r['median_ms']}ms")

# ── Full detect_faces comparison ──
print("\n[Extra] Full detect_faces() with different configs...")

# Current: YOLO + Haar eye validation + ArcFace
r_full_current = benchmark("full current", lambda arr: detector.detect_faces(test_set[0]["path"]), images)
results["full_current"] = r_full_current
print(f"  Current: avg={r_full_current['avg_ms']}ms")

# ── Summary ──
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)
base = results["baseline"]["avg_ms"]
print(f"{'Scheme':<30} {'Avg ms':>8} {'Median':>8} {'Speedup':>8}")
print("-"*56)
for key in ["baseline", "imgsz640", "preresize", "fp16", "keypoints_eye", "preresize_kpts", "eye_haar", "full_current"]:
    r = results[key]
    if isinstance(r.get("avg_ms"), (int, float)):
        sp = f"{base/r['avg_ms']:.2f}x"
        print(f"{r['name']:<30} {r['avg_ms']:>8.1f} {r['median_ms']:>8.1f} {sp:>8}")
    else:
        print(f"{r['name']:<30} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

# Save
out = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_results.json"
with open(out, 'w', encoding='utf-8') as f:
    json.dump({"test_set": test_set, "results": results}, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved to {out}")
