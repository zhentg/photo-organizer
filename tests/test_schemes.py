"""
Face Detection Scheme Comparison Test
Test 3 schemes against all test_reference.md images, with timing.
"""
import sys, os, cv2, numpy as np, time as time_mod
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.face_detector import FaceDetector

# ── Test Images ──
test_images = [
    (
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '10004c0.jpg'),
        4,
        'multi-face small/side-angle'
    ),
    (
        r'D:\photoes\Nikon相机照片\比特恢复的文件\02 本地磁盘 (E ) 删除文件  (5222 个文件)\Dir00310742\DSCN9691.JPG',
        1,
        'close-up portrait'
    ),
]

SEP = '=' * 75


def _iou(box_a, box_b):
    """Compute IoU between two (x,y,w,h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def test_scheme_a(img_path, expected):
    """Scheme A: OpenCV DNN with low conf threshold (0.12) + eye validation"""
    print(f'--- Scheme A: OpenCV DNN conf=0.12 + eye_validation ---')
    try:
        fd = FaceDetector(
            prefer_opencv_dnn=True,
            opencv_dnn_conf=0.12,
            opencv_dnn_min_face=30,
            eye_validation=True,
        )
        t0 = time_mod.perf_counter()
        faces = fd.detect_faces(img_path)
        elapsed = time_mod.perf_counter() - t0
        n = len(faces)
        status = 'PASS' if n == expected else ('OVER' if n > expected else 'UNDER')
        print(f'Result: {n} face(s)  [{status}]  ({elapsed:.2f}s)')
        for i, f in enumerate(faces):
            print(f'  [{i+1}] bbox={f["bbox"]}')
        return n, status, elapsed
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'ERROR: {e}')
        return -1, 'ERROR', 0.0


def test_scheme_b(img_path, expected):
    """Scheme B: OpenCV DNN (conf=0.5) first, YOLO supplement if < 2 faces"""
    print(f'--- Scheme B: OpenCV DNN(conf=0.5) + YOLO supplement (<2) ---')
    try:
        fd = FaceDetector(
            prefer_opencv_dnn=True,
            opencv_dnn_conf=0.5,
            opencv_dnn_min_face=30,
            eye_validation=True,
        )
        t0 = time_mod.perf_counter()
        faces = fd.detect_faces(img_path)
        n_initial = len(faces)

        added = 0
        if n_initial < 2 and fd._yolo_available:
            fd._load_yolo()
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            yolo_results = fd._yolo_model(bgr, verbose=False, conf=0.3)
            yolo_bboxes = []
            for r in yolo_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w >= 30 and h >= 30:
                        yolo_bboxes.append((int(x1), int(y1), w, h))

            existing = set(tuple(f['bbox']) for f in faces)
            for yb in yolo_bboxes:
                is_dup = any(_iou(yb, eb) > 0.4 for eb in existing)
                if not is_dup:
                    enc = fd._arcface_encoding(bgr, face_info=None, bbox=yb)
                    if enc is None:
                        enc = fd._sface_encoding(bgr, face_info=None, bbox=yb)
                    if enc is not None:
                        faces.append({
                            'location': (yb[1], yb[0] + yb[2], yb[1] + yb[3], yb[0]),
                            'encoding': enc,
                            'image_path': img_path,
                            'bbox': yb,
                        })
                        existing.add(yb)
                        added += 1

            if added:
                print(f'  (OpenCV DNN found {n_initial}, YOLO added {added})')

        elapsed = time_mod.perf_counter() - t0
        n = len(faces)
        status = 'PASS' if n == expected else ('OVER' if n > expected else 'UNDER')
        print(f'Result: {n} face(s)  [{status}]  ({elapsed:.2f}s)')
        for i, f in enumerate(faces):
            print(f'  [{i+1}] bbox={f["bbox"]}')
        return n, status, elapsed
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'ERROR: {e}')
        return -1, 'ERROR', 0.0


def test_scheme_c(img_path, expected):
    """Scheme C: YOLOv8n-Face as primary detector (conf=0.5)"""
    print(f'--- Scheme C: YOLOv8n-Face primary (conf=0.5) ---')
    try:
        fd = FaceDetector(prefer_opencv_dnn=False, eye_validation=True)
        fd._load_yolo()
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        t0 = time_mod.perf_counter()
        yolo_results = fd._yolo_model(bgr, verbose=False, conf=0.5)
        faces = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = int(x2 - x1), int(y2 - y1)
                if w >= 30 and h >= 30:
                    bbox = (int(x1), int(y1), w, h)
                    enc = fd._arcface_encoding(bgr, face_info=None, bbox=bbox)
                    if enc is None:
                        enc = fd._sface_encoding(bgr, face_info=None, bbox=bbox)
                    if enc is not None:
                        faces.append({
                            'location': (int(y1), int(x2), int(y2), int(x1)),
                            'encoding': enc,
                            'image_path': img_path,
                            'bbox': bbox,
                        })
        elapsed = time_mod.perf_counter() - t0

        n = len(faces)
        status = 'PASS' if n == expected else ('OVER' if n > expected else 'UNDER')
        print(f'Result: {n} face(s)  [{status}]  ({elapsed:.2f}s)')
        for i, f in enumerate(faces):
            print(f'  [{i+1}] bbox={f["bbox"]}')
        return n, status, elapsed
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'ERROR: {e}')
        return -1, 'ERROR', 0.0


# ── Main ──
if __name__ == '__main__':
    print(SEP)
    print('Face Detection Scheme Comparison Test  (with timing)')
    print(SEP)

    results_summary = []

    for img_path, expected, desc in test_images:
        if not os.path.exists(img_path):
            print(f'\n[SKIP] not found: {img_path}')
            continue

        name = os.path.basename(img_path)
        print(f'\n{SEP}')
        print(f'Image: {name}  ({desc})')
        print(f'Expected: {expected} face(s)')
        print(SEP)

        n_a, s_a, t_a = test_scheme_a(img_path, expected)
        print()
        n_b, s_b, t_b = test_scheme_b(img_path, expected)
        print()
        n_c, s_c, t_c = test_scheme_c(img_path, expected)

        results_summary.append({
            'image': name,
            'expected': expected,
            'A': (n_a, s_a, t_a),
            'B': (n_b, s_b, t_b),
            'C': (n_c, s_c, t_c),
        })

    # ── Final Summary ──
    print(f'\n{SEP}')
    print('FINAL SUMMARY')
    print(SEP)
    print(f'{"Image":<22s}  {"Exp":>3s}   {"A_det":>5s}  {"A_stat":>5s}  {"A_time":>7s}   {"B_det":>5s}  {"B_stat":>5s}  {"B_time":>7s}   {"C_det":>5s}  {"C_stat":>5s}  {"C_time":>7s}')
    print('-' * 105)
    for r in results_summary:
        a_s = 'OK' if r['A'][1] == 'PASS' else r['A'][1]
        b_s = 'OK' if r['B'][1] == 'PASS' else r['B'][1]
        c_s = 'OK' if r['C'][1] == 'PASS' else r['C'][1]
        print(f'{r["image"]:<22s}  {r["expected"]:>3d}   '
              f'{r["A"][0]:>5d}({a_s:>4s})  {r["A"][2]:>6.2f}s   '
              f'{r["B"][0]:>5d}({b_s:>4s})  {r["B"][2]:>6.2f}s   '
              f'{r["C"][0]:>5d}({c_s:>4s})  {r["C"][2]:>6.2f}s')

    print('-' * 105)
    total = len(results_summary)
    if total:
        a_pass = sum(1 for r in results_summary if r['A'][1] == 'PASS')
        b_pass = sum(1 for r in results_summary if r['B'][1] == 'PASS')
        c_pass = sum(1 for r in results_summary if r['C'][1] == 'PASS')
        a_avg = sum(r['A'][2] for r in results_summary) / total
        b_avg = sum(r['B'][2] for r in results_summary) / total
        c_avg = sum(r['C'][2] for r in results_summary) / total
        print(f'Pass rate: A={a_pass}/{total}  B={b_pass}/{total}  C={c_pass}/{total}')
        print(f'Avg time:  A={a_avg:.2f}s  B={b_avg:.2f}s  C={c_avg:.2f}s')

    print('Done.')
