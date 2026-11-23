"""批量找"照片躺着"（无 EXIF 但画面整体旋转 90/180/270）的片子。

判据只有一条能在整图上快速算出来的：YuNet 置信度。
正常正立照片上，最大那张脸的 YuNet 置信度稳定在 0.91~0.95；照片躺着时
原方向只有 0.6~0.75，转正后才回到 0.9 以上。

用法：
  python tools/sideways_scan.py --folder "比特恢复" [--limit N] [--workers 4]
输出 data/_sideways_scan.json（候选清单）与 data/_sideways_scan.csv
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

DB = os.path.join(ROOT, "databases", "photo_index.db")
OUT_JSON = os.path.join(ROOT, "data", "_sideways_scan.json")

BASE_MIN = 0.85          # 原方向最高分低于此值才开始试四个方向
ROT_MIN = 0.86           # 转正后必须达到
ROT_GAIN = 0.08          # 且要比原方向高这么多


def _yunet():
    model = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")
    if not os.path.isfile(model):
        for dirpath, _, files in os.walk(os.path.join(ROOT, "models")):
            for f in files:
                if "yunet" in f.lower() and f.endswith(".onnx"):
                    model = os.path.join(dirpath, f)
                    break
    return model


_MODEL = _yunet()


def _max_score(img_bgr, det):
    """整图缩到长边 640 后取 YuNet 最高置信度。"""
    h, w = img_bgr.shape[:2]
    s = 640.0 / max(h, w)
    small = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else img_bgr
    sh, sw = small.shape[:2]
    det.setInputSize((sw, sh))
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        return 0.0
    # faces: x, y, w, h, 5 landmarks(10), score
    return float(max(f[-1] for f in faces))


def scan_one(args):
    path, = args
    model, = (_MODEL,)
    try:
        det = cv2.FaceDetectorYN.create(model, "", (640, 640), 0.5, 0.3, 5000)
        from PIL import Image, ImageOps
        im = Image.open(path)
        im.draft("RGB", (800, 800))      # JPEG 快速降采样解码，只判方向够用
        im = ImageOps.exif_transpose(im).convert("RGB")
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        base = _max_score(bgr, det)
        if base >= BASE_MIN:
            return {"path": path, "base": round(base, 3), "best": None, "deg": 0}
        best, best_deg = base, 0
        for deg in (90, 180, 270):
            if deg == 90:
                rot = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif deg == 180:
                rot = cv2.rotate(bgr, cv2.ROTATE_180)
            else:
                rot = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            sc = _max_score(rot, det)
            if sc > best:
                best, best_deg = sc, deg
        return {"path": path, "base": round(base, 3), "best": round(best, 3), "deg": best_deg}
    except Exception as e:
        return {"path": path, "base": None, "best": None, "deg": 0, "err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None, help="路径包含该子串的照片（默认全库）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    q = "SELECT path FROM photos"
    rows = [r[0] for r in con.execute(q)]
    if args.folder:
        rows = [p for p in rows if args.folder in p]
    if args.limit:
        rows = rows[: args.limit]
    print(f"待扫描 {len(rows)} 张 | folder={args.folder} | workers={args.workers}", flush=True)

    t0 = time.time()
    results = []
    step = max(1, len(rows) // 40)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(scan_one, [(p,) for p in rows], chunksize=8), 1):
            results.append(r)
            if i % step == 0:
                el = time.time() - t0
                print(f"  {i}/{len(rows)}  {el:.0f}s  预计剩余 {el/i*(len(rows)-i):.0f}s", flush=True)

    cands = [r for r in results
             if r.get("best") is not None and r["best"] >= ROT_MIN
             and r["best"] >= r["base"] + ROT_GAIN]
    cands.sort(key=lambda r: -(r["best"] - r["base"]))
    print(f"扫完：{len(results)} 张，候选 {len(cands)} 张，耗时 {time.time()-t0:.0f}s")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"candidates": cands, "total": len(results),
                   "base_min": BASE_MIN, "rot_min": ROT_MIN, "rot_gain": ROT_GAIN},
                  fh, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "data", "_sideways_scan.csv"), "w", encoding="utf-8-sig") as fh:
        fh.write("文件名,原方向分,转正后分,需转(顺时针),路径\n")
        for r in cands:
            fh.write(f"{os.path.basename(r['path'])},{r['base']},{r['best']},"
                     f"{r['deg']},{r['path']}\n")
    for r in cands[:25]:
        print(f"  {os.path.basename(r['path']):16} {r['base']:.3f} -> {r['best']:.3f}  转 {r['deg']}°")
    print(f"清单: {OUT_JSON}")


if __name__ == "__main__":
    main()
