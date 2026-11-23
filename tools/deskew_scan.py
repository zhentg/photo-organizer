# -*- coding: utf-8 -*-
"""倾斜度普查：用 YuNet 双眼连线估算照片内人脸的倾斜角（只读，不写库）

用法: python tools/deskew_scan.py [样本数] [--all]
默认只抽样"库里已有人脸"的照片；加 --all 则全库随机抽。
"""
import sys, json, random, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from core.face_detector import FaceDetector
from core.image_utils import open_upright, get_orientation

BINS = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 45), (45, 91)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    con = sqlite3.connect(ROOT / "databases" / "photo_index.db")
    rows = con.execute("select path, faces from photos").fetchall()
    with_faces = [p for p, f in rows if json.loads(f or "[]")]
    pool = with_faces if "--all" not in sys.argv else [p for p, _ in rows]
    random.seed(20261019)
    picks = random.sample(pool, min(n, len(pool)))

    det = FaceDetector()
    angles, details = [], []
    skipped_rot = 0
    for i, p in enumerate(picks):
        try:
            if get_orientation(p) not in (0, 1, None):
                skipped_rot += 1
                continue
            img, _, _ = open_upright(p)
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = arr.shape[:2]
            s = min(800 / max(w, h), 1.0)
            if s < 1.0:
                arr = cv2.resize(arr, (int(w * s), int(h * s)))
            _, infos = det._detect_yunet(arr)
            if not infos:
                continue
            mx = 0.0
            for f in infos:
                rx, ry, lx, ly = f[4], f[5], f[6], f[7]
                a = abs(np.degrees(np.arctan2(ly - ry, lx - rx)))
                if a > 90:
                    a = 180 - a
                mx = max(mx, a)
            angles.append(mx)
            details.append((mx, p))
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(picks)}", flush=True)

    angles = np.array(angles)
    print(f"\n扫描 {len(picks)} 张（跳过 EXIF 横持 {skipped_rot} 张），YuNet 检出人脸 {len(angles)} 张")
    if len(angles) == 0:
        return
    print(f"倾斜角: 均值 {angles.mean():.1f}°  中位 {np.median(angles):.1f}°  "
          f"P90 {np.percentile(angles,90):.1f}°  最大 {angles.max():.1f}°")
    print("\n分布：")
    for lo, hi in BINS:
        k = int(((angles >= lo) & (angles < hi)).sum())
        print(f"  {lo:>2}-{hi:<2}° : {k:>4} 张  ({k/len(angles)*100:5.1f}%)")
        if lo >= 15:
            pass
    bad = sorted(details, reverse=True)[:15]
    print("\n倾斜最大的 15 张：")
    for a, p in bad:
        print(f"  {a:5.1f}°  {p}")


if __name__ == "__main__":
    main()
