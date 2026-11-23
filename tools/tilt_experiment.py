# -*- coding: utf-8 -*-
"""倾斜角 -> 人脸检测置信度 实验（只读，不写库）

用法: python tools/tilt_experiment.py
对若干张"像素本身已是正脸"的照片做整体旋转，观察 YOLO 最大置信度如何衰减。
"""
import sys, os, json, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from core.face_detector import FaceDetector
from core.image_utils import open_upright, get_orientation

ANGLES = [0, 10, 20, 30, 45, 60, 75, 90]


def yolo_conf(det, bgr):
    """直接调用 YOLO 取原始置信度（_detect_yolo 只返回框，丢弃了 conf）"""
    det._load_yolo()
    if det._yolo_model is None:
        return 0.0, 0
    h, w = bgr.shape[:2]
    scale = min(640 / w, 640 / h, 1.0)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
    res = det._yolo_model(small, verbose=False, conf=0.05, imgsz=640)
    confs = []
    for r in res:
        if r.boxes is None:
            continue
        for c in r.boxes.conf.cpu().numpy():
            confs.append(float(c))
    confs.sort(reverse=True)
    return (confs[0] if confs else 0.0), len([c for c in confs if c >= 0.5])


def rotate(img, deg):
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    det = FaceDetector()
    con = sqlite3.connect(ROOT / "databases" / "photo_index.db")
    rows = con.execute("select path, faces from photos").fetchall()
    picks = []
    for p, f in rows:
        try:
            n = len(json.loads(f or "[]"))
        except Exception:
            continue
        if n in (1, 2) and "2024春节古装照片" in p and get_orientation(p) in (0, 1, None):
            picks.append(p)
    # 均匀取样，避免全是同一场景
    step = max(1, len(picks) // 8)
    picks = picks[::step][:8]

    print(f"样本 {len(picks)} 张，正脸像素基准（EXIF 正常）")
    print("path".ljust(46), *[f"{a:>5}°" for a in ANGLES])
    summary = {a: [] for a in ANGLES}
    for p in picks:
        img, _, _ = open_upright(p)
        arr0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 已是 BGR 输入
        row = []
        for a in ANGLES:
            arr = arr0 if a == 0 else rotate(arr0, a)
            conf, n_ok = yolo_conf(det, arr)
            row.append((conf, n_ok))
            summary[a].append(conf)
        name = Path(p).name
        print(name.ljust(46), *[f"{c:.2f}/{n}" for c, n in row])

    print("\n角度  | 平均最大conf | 中位 | ≥0.5 命中张数")
    for a in ANGLES:
        v = np.array(summary[a], dtype=float)
        print(f"{a:>4}° | {v.mean():11.3f} | {np.median(v):4.2f} | {int((v>=0.5).sum())}/{len(v)}")


if __name__ == "__main__":
    main()
