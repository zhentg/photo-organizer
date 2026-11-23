# -*- coding: utf-8 -*-
"""倾斜角 -> 检测 + ArcFace 识别 实验（只读，写临时图到 data/_tilt_exp）

回答：把照片转正，到底是为了"检测得到"还是"认得出人"。
用法: python tools/tilt_experiment2.py
"""
import sys, json, sqlite3, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from core.face_detector import FaceDetector, FaceMatcher
from core.image_utils import open_upright, get_orientation

ANGLES = [0, 15, 30, 45, 60, 90]
TMP = ROOT / "data" / "_tilt_exp"


def rotate_expand(bgr, deg):
    h, w = bgr.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    det = FaceDetector()
    matcher = FaceMatcher(str(ROOT / "databases" / "face_db.db"))
    con = sqlite3.connect(ROOT / "databases" / "photo_index.db")
    rows = con.execute("select path, faces, persons from photos").fetchall()
    picks = []
    for p, f, pe in rows:
        try:
            n = len(json.loads(f or "[]"))
            names = json.loads(pe or "[]")
        except Exception:
            continue
        if n == 1 and names and names[0] != "陌生人" and "2024春节古装照片" in p \
                and get_orientation(p) in (0, 1, None):
            picks.append((p, names[0]))
    picks = picks[:: max(1, len(picks) // 6)][:6]
    TMP.mkdir(parents=True, exist_ok=True)

    print(f"样本 {len(picks)} 张（单真人、已命名、EXIF 正常）")
    print(f"{'照片':<14}{'基准姓名':<10}" + "".join(f"{a:>16}°" for a in ANGLES[1:]))
    stats = {a: {"d": [], "hit": 0, "n": 0} for a in ANGLES[1:]}
    for p, name in picks:
        img, _, _ = open_upright(p)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        base_path = TMP / f"base_{Path(p).stem}.jpg"
        cv2.imwrite(str(base_path), bgr)
        f0 = det.detect_faces(str(base_path))
        if not f0:
            print(Path(p).name, "基准检测失败，跳过")
            continue
        base = max(f0, key=lambda x: x["bbox"][2] * x["bbox"][3])["encoding"]
        cells = []
        for a in ANGLES[1:]:
            fp = TMP / f"{Path(p).stem}_{a}.jpg"
            cv2.imwrite(str(fp), rotate_expand(bgr, a))
            fs = det.detect_faces(str(fp))
            if not fs:
                cells.append("未检测")
                continue
            big = max(fs, key=lambda x: x["bbox"][2] * x["bbox"][3])
            d = matcher.face_distance(base, big["encoding"])
            who, _ = matcher.find_match(big["encoding"])
            stats[a]["d"].append(d)
            stats[a]["n"] += 1
            if who == name:
                stats[a]["hit"] += 1
            cells.append(f"{d:.2f}{'✓' if who == name else '✗'}")
        print(f"{Path(p).name:<14}{name:<10}" + "".join(f"{c:>17}" for c in cells))

    print("\n角度 | 平均距离(越小越像本人) | 认对人脸 | 检测到")
    for a in ANGLES[1:]:
        s = stats[a]
        dm = np.mean(s["d"]) if s["d"] else float("nan")
        print(f"{a:>3}° | {dm:20.3f} | {s['hit']}/{len(picks)} | {s['n']}/{len(picks)}")
    print("\n（本项目 find_match 阈值见 core 常量；距离 > 阈值 即被判成陌生人）")

    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
