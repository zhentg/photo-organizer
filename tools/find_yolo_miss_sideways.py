"""找出"YOLO 漏检 → 闸门没开四向探索"的侧躺照片（2026-10-20）。

背景：101bcc0.jpg 侧躺，YOLO+DNN 全漏，YuNet 只在草地捡到 62×68 的竖框，
"横框"闸门不触发 → 整张照片被判成 1 张脸、没有 roll。
修好闸门（backend != yolo 也算可疑）后，全库里同类照片有多少还不清楚。

本工具两段式，只做检测、不写库：
  A. 快筛：每张（索引里没有 roll 的）照片只跑 YOLO。YOLO 命中 → 闸门无关，跳过；
  B. 确认：YOLO 漏检的那些跑完整 detect_faces，和索引对比，输出"真的变了"的清单。

用法：
  python tools/find_yolo_miss_sideways.py [--workers 6] [--limit N]
输出：data/_yolo_miss_sideways.json + data/_yolo_miss_fix.txt（可喂给
      tools/batch_repair.py / tools/rewrite_sideways.py）
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

DB = os.path.join(ROOT, "databases", "photo_index.db")
REPORT = os.path.join(ROOT, "data", "_yolo_miss_sideways.json")
FIX_LIST = os.path.join(ROOT, "data", "_yolo_miss_fix.txt")

_DET = None


def _get_det():
    global _DET
    if _DET is None:
        from core.face_detector import FaceDetector
        _DET = FaceDetector(deskew=False)   # 只要 YOLO，关掉后续摆正逻辑
    return _DET


def _yolo_count(path):
    import cv2
    import numpy as np
    from core.image_utils import open_upright
    try:
        img, _o, _s = open_upright(path)
        arr = np.array(img)
        det = _get_det()
        boxes = det._detect_yolo(arr)
        return {"path": path, "yolo": len(boxes), "err": None}
    except Exception as e:
        return {"path": path, "yolo": -1, "err": repr(e)}


def _full(path):
    det = _get_det()
    det.deskew = True
    try:
        faces = det.detect_faces(path)
        return {"path": path,
                "n": len(faces),
                "rolls": [f.get("roll") for f in faces],
                "boxes": [[int(v) for v in f["bbox"]] for f in faces]}
    except Exception as e:
        return {"path": path, "n": -1, "err": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-zero-face", action="store_true",
                    help="连索引里 0 张脸的照片一起扫（5xx 张，慢且噪声大，默认跳过）")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    rows = con.execute("SELECT path, faces, persons FROM photos").fetchall()
    todo, old = [], {}
    for p, fj, pj in rows:
        fs = json.loads(fj or "[]")
        old[p] = (len(fs), bool(json.loads(pj or "[]")))
        if any(f.get("roll") for f in fs):
            continue
        if not fs and not args.include_zero_face:
            continue
        if not os.path.exists(p):
            continue
        todo.append(p)
    if args.limit:
        todo = todo[: args.limit]
    print(f"索引里无 roll 的照片 {len(todo)} 张 | workers={args.workers}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = []
        for i, r in enumerate(ex.map(_yolo_count, todo, chunksize=8), 1):
            res.append(r)
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  快筛 {i}/{len(todo)} {el:.0f}s", flush=True)

    miss = [r["path"] for r in res if r["yolo"] == 0]
    errs = [r for r in res if r["yolo"] == -1]
    print(f"YOLO 漏检 {len(miss)} 张（出错 {len(errs)}）→ 跑完整检测确认", flush=True)

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        full = []
        for i, r in enumerate(ex.map(_full, miss, chunksize=4), 1):
            full.append(r)
            if i % 25 == 0 or i == len(miss):
                el = time.time() - t1
                print(f"  确认 {i}/{len(miss)} {el:.0f}s", flush=True)

    changed, rolled, same = [], [], []
    for r in full:
        p = r["path"]
        if r["n"] < 0:
            continue
        on, labeled = old[p]
        if any(r["rolls"]) or r["n"] != on:
            item = {"path": p, "old_n": on, "new_n": r["n"], "rolls": r["rolls"],
                    "labeled": labeled, "boxes": r["boxes"]}
            changed.append(item)
            # 只有"带 roll"的才是这次要修的（数量变化多是 YuNet 在没人的照片上瞎报）
            (rolled if any(r["rolls"]) else same).append(item)
    print(f"\n==== 结果 ====")
    print(f"  YOLO 漏检: {len(miss)}")
    print(f"  完整检测后真的变了: {len(changed)}（其中带 roll {len(rolled)}，只变数量 {len(same)}）")
    print(f"  未标注: {sum(1 for c in changed if not c['labeled'])} | "
          f"已标注: {sum(1 for c in changed if c['labeled'])}")

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump({"miss": miss, "errors": errs, "changed": changed}, fh, ensure_ascii=False)
    unlabeled = [c["path"] for c in rolled if not c["labeled"]]
    with open(FIX_LIST, "w", encoding="utf-8") as fh:
        fh.write("\n".join(unlabeled) + ("\n" if unlabeled else ""))
    print(f"  报告: {os.path.relpath(REPORT, ROOT)}")
    print(f"  未标注待修清单: {os.path.relpath(FIX_LIST, ROOT)}")
    for c in rolled[:15]:
        print(f"    {os.path.basename(c['path'])}: {c['old_n']}→{c['new_n']} roll={c['rolls']}"
              f"{' [已标注]' if c['labeled'] else ''}")


if __name__ == "__main__":
    main()
