"""并行版 repair_faces：多进程检测、单进程写库。

单进程跑 500+ 张要一个多小时（每张 1.5~12s，卡在检测上），这里用进程池
把检测摊开，写库仍在主进程里串行——SQLite 不会被并发写坏。

护栏与 repair_faces 完全一致：
  * 已标注照片默认跳过；
  * 检出框数 > max_faces 视为误检簇，跳过待人工；
  * 新旧框位完全一致视为无变化，不写。

用法：
  python tools/batch_repair.py --from-file data/_sideways_unlabeled.txt \
      [--include-labeled] [--workers 6] [--dry-run]
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


def _detect(path):
    """子进程里检测一张（FaceDetector 按进程懒加载）。"""
    global _DET
    try:
        _DET
    except NameError:
        from core.face_detector import FaceDetector
        # 带 photo_index_path：手动旋转过的照片必须按用户定的方向重检测，
        # 否则批量修复会把人工纠正抹掉（detect_faces 会自动读 manual_rotate）
        _DET = FaceDetector(photo_index_path=DB)
    try:
        faces = _DET.detect_faces(path)
        out = []
        for f in faces:
            enc = f.get("encoding")
            if enc is not None and not isinstance(enc, list):
                enc = [float(v) for v in enc]
            out.append({
                "location": [int(v) for v in f["location"]],
                "encoding": enc,
                "image_path": f.get("image_path", path),
                "bbox": [int(v) for v in f["bbox"]],
                **({"roll": f["roll"]} if f.get("roll") else {}),
            })
        return {"path": path, "faces": out}
    except Exception as e:
        return {"path": path, "faces": None, "err": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", required=True)
    ap.add_argument("--include-labeled", action="store_true")
    ap.add_argument("--max-faces", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="data/_batch_repair_report.json")
    args = ap.parse_args()

    from core.face_detector import PhotoManager
    from core.shared import PHOTO_INDEX_PATH, DEFAULT_OUTPUT_DIR

    paths = [ln.strip() for ln in open(args.from_file, encoding="utf-8-sig") if ln.strip()]
    con = sqlite3.connect(DB)
    idx = {}
    for p, faces, persons in con.execute("SELECT path, faces, persons FROM photos"):
        idx[p] = (json.loads(faces or "[]"), json.loads(persons or "[]"))

    todo, stats = [], {"not_indexed": 0, "skipped_missing": 0, "skipped_labeled": 0}
    for p in paths:
        if p not in idx:
            stats["not_indexed"] += 1
            continue
        if not os.path.exists(p):
            stats["skipped_missing"] += 1
            continue
        if idx[p][1] and not args.include_labeled:
            stats["skipped_labeled"] += 1
            continue
        todo.append(p)

    print(f"待检测 {len(todo)} 张 | workers={args.workers} | dry_run={args.dry_run}", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_detect, todo, chunksize=4), 1):
            results.append(r)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} {el:.0f}s 剩余约 {el / i * (len(todo) - i):.0f}s", flush=True)

    pm = PhotoManager(str(PHOTO_INDEX_PATH), str(DEFAULT_OUTPUT_DIR))
    stats.update({"repaired": 0, "unchanged": 0, "no_face": 0, "skipped_suspicious": 0,
                  "errors": 0})
    review, applied = [], []
    for r in results:
        p = r["path"]
        name = os.path.basename(p)
        if r["faces"] is None:
            stats["errors"] += 1
            print(f"  ! {name}: 检测异常 {r.get('err')}")
            continue
        old_faces = idx[p][0]
        new_faces = r["faces"]
        old_boxes = sorted(tuple(f.get("bbox") or ()) for f in old_faces)
        new_boxes = sorted(tuple(f.get("bbox") or ()) for f in new_faces)

        if len(new_faces) > args.max_faces:
            stats["skipped_suspicious"] += 1
            review.append({"path": p, "faces": len(new_faces), "boxes": new_boxes})
            print(f"  ~ {name}: 检出 {len(new_faces)} 个框 > {args.max_faces}，跳过待人工")
            continue
        if not new_faces:
            stats["no_face"] += 1
            print(f"  - {name}: 重检测无脸，跳过")
            continue
        if new_boxes == old_boxes:
            stats["unchanged"] += 1
            continue

        stats["repaired"] += 1
        applied.append({"path": p, "old": len(old_faces), "new": len(new_faces),
                        "roll": [f.get("roll") for f in new_faces]})
        if not args.dry_run:
            pm.clear_photo_faces(p)
            pm.add_photo(p, new_faces)

    print("\n==== 汇总 ====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if review:
        print("\n==== 待人工复核（框数超限）====")
        for r in review[:30]:
            print(f"  {os.path.basename(r['path'])}: {r['faces']} 个框")
        print(f"  ... 共 {len(review)} 张")
    with open(os.path.join(ROOT, args.report), "w", encoding="utf-8") as fh:
        json.dump({"applied": applied, "review": review, "stats": stats,
                   "dry_run": args.dry_run}, fh, ensure_ascii=False)
    print(f"\n报告: {args.report}")


if __name__ == "__main__":
    main()
