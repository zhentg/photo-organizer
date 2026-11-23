"""已标注的"躺着"照片：重检测 + 人名按位置搬运。

背景：这些照片当年在侧躺像素上就检出了脸（YOLO 对旋转不敏感），人名也标了，
但编码是歪的 → 匹配不可靠。这里重跑一次检测（自动摆正），再把旧人名按
"框中心最近"搬到新框上，**名字不丢、编码变正**。

护栏（任一不满足就跳过，绝不猜）：
  * 新旧人脸数量必须一致；
  * 每个旧框都能在 0.6 倍对角线距离内找到唯一的新框；
  * 新框数 > 6 视为可疑（误检簇），不动。

用法：
  python tools/rewrite_sideways.py --from-file data/_sideways_labeled.txt [--apply]
不加 --apply 只打印计划。
"""
import argparse
import json
import math
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "databases", "photo_index.db")


def center(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-faces", type=int, default=6)
    args = ap.parse_args()

    from core.face_detector import FaceDetector, PhotoManager
    from core.shared import PHOTO_INDEX_PATH, PHOTO_INDEX_DB, DEFAULT_OUTPUT_DIR

    paths = [ln.strip() for ln in open(args.from_file, encoding="utf-8-sig") if ln.strip()]
    det = FaceDetector(photo_index_path=str(PHOTO_INDEX_DB))
    pm = PhotoManager(str(PHOTO_INDEX_PATH), str(DEFAULT_OUTPUT_DIR))

    con = sqlite3.connect(DB)
    stats = {"rewritten": 0, "skipped_count": 0, "skipped_match": 0, "skipped_guard": 0,
             "unchanged": 0, "missing": 0}
    for n, path in enumerate(paths, 1):
        row = con.execute("SELECT faces, persons FROM photos WHERE path=?", (path,)).fetchone()
        if row is None:
            stats["missing"] += 1
            print(f"[{n}/{len(paths)}] {os.path.basename(path)}: 不在索引里，跳过")
            continue
        old_faces = json.loads(row[0] or "[]")
        old_persons = json.loads(row[1] or "[]")
        if not old_faces:
            stats["unchanged"] += 1
            continue
        # persons 与 faces 一一对应；长度不齐时右补空
        old_persons = (old_persons + [""] * len(old_faces))[: len(old_faces)]

        new_faces = det.detect_faces(path)
        if len(new_faces) > args.max_faces:
            stats["skipped_guard"] += 1
            print(f"[{n}/{len(paths)}] {os.path.basename(path)}: 检出 {len(new_faces)} 张脸 > {args.max_faces}，待人工")
            continue
        if not new_faces:
            stats["unchanged"] += 1
            print(f"[{n}/{len(paths)}] {os.path.basename(path)}: 重检测无脸，跳过")
            continue
        if len(new_faces) != len(old_faces):
            stats["skipped_count"] += 1
            print(f"[{n}/{len(paths)}] {os.path.basename(path)}: 旧 {len(old_faces)} 张 → 新 {len(new_faces)} 张，数量不一致，跳过")
            continue

        # 按中心距离一一定位，要求"唯一且够近"
        pairs, used = [], set()
        ok = True
        for i, of in enumerate(old_faces):
            oc = center(of["bbox"])
            diag = math.hypot(of["bbox"][2], of["bbox"][3])
            best, bestd = None, None
            for j, nf in enumerate(new_faces):
                if j in used:
                    continue
                nc = center(nf["bbox"])
                dist = math.hypot(oc[0] - nc[0], oc[1] - nc[1])
                if bestd is None or dist < bestd:
                    best, bestd = j, dist
            if best is None or bestd > 0.6 * max(diag, 1):
                ok = False
                break
            used.add(best)
            pairs.append((i, best))
        if not ok:
            stats["skipped_match"] += 1
            print(f"[{n}/{len(paths)}] {os.path.basename(path)}: 旧框对不上新框，跳过")
            continue

        persons_new = [""] * len(new_faces)
        for i, j in pairs:
            persons_new[j] = old_persons[i]
        changed = any(f.get("roll") for f in new_faces) or any(
            json.dumps(a.get("bbox")) != json.dumps(b.get("bbox"))
            for a, b in zip(old_faces, new_faces))

        print(f"[{n}/{len(paths)}] {os.path.basename(path)}: {len(new_faces)} 张脸，"
              f"人名 {old_persons} → {persons_new}，roll={[f.get('roll') for f in new_faces]}"
              f"{'' if args.apply else '  (dry-run)'}")
        if args.apply:
            pm.clear_photo_faces(path)
            pm.add_photo(path, new_faces)
            pm.update_photo_persons(path, persons_new)
        stats["rewritten" if changed else "unchanged"] += 1

    print("\n==== 汇总 ====")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
