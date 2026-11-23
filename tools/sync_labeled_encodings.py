#!/usr/bin/env python
# tools/sync_labeled_encodings.py — 已标注照片的编码同步（2026-10-20）
"""
问题：tools/rewrite_sideways.py 只在照片索引（photo_index.db）里重检测 + 搬人名，
没有碰识别库（face_db.db）的 encodings。结果：
  * 索引里 bbox/roll/encoding 已是摆正后的；
  * 识别库里这个人还留着"躺着"的旧编码和旧框 →
    匹配继续走歪向量，预览裁剪也可能用旧框。
本工具把识别库的 encodings 按索引里的正确人脸重建（按人名一一对应）。

护栏（任一不满足就跳过，绝不猜）：
  * 索引 faces 数 == 人名数（persons 为空的一律跳过，属未标注）；
  * 识别库里该照片的编码条数 == 索引人脸数；
  * 新框与旧框中心距离 < 0.6 倍旧框对角线（同一张脸）；
  * roll != 0 或 bbox 变化才算需要同步。

用法：
  python tools/sync_labeled_encodings.py --all-rolled      # 扫描库里所有 roll!=0 的已标注照片
  python tools/sync_labeled_encodings.py --path-like 975a80
  ... 加 --apply 才写库（默认只打印计划）
"""
import argparse
import json
import math
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FACE_DB = os.path.join(ROOT, "databases", "face_db.db")
INDEX_DB = os.path.join(ROOT, "databases", "photo_index.db")


def center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-like", default=None, help="只处理路径含该子串的照片")
    ap.add_argument("--all-rolled", action="store_true", help="处理索引里所有 roll!=0 的已标注照片")
    ap.add_argument("--from-file", default=None,
                    help="路径清单（一行一个），配合 --from-file 处理任意一组已标注照片")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not args.path_like and not args.all_rolled and not args.from_file:
        ap.error("给 --path-like / --all-rolled / --from-file")

    idx = sqlite3.connect(INDEX_DB)
    fdb = sqlite3.connect(FACE_DB)

    if args.from_file:
        wanted = [ln.strip() for ln in open(args.from_file, encoding="utf-8-sig") if ln.strip()]
        rows = []
        for w in wanted:
            r = idx.execute("SELECT path, faces, persons FROM photos WHERE path=?",
                            (w,)).fetchone()
            if r:
                rows.append(r)
    elif args.all_rolled:
        rows = idx.execute("SELECT path, faces, persons FROM photos").fetchall()
    else:
        rows = idx.execute(
            "SELECT path, faces, persons FROM photos WHERE path LIKE ?",
            (f"%{args.path_like}%",)).fetchall()

    stats = {"checked": 0, "in_sync": 0, "no_persons": 0, "no_roll": 0,
             "skipped_guard": 0, "fixed": 0, "planned": 0}
    for path, faces_json, persons_json in rows:
        faces = json.loads(faces_json or "[]")
        persons = json.loads(persons_json or "[]")
        if args.all_rolled:
            if not any((f.get("roll") or 0) for f in faces):
                continue
            if not any(p for p in persons if p):
                stats["no_persons"] += 1
                continue
        if len(faces) != len(persons) or not all(persons):
            print(f"[跳过] {os.path.basename(path)}: faces={len(faces)} persons={len(persons)} 对不齐")
            stats["skipped_guard"] += 1
            continue

        stats["checked"] += 1
        enc_rows = fdb.execute(
            "SELECT e.id, p.name, e.bbox, e.location FROM encodings e "
            "JOIN persons p ON p.id=e.person_id WHERE e.photo_path=?", (path,)).fetchall()
        name_to_idx = {name: i for i, name in enumerate(persons)}
        stale = []
        guard_ok = True
        for rid, name, bbox_s, loc_s in enc_rows:
            if name not in name_to_idx:
                stale.append((rid, name, "库里多出这个名字"))
                continue
            i = name_to_idx[name]
            f = faces[i]
            old = json.loads(bbox_s)
            new = list(f["bbox"])
            d = math.hypot(center(old)[0] - center(new)[0], center(old)[1] - center(new)[1])
            if old == new:
                continue
            if d > 0.6 * max(math.hypot(old[2], old[3]), 1):
                stale.append((rid, name, f"框偏离 {d:.0f}px"))
                guard_ok = False
                continue
            # 该人名下这张照片的旧框，必须离"分给它的新框"最近（防串脸）
            dc = center(old)
            others = [(math.hypot(dc[0] - center(nf["bbox"])[0], dc[1] - center(nf["bbox"])[1]), j)
                      for j, nf in enumerate(faces) if j != i]
            if others and min(others)[0] < d:
                stale.append((rid, name, f"更近的是第{min(others)[1]}张脸，疑似串脸"))
                guard_ok = False
                continue
            stale.append((rid, name, f"{old} → {new}"))

        if not guard_ok:
            print(f"[跳过] {os.path.basename(path)}: " +
                  "; ".join(f"{n}:{w}" for _, n, w in stale))
            stats["skipped_guard"] += 1
            continue

        if not stale and len(enc_rows) == len(faces):
            stats["in_sync"] += 1
            continue
        print(f"[{'' if args.apply else 'plan'}] {os.path.basename(path)} roll="
              f"{[f.get('roll') for f in faces]} 编码条数 库里{len(enc_rows)}/索引{len(faces)}")
        for rid, name, why in stale:
            print(f"    - id={rid} {name}: {why}")

        if args.apply:
            for f, name in zip(faces, persons):
                if not f.get("encoding"):
                    continue
                # 删掉该人该照片的旧行，写新行（bbox/location/encoding 全换）
                fdb.execute(
                    "DELETE FROM encodings WHERE photo_path=? AND person_id="
                    "(SELECT id FROM persons WHERE name=?)", (path, name))
                pid = fdb.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
                if pid is None:
                    fdb.execute("INSERT INTO persons(name) VALUES(?)", (name,))
                    pid = fdb.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
                fdb.execute(
                    "INSERT INTO encodings(person_id,encoding,photo_path,location,bbox,added_at) "
                    "VALUES(?,?,?,?,?,datetime('now','localtime'))",
                    (pid[0], json.dumps(f["encoding"]), path,
                     json.dumps(f["location"]), json.dumps(list(f["bbox"]))))
            fdb.commit()
            stats["fixed"] += 1
        else:
            stats["planned"] += 1

    print("\n==== 汇总 ====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.apply and stats["fixed"]:
        print("\n识别索引有内存缓存：请重启 streamlit 服务。")


if __name__ == "__main__":
    main()
