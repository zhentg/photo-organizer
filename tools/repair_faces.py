#!/usr/bin/env python
# tools/repair_faces.py — 照片人脸索引修复工具（2026-10-19）
"""
统一解决方案：修复 4 月旧检测器遗留的人脸索引错误。

背景（2026-10-19 用户连续报障）：
  4 月建库用的是 OpenCV DNN 首选的老管线，在部分照片上
    (a) 把袍子/手/灯笼/饰品误检成人脸入库，
    (b) 漏检真人脸。
  用户在标注页把这些误检框逐个 X 掉后，"全排除"分支会
  clear_photo_faces 把整张照片清空——真人脸从未入库，
  于是永久掉出标注队列。当前 YOLO 管线对这些照片全部正确。

用法：
    python tools/repair_faces.py <照片路径或文件夹> [更多路径...] [--dry-run]
    python tools/repair_faces.py "D:\\photoes\\2024春节古装照片" --dry-run
    python tools/repair_faces.py "D:\\photoes\\2024春节古装照片"  # 实际执行

行为：
  对每张 persons 为空（未标注）的照片：
      记录旧框数 → clear_photo_faces → detect_faces（当前管线）→ add_photo 写回
  已标注照片默认跳过——重检测会打乱"第几张脸对应哪个名字"的位置映射，
  改脸框等于把人名贴到别人脸上（如确需处理，加 --include-labeled 并人工核对）。
  --dry-run 只检测和报告，不写库。

执行后必须重启 streamlit 服务：未标注列表是进程内快照缓存
  （st.cache_data），外部写库不会自动刷新（见 HANDOVER 坑10）。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.face_detector import FaceDetector, PhotoManager  # noqa: E402
from core.shared import PHOTO_INDEX_PATH, PHOTO_INDEX_DB, DEFAULT_OUTPUT_DIR  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_paths(inputs):
    """把输入（文件或文件夹）展开成照片路径列表（保持顺序，去重）。

    注意：经 Git Bash 传进来的 Windows 路径可能被转成正斜杠
    （D:/photoes/...），与库里的 D:\\photoes\\... 键不匹配，
    统一用 normpath 规范化（Windows 下转回反斜杠）。
    """
    out = []
    seen = set()
    for p in inputs:
        p = os.path.normpath(p)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                fp = os.path.normpath(os.path.join(p, name))
                if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in IMG_EXTS:
                    if fp not in seen:
                        seen.add(fp)
                        out.append(fp)
        elif os.path.isfile(p):
            if p not in seen:
                seen.add(p)
                out.append(p)
        else:
            print(f"[跳过] 路径不存在: {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description="用当前管线重检测并写回照片人脸索引")
    ap.add_argument("paths", nargs="*", help="照片文件或文件夹（可多个）")
    ap.add_argument("--from-file", default=None,
                    help="从文本文件读路径清单（每行一个）——路径含空格时用这个，"
                         "避免 shell 拆词")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写库")
    ap.add_argument("--include-labeled", action="store_true",
                    help="连已标注照片一起处理（危险：人名会与脸错位）")
    ap.add_argument("--max-faces", type=int, default=6,
                    help="单张检出超过 N 个框视为可疑（误检簇），跳过并列入待人工复核"
                        "（例：5C3A4656 凤冠+石栏误检 11 个；默认 6）")
    args = ap.parse_args()

    inputs = list(args.paths)
    if args.from_file:
        with open(args.from_file, encoding="utf-8-sig") as fh:
            inputs += [ln.strip() for ln in fh if ln.strip()]
    if not inputs:
        ap.error("至少给一个路径，或用 --from-file")
    photos = collect_paths(inputs)
    print(f"待处理照片: {len(photos)} 张 | dry_run={args.dry_run} | max_faces={args.max_faces}")

    pm = PhotoManager(str(PHOTO_INDEX_PATH), str(DEFAULT_OUTPUT_DIR))
    detector = FaceDetector(photo_index_path=str(PHOTO_INDEX_DB))
    all_photos = pm.get_all_photos()  # path -> {faces, persons, ...}

    stats = {"repaired": 0, "unchanged": 0, "no_face": 0,
             "skipped_labeled": 0, "skipped_missing": 0, "not_indexed": 0,
             "skipped_suspicious": 0}
    review = []
    for i, path in enumerate(photos, 1):
        name = os.path.basename(path)
        rec = all_photos.get(path)
        if rec is None:
            print(f"[{i}/{len(photos)}] {name}: 不在照片索引中，跳过")
            stats["not_indexed"] += 1
            continue
        if not os.path.exists(path):
            print(f"[{i}/{len(photos)}] {name}: 磁盘文件缺失，跳过")
            stats["skipped_missing"] += 1
            continue
        persons = rec.get("persons") or []
        if persons and not args.include_labeled:
            print(f"[{i}/{len(photos)}] {name}: 已标注 {persons}，跳过")
            stats["skipped_labeled"] += 1
            continue

        old_n = len(rec.get("faces") or [])
        old_boxes = sorted(tuple(f.get("bbox") or ()) for f in (rec.get("faces") or []))
        new_faces = detector.detect_faces(path)
        new_boxes = sorted(tuple(f.get("bbox") or ()) for f in new_faces)
        new_n = len(new_faces)

        if new_n > args.max_faces:
            # 典型误检簇：凤冠/首饰/石栏/灯笼等被检出一串。宁可不动，
             # 也不能把垃圾框写进标注队列让用户逐个 X。
            print(f"[{i}/{len(photos)}] {name}: 检出 {new_n} 个框 > 上限 {args.max_faces}，"
                  f"疑似误检簇，跳过（待人工复核）")
            stats["skipped_suspicious"] += 1
            review.append((name, new_boxes))
            continue

        tag = "dry-run" if args.dry_run else "写回"
        if new_boxes == old_boxes:
            print(f"[{i}/{len(photos)}] {name}: {old_n} 张，框位一致，跳过")
            stats["unchanged"] += 1
        else:
            print(f"[{i}/{len(photos)}] {name}: {old_n} → {new_n} 张"
                  f"（新框: {new_boxes}）{tag}")
            stats["repaired"] += 1
        if new_n == 0:
            stats["no_face"] += 1

        if not args.dry_run:
            # 与标注页「重新匹配」同一路径：清旧 → 重检测 → 写回。
            # add_photo 对已存在的行只更新 faces/md5/indexed_at，不动 persons。
            pm.clear_photo_faces(path)
            pm.add_photo(path, new_faces)

    print("\n==== 汇总 ====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if review:
        print("\n==== 待人工复核（检出框数超限，已跳过）====")
        for name, boxes in review:
            print(f"  {name}: {len(boxes)} 个框 {boxes}")
    if not args.dry_run and stats["repaired"]:
        print("\n请重启 streamlit 服务以刷新未标注列表快照缓存。")


if __name__ == "__main__":
    main()
