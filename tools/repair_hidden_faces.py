"""把「部分自动匹配」导致被藏起来的脸放回标注队列。

背景（2026-11-11 全页体检发现）：
旧版「导入照片」页只要有一张脸自动匹配上，就把已匹配的名字写进
photos.persons；而 PhotoManager.get_unlabeled_photos() 只捞 persons 为空的
照片，于是同一张照片里其余没匹配的脸永远不出现在「人脸标注」队列里。
实测全库 271 张照片、408 张脸被这样藏住。

页面 1 已修（部分匹配时 persons 写空），但**历史数据**需要跑一次本脚本。
做法：把这些照片的 persons 清空；已有 encodings 不动，标注页会用
find_match 预填已匹配的名字，确认时 add_person 按 bbox 去重，不会重复入库。

用法（默认 dry-run，只报告不写库）：
    python tools/repair_hidden_faces.py
    python tools/repair_hidden_faces.py --apply      # 先自动备份再写
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "databases", "photo_index.db")
BACKUP_DIR = os.path.join(ROOT, "data", "_backup")
STRANGER = "陌生人"


def find_hidden(con):
    hidden = []
    for path, faces_raw, persons_raw in con.execute(
            "SELECT path,faces,persons FROM photos"):
        try:
            faces = json.loads(faces_raw) or []
            persons = json.loads(persons_raw) or []
        except (TypeError, ValueError):
            continue
        # 只处理「有脸、有名字、但名字数少于脸数」的照片；
        # 陌生人桶标签是照片级的，不算漏标。
        if not faces or not persons or len(persons) >= len(faces):
            continue
        if STRANGER in persons:
            continue
        hidden.append((path, len(faces), len(persons)))
    return hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正写库（会先备份 photo_index.db）")
    args = ap.parse_args()

    if not os.path.exists(DB):
        print(f"找不到 {DB}")
        return 1
    con = sqlite3.connect(DB)
    hidden = find_hidden(con)
    total_faces = sum(f - n for _, f, n in hidden)
    print(f"部分标注照片: {len(hidden)} 张，被藏起来的脸: {total_faces} 张")
    for path, f, n in hidden[:10]:
        print(f"  {os.path.basename(path):<28} 脸 {f} / 名字 {n}")
    if len(hidden) > 10:
        print(f"  ... 其余 {len(hidden) - 10} 张")

    if not args.apply:
        print("\n[dry-run] 未修改任何数据。确认后加 --apply 执行。")
        con.close()
        return 0

    con.close()
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"photo_index.{stamp}.db")
    shutil.copy2(DB, backup)
    print(f"\n已备份到 {backup}")

    con = sqlite3.connect(DB)
    paths = [p for p, _, _ in hidden]
    con.executemany("UPDATE photos SET persons='[]' WHERE path=?",
                    [(p,) for p in paths])
    con.commit()
    left = find_hidden(con)
    con.close()
    print(f"已清空 {len(paths)} 张照片的 persons；剩余部分标注照片: {len(left)}")
    print("重启 Streamlit 后，这些照片会回到「人脸标注」队列（已匹配的名字会自动预填）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
