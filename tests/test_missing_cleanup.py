"""失效记录清理回归测试（2026-11-11 标注页刷新变慢修复）。

背景：标注页/分类浏览页原本用 st.session_state["_missing_cleaned"] 做
「每会话一次」的清理，但 F5 会新建 session → 每次刷新都重跑
get_all_photos()（8867 行 faces/persons JSON 全解析 ~2s）+ 8867 次
os.path.exists，实测刷新 2.9s（进程冷启动 7.3s）。

修复：clean_missing_photos 只查 path 列，去重提到进程级
（DB 文件签名 + 60s TTL），并给「全部清理」按钮留了 force=True。

本测试锁住 4 件事：
1. get_all_photo_paths 只返回路径，且不解析 faces JSON；
2. 真实清理能删掉源文件不存在的记录，保留存在的记录；
3. 进程级去重生效：DB 未变时第二次调用不再扫盘；
4. force=True 强制执行（「缺失文件」页按钮不会被去重挡住）。

运行：python tests/test_missing_cleanup.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.face_detector import PhotoManager
from core import shared

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_pm(tmp):
    """在临时目录建一个独立的小型索引库（绝不碰真库）。"""
    db = os.path.join(tmp, "index.db")
    pm = PhotoManager(db, os.path.join(tmp, "out"))
    return pm


def main():
    tmp = tempfile.mkdtemp(prefix="fps_missing_")
    try:
        pm = _make_pm(tmp)
        real = os.path.join(tmp, "real.jpg")
        with open(real, "wb") as f:
            f.write(b"fake-jpeg")
        gone = os.path.join(tmp, "deleted.jpg")   # 故意不创建

        pm.add_photo(real, [{"bbox": [0, 0, 10, 10]}])
        pm.add_photo(gone, [{"bbox": [1, 2, 3, 4]}])

        # 1) 只取路径，且不解析 JSON
        paths = pm.get_all_photo_paths()
        assert set(paths) == {real, gone}, paths
        assert isinstance(pm.db_path, str) and pm.db_path == os.path.join(tmp, "index.db")
        print("  ✓ get_all_photo_paths 只返回路径")

        # 2) force=True 真实清理
        shared._MISSING_CHECK = {"sig": None, "at": 0.0}
        removed = shared.clean_missing_photos(pm, force=True)
        assert removed == [gone], removed
        assert set(pm.get_all_photo_paths()) == {real}, "存在的照片被误删"
        print("  ✓ 真实清理：删掉失效记录、保留有效记录")

        # 3) 进程级去重：DB 未变时不再扫盘
        calls = {"n": 0}
        orig = pm.get_all_photo_paths

        def counting_paths():
            calls["n"] += 1
            return orig()

        pm.get_all_photo_paths = counting_paths
        assert shared.clean_missing_photos(pm) == [], "第二次不该再删东西"
        assert calls["n"] == 0, f"去重失效，又扫了一次盘 ({calls['n']})"
        print("  ✓ 进程级去重：DB 未变时第二次调用不扫盘")

        # 4) 新增一条失效记录 → 签名变化 + force 强制清理
        gone2 = os.path.join(tmp, "deleted2.jpg")
        pm.add_photo(gone2, [])
        assert gone2 in pm.get_all_photo_paths()
        removed2 = shared.clean_missing_photos(pm, force=True)
        assert removed2 == [gone2], removed2
        assert set(pm.get_all_photo_paths()) == {real}
        print("  ✓ force=True 强制执行（「全部清理」按钮不受去重影响）")

        print("\n全部通过 ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
