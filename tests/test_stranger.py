# tests/test_stranger.py — 「陌生人」一键标记回归测试
"""直接运行：python tests/test_stranger.py

覆盖：
  1. mark_photo_as_strangers 各分支（录音替身，不碰真实库）
  2. PhotoManager.get_stranger_photo_count 与直连 SQL 一致（只读）
  3. AppTest 集成：点「👤 本张全是陌生人」→ 只写 persons、不写 encodings、
     会话键按新队列位置重映射（照片移出后状态不串脸）
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stranger import STRANGER_NAME, mark_photo_as_strangers
from core.face_detector import PhotoManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_INDEX_DB = os.path.join(PROJECT_ROOT, "databases", "photo_index.db")


# ─────────────────────────────────────────────
# 录音替身：记录调用，不写库
# ─────────────────────────────────────────────
class RecPM:
    def __init__(self):
        self.calls = []

    def update_photo_persons(self, path, persons):
        self.calls.append(("persons", path, persons))

    def clear_photo_faces(self, path):
        self.calls.append(("clear", path))

    def remove_face_from_photo(self, path, bbox):
        self.calls.append(("rmface", path, bbox))


class RecMatcher:
    def __init__(self):
        self.calls = []

    def add_person(self, **kw):
        self.calls.append(("add_person", kw))

    def add_false_face(self, **kw):
        self.calls.append(("add_false", kw))


def _faces(n):
    return [{"encoding": [0.1] * 8, "location": (0, 0, 0, 0),
             "bbox": (i * 20, 0, 10, 10)} for i in range(n)]


def _assert_no_exception(at):
    """本版本 Streamlit 的 at.exception 是 ElementList（空 = 无异常）。"""
    excs = [str(e) for e in at.exception]
    assert not excs, excs


def _ss_get(at, key, default=None):
    """AppTest 的 session_state 代理没有 .get()，用 try 包一层。"""
    try:
        return at.session_state[key]
    except KeyError:
        return default


def test_branches():
    path = r"D:\photoes\test\a.jpg"

    # 1) 3 脸、1 个排除 → 2 个陌生人 + 1 个假脸 + 移出 faces
    pm, mt = RecPM(), RecMatcher()
    n = mark_photo_as_strangers(pm, mt, path, _faces(3), excluded={1})
    assert n == 2, n
    assert ("persons", path, [STRANGER_NAME, STRANGER_NAME]) in pm.calls
    assert [c[0] for c in mt.calls] == ["add_false"], mt.calls
    assert [c[0] for c in pm.calls] == ["persons", "rmface"], pm.calls

    # 2) 全部排除 → clear_photo_faces + 每脸进 false_faces，返回 0
    pm, mt = RecPM(), RecMatcher()
    n = mark_photo_as_strangers(pm, mt, path, _faces(2), excluded={0, 1})
    assert n == 0, n
    assert ("clear", path) in pm.calls
    assert [c[0] for c in mt.calls] == ["add_false", "add_false"]
    assert not any(c[0] == "persons" for c in pm.calls)

    # 3) 无排除 → 全写陌生人，不碰假脸
    pm, mt = RecPM(), RecMatcher()
    n = mark_photo_as_strangers(pm, mt, path, _faces(4))
    assert n == 4, n
    assert ("persons", path, [STRANGER_NAME] * 4) in pm.calls
    assert mt.calls == [], mt.calls

    # 4) 陌生人绝不写 encodings（污染全局最近邻匹配的防线）
    pm, mt = RecPM(), RecMatcher()
    mark_photo_as_strangers(pm, mt, path, _faces(3))
    assert not any(c[0] == "add_person" for c in mt.calls)

    print("✓ mark_photo_as_strangers 分支语义正确（陌生人只写 persons，不写 encodings）")


def test_stranger_count():
    """get_stranger_photo_count 与直连 SQL 一致（只读，不改库）。"""
    if not os.path.exists(PHOTO_INDEX_DB):
        print("· 跳过：photo_index.db 不存在")
        return
    pm = PhotoManager(PHOTO_INDEX_DB, PROJECT_ROOT)
    got = pm.get_stranger_photo_count()
    con = sqlite3.connect(PHOTO_INDEX_DB)
    expect = con.execute(
        "SELECT COUNT(*) FROM photos WHERE persons LIKE ?",
        ('%"' + STRANGER_NAME + '"%',)).fetchone()[0]
    con.close()
    assert got == expect, (got, expect)
    print(f"✓ get_stranger_photo_count = {got}（与直连 SQL 一致）")


def test_apptest_button():
    """AppTest：点陌生人按钮 → 写 persons、不写 encodings、键重映射。"""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("· 跳过：无 streamlit.testing")
        return

    import core.shared as shared

    page = os.path.join(PROJECT_ROOT, "pages", "2_👤_人脸标注.py")
    if not os.path.exists(page):
        print("· 跳过：标注页不存在")
        return

    # 录音替身（子类化真实管理器，只覆盖写方法 → 零库写入）
    rec_pm, rec_mt = RecPM(), RecMatcher()

    class PM(shared.PhotoManager):
        def update_photo_persons(self, p, persons):
            rec_pm.calls.append(("persons", p, persons))

        def clear_photo_faces(self, p):
            rec_pm.calls.append(("clear", p))

        def remove_face_from_photo(self, p, bbox):
            rec_pm.calls.append(("rmface", p, bbox))

    class MT(shared.FaceMatcher):
        def add_person(self, **kw):
            rec_mt.calls.append(("add_person", kw))

        def add_false_face(self, **kw):
            rec_mt.calls.append(("add_false", kw))

    pm = PM(str(shared.PHOTO_INDEX_PATH), str(shared.DEFAULT_OUTPUT_DIR))
    mt = MT(str(shared.FACE_DB_PATH))

    orig_pm, orig_mt = shared.get_photo_manager, shared.get_matcher
    shared.get_photo_manager = lambda: pm
    shared.get_matcher = lambda *a, **k: mt
    try:
        at = AppTest.from_file(page, default_timeout=240)
        at.run()
        _assert_no_exception(at)

        btns = [b for b in at.button if b.key == "stranger_0"]
        assert btns, "找不到「👤 本张全是陌生人」按钮"
        # 先给第 2 张照片的第 1 个人脸塞一个已选姓名（模拟用户操作）
        at.session_state["sb_1_0"] = "测试甲"
        at.run()
        btns[0].click().run()
        _assert_no_exception(at)

        # 1) 只写 persons，不写 encodings
        person_calls = [c for c in rec_pm.calls if c[0] == "persons"]
        assert person_calls, "没有写入 persons"
        assert all(c[2] and all(p == STRANGER_NAME for p in c[2])
                   for c in person_calls), person_calls
        assert not any(c[0] == "add_person" for c in rec_mt.calls), \
            "陌生人不得写入 encodings"

        # 2) 会话键重映射：照片 0 移出后，照片 1 的键应落到位置 0
        #    （录音替身不落库，照片 0 仍在队列里，sb_0_0 不会被重新预填覆盖）
        assert _ss_get(at, "sb_0_0") == "测试甲", \
            dict((k, v) for k, v in at.session_state.items() if k.startswith("sb_"))
        print("✓ AppTest：陌生人按钮只写 persons、不写 encodings、键按新队列重映射")
    finally:
        shared.get_photo_manager, shared.get_matcher = orig_pm, orig_mt


if __name__ == "__main__":
    test_branches()
    test_stranger_count()
    test_apptest_button()
    print("\n全部通过 ✓")
