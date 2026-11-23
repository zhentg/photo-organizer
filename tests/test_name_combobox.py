"""姓名 combobox 回归测试：拼音首字母撞车时必须保留「➕ 新增」入口。

直接运行： python tests/test_name_combobox.py

背景（2026-10-19 用户报障）：姓名框输入 "lzg"（库里没有这个人），
没有弹出新增选项。根因是 auto_select_name 的「唯一匹配即自动确认」——
"lzg" 恰好唯一命中某个姓名时就替用户确认并收起列表，用户来不及点
「➕ 新增」。首字母会撞车（lzg ≠ 刘志高），已改为只有原样等于已有
姓名才自动确认。

覆盖：
  1. auto_select_name：唯一匹配不自动确认
  2. resolve_pick：新增 / 清除 / 普通姓名 / 过期标签
  3. AppTest 集成：有匹配时列表同时带匹配姓名和新增；点新增后
     sb_ 生效、底部批量统计当轮更新
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.name_combobox import auto_select_name, resolve_pick

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pages", "2_👤_人脸标注.py")


def test_auto_select():
    names = ["Anne", "高振雯", "刘志高"]
    # 原样等于已有姓名 → 自动确认（无歧义）
    assert auto_select_name(names, "Anne") == "Anne"
    assert auto_select_name(names, " Anne ") == "Anne"
    # 唯一匹配也不自动确认（首字母撞车，用户可能想新增）
    assert auto_select_name(names, "lzg") is None   # 唯一命中 刘志高
    assert auto_select_name(names, "gzw") is None   # 唯一命中 高振雯
    # 多个匹配 / 无匹配 / 空输入 → 不确认
    assert auto_select_name(names, "a") is None
    assert auto_select_name(names, "zzz") is None
    assert auto_select_name(names, "") is None
    assert auto_select_name(names, None) is None
    print("PASS auto_select_name")


def test_resolve_pick():
    names = ["Anne", "高振雯"]
    assert resolve_pick("Anne", names) == "Anne"
    assert resolve_pick("➕ 新增「lzg」", names) == "lzg"
    assert resolve_pick("✕ 清除选择", names) == ""
    assert resolve_pick("已删除的人", names) is None  # 过期标签，忽略
    print("PASS resolve_pick")


def test_apptest():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(PAGE, default_timeout=240)
    at.run()
    assert not at.exception, at.exception

    # 队列头部会随用户标注漂移（第一张照片可能只有 1 张脸）——
    # 动态取可用的姓名框键，不写死下标。
    cb_keys = [t.key for t in at.text_input if t.key and t.key.startswith("cb_")]
    assert cb_keys, "页面上没有姓名框"
    key1 = cb_keys[0]
    key2 = cb_keys[1] if len(cb_keys) > 1 else key1
    idx1 = key1.split("_")[1:]      # (photo_idx, face_idx)
    idx2 = key2.split("_")[1:]
    rd1 = "rd_" + "_".join(idx1)
    rd2 = "rd_" + "_".join(idx2)
    sb1 = "sb_" + "_".join(idx1)
    sb2 = "sb_" + "_".join(idx2)

    # 场景 1：有匹配时，列表必须同时带匹配姓名和「➕ 新增」
    ti = next(t for t in at.text_input if t.key == key1)
    ti.set_value("a").run()
    opts = list(next(r for r in at.radio if r.key == rd1).options)
    assert "➕ 新增「a」" in opts, opts                      # 关键回归点
    real = [o for o in opts if not o.startswith(("➕", "✕"))]
    assert len(real) >= 1, opts                             # 同时有匹配姓名

    # 场景 2：无匹配的输入（如 "lzg"），也要能弹出新增选项
    ti = next(t for t in at.text_input if t.key == key2)
    ti.set_value("lzg").run()
    opts = list(next(r for r in at.radio if r.key == rd2).options)
    assert "➕ 新增「lzg」" in opts, opts

    # 场景 3：点「➕ 新增」→ 当轮生效，底部批量统计更新
    # 注意：队列头部可能有 NN 已经预填好姓名的人脸，那种框改不改名字都不会
    # 改变「已处理/可确认」计数（本来就是可确认）。所以要挑一个原本空着的框。
    t_key = t_rd = t_sb = None
    for k in cb_keys:
        sb = "sb_" + "_".join(k.split("_")[1:])
        try:
            empty = not at.session_state[sb]
        except KeyError:
            empty = True
        if empty:
            t_key, t_rd, t_sb = k, "rd_" + "_".join(k.split("_")[1:]), sb
            break
    assert t_key is not None, "页面上所有姓名框都已预填，无法验证「新增」生效"

    before = [m.value for m in at.markdown if "批量" in str(m.value)]
    next(t for t in at.text_input if t.key == t_key).set_value("lzg").run()
    next(r for r in at.radio if r.key == t_rd).set_value("➕ 新增「lzg」").run()
    assert at.session_state[t_sb] == "lzg", at.session_state[t_sb]
    after = [m.value for m in at.markdown if "批量" in str(m.value)]
    assert before != after, (before, after)
    print(f"PASS AppTest 集成（用 {key1} / {key2} / {t_key}）")


if __name__ == "__main__":
    test_auto_select()
    test_resolve_pick()
    test_apptest()
    print("ALL PASS")
