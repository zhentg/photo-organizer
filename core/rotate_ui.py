# core/rotate_ui.py — 「旋转 90°」按钮组（标注页 / 分类浏览页共用）
"""统一的手动旋转交互，避免两个页面各写一份、行为不一致。

交互约定（用户视角）：
- 每点一次转 90°，**立刻**用新方向重检测这张照片的人脸（可连点直到摆正）；
- 右转 = 顺时针 = 库里 manual_rotate -90，左转 = 逆时针 = +90
  （与 roll 的"逆时针为正"一致，排查时不会看反）；
- 转过之后自动"四向探索"关闭：方向由用户拍板，不会再被算法偷偷转回去。
"""
import streamlit as st
from pathlib import Path

from core.rotation import rotate_photo, reset_manual_rotate, label_of


def clear_photo_session_keys(idx, n_faces: int = 12):
    """清掉某张照片（队列下标 idx）所有会话键，防止旋转后新框读到旧姓名。"""
    for i in range(max(n_faces, 1)):
        for pfx in ("sb_", "cb_", "ti_", "q_", "px_"):
            st.session_state.pop(pfx + str(idx) + "_" + str(i), None)


def rotate_and_flash(detector, matcher, pm, photo_path, delta):
    """旋转 + 重检测，结果存进 session_state（rerun 后由页面显示提示）。"""
    res = rotate_photo(pm, detector, photo_path, delta, matcher=matcher)
    st.session_state["_rot_flash"] = (Path(photo_path).name, res)
    return res


def render_rotate_buttons(detector, matcher, pm, photo_path, manual_rotate,
                          key_prefix: str, idx: int = None,
                          compact: bool = False):
    """一行「↺ 左转90° / ↻ 右转90° / ⟲ 重置」+ 当前角度说明。

    Args:
        idx: 该照片在当前页队列里的下标；给了就顺手清掉它的姓名会话键
            （旋转后脸数会变，旧键会串脸）。
        compact: 分类浏览页的网格里用（按钮只有图标，省地方）。
    """
    from core.shared import invalidate_snapshot_caches   # 延迟导入避免循环

    cur = int(manual_rotate or 0)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        left = st.button("↺" + ("" if compact else " 左转90°"),
                         key=key_prefix + "_L", use_container_width=True,
                         help="逆时针转 90° 后重新检测人脸")
    with c2:
        right = st.button("↻" + ("" if compact else " 右转90°"),
                          key=key_prefix + "_R", use_container_width=True,
                          help="顺时针转 90° 后重新检测人脸")
    with c3:
        reset = st.button("⟲" + ("" if compact else " 重置"),
                          key=key_prefix + "_0", use_container_width=True,
                          disabled=(cur == 0),
                          help="手动旋转清零，回到自动判断的方向")
    if left or right or reset:
        if reset:
            reset_manual_rotate(pm, detector, photo_path, matcher=matcher)
        else:
            rotate_and_flash(detector, matcher, pm, photo_path,
                             -90 if right else 90)
        if idx is not None:
            clear_photo_session_keys(idx)
        invalidate_snapshot_caches()
        st.rerun()

    if cur:
        st.caption(f"🧭 手动 {label_of(cur)}（manual_rotate={cur}）")
    elif not compact:
        st.caption("🧭 方向由自动判断（未手动旋转）")


def render_rotate_flash():
    """显示上一轮旋转的结果（页面顶部调用；没有就什么都不显示）。"""
    flash = st.session_state.pop("_rot_flash", None)
    if not flash:
        return
    name, res = flash
    txt = (f"《{name}》→ {label_of(res['new_manual'])}，"
           f"人脸 {res['faces_before']} → {res['faces_after']} 张")
    status = res.get("status")
    if status == "ok":
        st.success("🧭 " + txt)
    elif status == "names_lost":
        st.warning("🧭 " + txt + f"；⚠️ 名字未能全部搬移：{res['lost']}（请人工补标）")
    elif status == "no_face":
        st.warning("🧭 " + txt + "；这个方向一张脸都没检出 → 可继续点方向按钮，"
                   "或到「手动旋转待复查」里找到它")
    else:
        st.info("🧭 " + txt + f"（{status}）")
