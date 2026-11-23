# pages/8_🚫_假脸管理.py
import streamlit as st
from pathlib import Path

from core.shared import (
    page_config, render_sidebar, get_matcher, extract_face_image,
    get_false_faces_cached, invalidate_snapshot_caches
)

# 页面配置
page_config("假脸管理 - 人脸照片分类工具", "🚫")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("🚫 已排除假人脸管理")
st.markdown("---")

false_faces = get_false_faces_cached()

if not false_faces:
    st.success("✅ 暂无已排除的假人脸")
    st.info("💡 在「人脸标注」页面点击 X 按钮可将误检测结果标记为假人脸，这些记录将在新照片处理时被自动过滤。")
else:
    st.warning(f"共有 {len(false_faces)} 个被排除的假人脸记录")
    st.caption("这些检测结果将在新照片导入时被自动过滤，不会出现在人脸标注流程中。")

    # 批量操作
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空所有记录", type="secondary"):
            for ff in false_faces:
                matcher.remove_false_face(ff["photo_path"], ff["bbox"])
            invalidate_snapshot_caches()   # 列表已进快照缓存，撤销后必须失效
            st.success(f"已清空 {len(false_faces)} 条记录")
            st.rerun()
    with col2:
        if st.button("🔄 刷新列表"):
            st.rerun()

    st.markdown("---")

    # 显示假人脸列表（网格布局）
    # 2026-11-11：原来每轮 rerun 把 215 张脸全部解码一遍（实测 9.9s / 10.5s 页面），
    # 而图片解码 LRU 只有 16 条 → 必然抖动。改成每页 24 条，只解码当前页。
    PAGE_SIZE = 24
    total = len(false_faces)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(int(st.session_state.get("ff_page", 1)), 1), n_pages)
    p_prev, p_next, p_info = st.columns([1, 1, 4])
    with p_prev:
        if st.button("◀ 上一页", disabled=page <= 1, use_container_width=True):
            st.session_state["ff_page"] = page - 1
            st.rerun()
    with p_next:
        if st.button("下一页 ▶", disabled=page >= n_pages, use_container_width=True):
            st.session_state["ff_page"] = page + 1
            st.rerun()
    with p_info:
        st.caption(f"第 {page}/{n_pages} 页 · 共 {total} 条 · 每页 {PAGE_SIZE} 条"
                   "（只解码当前页，避免每轮全量解码）")

    offset = (page - 1) * PAGE_SIZE
    shown = false_faces[offset:offset + PAGE_SIZE]
    cols_per_row = 6
    for row_start in range(0, len(shown), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row_start + col_idx
            if idx >= len(shown):
                break
            ff = shown[idx]
            with cols[col_idx]:
                try:
                    face_img = extract_face_image(ff["photo_path"], ff["location"], max_width=200)
                    st.image(face_img, width=120)
                except Exception:
                    st.write("❓")

                st.caption(f"**{Path(ff['photo_path']).name[:15]}...**")
                st.caption(f"排除: {ff['marked_at'][:10]}")

                if st.button("↩ 撤销", key=f"undo_ff_page_{offset + idx}",
                             use_container_width=True):
                    matcher.remove_false_face(ff["photo_path"], ff["bbox"])
                    invalidate_snapshot_caches()
                    st.success("已撤销")
                    st.rerun()

    st.markdown("---")
    st.info("💡 撤销只从「假脸黑名单」里移除这条记录（下次导入不再拦截它）；"
            "它不会自动回到「人脸标注」队列——那张脸当时已从照片索引里移除，"
            "需要到「照片索引」页对原照片重新检测。")
