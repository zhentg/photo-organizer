# pages/6_🔍_重复管理.py
import streamlit as st
from pathlib import Path

from core.shared import (
    page_config, render_sidebar, get_matcher, get_photo_manager,
    get_duplicate_photos_cached, invalidate_snapshot_caches
)

# 页面配置
page_config("重复管理 - 人脸照片分类工具", "🔍")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("🔍 重复数据管理")
st.markdown("---")

# 创建两列：照片重复和人脸编码重复
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 照片重复")
    duplicates = get_duplicate_photos_cached()

    if not duplicates["by_md5"]:
        st.success("✅ 无重复照片（MD5）")
    else:
        total_dup = sum(len(v) for v in duplicates["by_md5"].values())
        st.warning(f"⚠️ 发现 {len(duplicates['by_md5'])} 组重复照片，共 {total_dup} 条记录")

        for md5, photos in duplicates["by_md5"].items():
            with st.expander(f"MD5: {md5[:8]}... ({len(photos)} 张)"):
                for i, photo in enumerate(photos):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{Path(photo['path']).name}**")
                        st.caption(photo['path'])
                    with c2:
                        if st.button("删除", key=f"del_photo_{md5}_{i}"):
                            photo_manager.remove_photo(photo['path'])
                            matcher.remove_photo_encodings(photo['path'])
                            invalidate_snapshot_caches()
                            st.success("已删除")
                            st.rerun()

with col2:
    st.subheader("👤 人脸编码重复")

    threshold = st.slider("相似度阈值", 0.3, 0.6, 0.4, 0.05,
                          help="距离小于此值视为重复，值越小越严格")

    if st.button("检测重复编码"):
        with st.spinner("检测中..."):
            dup_encodings = matcher.get_duplicate_encodings(threshold)

        if not dup_encodings:
            st.success("✅ 无重复人脸编码")
        else:
            st.warning(f"⚠️ 发现 {len(dup_encodings)} 对相似编码")
            st.session_state['dup_encodings'] = dup_encodings

    if 'dup_encodings' in st.session_state:
        dup_encodings = st.session_state['dup_encodings']
        for i, dup in enumerate(dup_encodings[:20]):  # 只显示前20个
            with st.container():
                st.markdown(f"**{dup['person']}** ↔ **{dup['duplicate_of']['person']}**")
                st.caption(f"距离: {dup['distance']} | "
                          f"{Path(dup['photo_path']).name} ↔ {Path(dup['duplicate_of']['photo_path']).name}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("删除左侧", key=f"del_enc_{i}_a"):
                        matcher.remove_encoding(dup['encoding_id'])
                        invalidate_snapshot_caches()
                        st.session_state.pop('dup_encodings', None)  # 结果已过期
                        st.success("已删除")
                        st.rerun()
                with c2:
                    if st.button("删除右侧", key=f"del_enc_{i}_b"):
                        matcher.remove_encoding(dup['duplicate_of']['encoding_id'])
                        invalidate_snapshot_caches()
                        st.session_state.pop('dup_encodings', None)
                        st.success("已删除")
                        st.rerun()
                st.markdown("---")
