# pages/7_⚠️_缺失文件.py
import streamlit as st
import pandas as pd
from pathlib import Path

from core.shared import (
    page_config, render_sidebar, clean_missing_photos,
    get_missing_photos_cached, invalidate_snapshot_caches
)

# 页面配置
page_config("缺失文件管理 - 人脸照片分类工具", "⚠️")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("⚠️ 缺失文件管理")
st.markdown("---")

st.markdown("""
这些照片的**源文件已不在磁盘上**（被移动、删除或重命名），但索引中仍有记录。
清理失效记录不会影响照片本身，只是让索引与磁盘保持同步。
""")

missing = get_missing_photos_cached()

if not missing:
    st.success("✅ 没有缺失文件，索引与磁盘完全同步")
else:
    st.warning(f"共有 {len(missing)} 条记录对应的源文件已缺失")
    st.caption("勾选后点击「删除选中」仅清理索引记录；「全部清理」会移除所有失效记录。")

    # 构建表格数据
    table_data = []
    for m in missing:
        faces = m.get("faces", [])
        persons = m.get("persons", [])
        md5 = m.get("md5") or ""
        indexed_at = m.get("indexed_at") or ""
        table_data.append({
            "选择": False,
            "路径": m["path"],
            "文件名": Path(m["path"]).name,
            "MD5": md5[:8] + "..." if md5 else "-",
            "人脸数": len(faces),
            "标注人物": ", ".join(persons) if persons else "-",
            "索引时间": indexed_at[:16] if indexed_at else "-",
        })

    st.markdown(f"**共 {len(table_data)} 条失效记录**")

    df = pd.DataFrame(table_data)
    edited_df = st.data_editor(
        df,
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", default=False),
            "路径": st.column_config.TextColumn("完整路径", width="large"),
            "文件名": st.column_config.TextColumn("文件名", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )

    # 操作按钮
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        selected_paths = edited_df[edited_df["选择"] == True]["路径"].tolist()
        if st.button(f"🗑️ 删除选中 ({len(selected_paths)})",
                     type="primary", disabled=len(selected_paths) == 0):
            for path in selected_paths:
                photo_manager.remove_photo(path)
                matcher.remove_photo_encodings(path)
            invalidate_snapshot_caches()
            st.success(f"已删除 {len(selected_paths)} 条失效记录")
            st.rerun()

    with col2:
        if st.button("💥 全部清理", type="secondary"):
            removed = clean_missing_photos(photo_manager, force=True)
            st.success(f"已清理 {len(removed)} 条失效记录")
            st.rerun()

    st.markdown("---")
    st.info("💡 「照片索引」页面打开时会自动清理全部失效记录；本页面用于先查看、再有选择地清理。")
