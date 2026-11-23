# pages/3_📂_分类浏览.py
import streamlit as st
from pathlib import Path

from core.image_utils import upright_photo_view
from core.rotate_ui import render_rotate_buttons, render_rotate_flash
from core.shared import (
    page_config, render_sidebar, get_matcher, get_photo_manager,
    clean_missing_photos, get_persons_snapshot, invalidate_snapshot_caches,
    STRANGER_NAME, get_stranger_count_cached, get_photos_by_person_cached
)

# 页面配置
page_config("分类浏览 - 人脸照片分类工具", "📂")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("📂 分类浏览")
render_rotate_flash()
st.markdown("---")

# 快照缓存只读，加「陌生人」前先拷贝，避免污染进程级缓存
persons = dict(get_persons_snapshot())

# 「陌生人」是照片级桶标签（无 encodings，不在人脸库人物表里，见
# core/stranger.py）。有标记照片时并入人物列表：照片详情走
# get_photos_by_person 的 LIKE 查询，统计数用占位照片列表的长度。
stranger_count = get_stranger_count_cached()
if stranger_count:
    persons[STRANGER_NAME] = {
        "name": STRANGER_NAME,
        "photos": [{"path": "", "location": None, "bbox": None, "added_at": None}
                   for _ in range(stranger_count)],
    }

if not persons:
    st.info("暂无已标注人物，请先导入照片并标注人脸")
else:
    # 人物照片统计
    st.subheader("人物照片统计")
    cols = st.columns(min(4, len(persons)))
    for i, (name, data) in enumerate(persons.items()):
        with cols[i % 4]:
            # get_all_persons()["photos"] 是「每个编码一条」，多人合照会重复计数；
            # 按 photo_path 去重才是真正的「张」数。陌生人用上面的真实照片数。
            if name == STRANGER_NAME:
                photo_count = stranger_count
            else:
                photo_count = len({p.get("path") for p in data.get("photos", [])
                                   if p.get("path")})
            st.metric(name, f"{photo_count} 张")

    # 选择要查看的人物
    selected_person = st.selectbox("选择要查看的人物", list(persons.keys()))

    # 切换人物时重置页码
    if "last_selected_person" not in st.session_state or st.session_state.last_selected_person != selected_person:
        st.session_state.page_num = 1
        st.session_state.last_selected_person = selected_person

    missing = clean_missing_photos(photo_manager)
    if missing:
        st.warning(f"已清理 {len(missing)} 条失效记录")

    if selected_person and selected_person in persons:
        photos = get_photos_by_person_cached(selected_person)

        if photos:
            st.subheader(f"📷 {selected_person} 的照片 ({len(photos)} 张)")

            # 分页参数
            PAGE_SIZE = st.session_state.get("sb_page_size", 20)
            total = len(photos)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

            if "page_num" not in st.session_state:
                st.session_state.page_num = 1

            page = st.session_state.page_num
            start = (page - 1) * PAGE_SIZE
            end = start + PAGE_SIZE
            page_photos = photos[start:end]

            # 分页控件
            pn_col1, pn_col2, pn_col3, pn_col4 = st.columns([1, 1, 1, 1])
            with pn_col1:
                if st.button("⬅ 上一页", disabled=(page <= 1), width='stretch'):
                    st.session_state.page_num -= 1
                    st.rerun()
            with pn_col2:
                st.markdown(f"<div style='text-align:center;padding-top:6px'>第 **{page}** / **{total_pages}** 页</div>", unsafe_allow_html=True)
            with pn_col3:
                if st.button("下一页 ➡", disabled=(page >= total_pages), width='stretch'):
                    st.session_state.page_num += 1
                    st.rerun()
            with pn_col4:
                new_page = st.number_input("跳转", min_value=1, max_value=total_pages, value=page, key="jump_page")
                if new_page != page:
                    st.session_state.page_num = new_page
                    st.rerun()

            st.markdown(f"<small style='color:gray'>显示第 {start+1}-{min(end, total)} 张</small>", unsafe_allow_html=True)

            # 批量操作栏
            ba_col1, ba_col2 = st.columns([1, 5])
            with ba_col1:
                if st.button("🔄 批量重新识别", disabled=True, key="batch_reid_btn", width='stretch'):
                    pass
            with ba_col2:
                st.caption("勾选照片后点击「批量重新识别」")

            st.markdown("---")

            # checkbox 版本号，提交后递增使旧 key 全部失效
            if "cb_version" not in st.session_state:
                st.session_state.cb_version = 0
            cb_ver = st.session_state.cb_version

            # 照片网格（带复选框）
            cols = st.columns(4)
            for i, photo_info in enumerate(page_photos):
                photo_path = photo_info["path"]
                cb_key = f"cb_{cb_ver}_{page}_{i}"
                with cols[i % 4]:
                    try:
                        # 摆正后显示：EXIF 横持 + 人脸 roll 都算进来，
                        # 浏览时不会看到"躺着的"照片
                        _roll = next((float(f["roll"]) for f in (photo_info.get("faces") or [])
                                      if f.get("roll")), 0.0)
                        _img, _ = upright_photo_view(photo_path, 640, _roll)
                        st.image(_img, width='stretch')
                        st.caption(f"**{Path(photo_path).name}**")
                        # 手动旋转（已标注照片）：转过之后人名按框位置自动搬移，
                        # 人脸库编码一起重写；对不上就只记日志、提示人工补标
                        render_rotate_buttons(
                            detector, matcher, photo_manager, photo_path,
                            photo_info.get("manual_rotate", 0),
                            key_prefix=f"rot3_{cb_ver}_{page}_{i}", compact=True)
                        st.checkbox("选择", key=cb_key, label_visibility="collapsed")
                        st.markdown(f"<small style='color:gray'>{photo_path}</small>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"无法加载: {Path(photo_path).name}")
                        st.markdown(f"<small style='color:gray'>{photo_path}</small>", unsafe_allow_html=True)

            # 批量操作（放在网格下方，利用 callback 处理）
            selected = [page_photos[i]["path"] for i in range(len(page_photos))
                        if st.session_state.get(f"cb_{cb_ver}_{page}_{i}", False)]
            if selected:
                st.markdown(f"已选中 **{len(selected)}** 张照片")
                if st.button(f"🔄 批量重新识别（{len(selected)} 张）", key="batch_reid_go", width='stretch'):
                    for p in selected:
                        try:
                            matcher.remove_photo_encodings(p)
                            photo_manager.update_photo_persons(p, [])
                        except Exception:
                            pass
                    # 递增 checkbox 版本号，旧 key 全部失效，新 checkbox 从 False 开始
                    st.session_state.cb_version += 1
                    invalidate_snapshot_caches()
                    st.success(f"已重置 {len(selected)} 张照片，请重新扫描人脸")
                    st.rerun()
        else:
            st.info("该人物暂无照片")
