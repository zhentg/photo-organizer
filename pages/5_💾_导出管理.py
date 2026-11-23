# pages/5_💾_导出管理.py
import streamlit as st
import os
import json
from pathlib import Path

from core.shared import (
    page_config, render_sidebar, get_detector, get_matcher, get_photo_manager,
    DEFAULT_OUTPUT_DIR, export_photos_by_person, invalidate_snapshot_caches
)

# 页面配置
page_config("导出管理 - 人脸照片分类工具", "💾")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("💾 导出管理")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 导出")

    # 导出人脸库
    st.markdown("**导出人脸库 (face_db.json):**")
    if st.button("导出人脸库"):
        db_data = matcher.export_db()
        st.download_button(
            "下载 face_db.json",
            json.dumps(db_data, ensure_ascii=False, indent=2),
            "face_db.json",
            "application/json",
            key="download_db"
        )

    st.markdown("---")

    # 导出照片索引
    st.markdown("**导出照片索引 (photo_index.json):**")
    if st.button("导出照片索引"):
        index_data = photo_manager.export_index()
        st.download_button(
            "下载 photo_index.json",
            json.dumps(index_data, ensure_ascii=False, indent=2),
            "photo_index.json",
            "application/json",
            key="download_index"
        )

    st.markdown("---")

    # 按人物导出照片
    st.markdown("**按人物导出照片到文件夹:**")
    output_dir = st.text_input("输出目录", str(DEFAULT_OUTPUT_DIR))

    if st.button("🚀 开始导出照片", type="primary"):
        if os.path.isdir(output_dir):
            with st.spinner("导出中..."):
                result = export_photos_by_person(photo_manager, matcher, output_dir)

                if result:
                    st.success("导出完成！")
                    for name, count in result.items():
                        st.write(f"  - {name}: {count} 张")
                    st.info(f"照片已保存到: {output_dir}")
                else:
                    st.info("没有可导出的照片")
        else:
            st.error("输出目录无效")

with col2:
    st.subheader("📤 导入")

    # 导入人脸库
    st.markdown("**导入人脸库 (face_db.json):**")
    uploaded_db = st.file_uploader("选择 face_db.json", type="json", key="db_upload")
    if uploaded_db:
        if st.button("确认导入人脸库"):
            try:
                data = json.load(uploaded_db)
                matcher.import_db(data)
                invalidate_snapshot_caches()   # 坑9：导入后快照必须失效
                st.success("人脸库导入成功！")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")

    st.markdown("---")

    # 导入照片索引
    st.markdown("**导入照片索引 (photo_index.json):**")
    uploaded_index = st.file_uploader("选择 photo_index.json", type="json", key="index_upload")
    if uploaded_index:
        if st.button("确认导入照片索引"):
            try:
                data = json.load(uploaded_index)
                photo_manager.import_index(data)
                invalidate_snapshot_caches()
                st.success("照片索引导入成功！")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")

    st.markdown("---")

    # 重新扫描文件夹
    st.markdown("**重新扫描照片文件夹:**")
    rescan_folder = st.text_input("照片所在文件夹", placeholder="选择包含照片的文件夹", key="rescan_folder")

    if rescan_folder and st.button("🔄 重新扫描", type="primary"):
        if os.path.isdir(rescan_folder):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            # 与「导入照片」页保持一致：递归子文件夹 + 过滤已知假人脸。
            # 单文件夹 listdir 会漏掉子目录，且不拦假脸 → 误检又回到标注队列。
            photos = []
            for root, _dirs, files in os.walk(rescan_folder):
                for f in files:
                    if Path(f).suffix.lower() in image_extensions:
                        photos.append(os.path.join(root, f))

            st.info(f"扫描到 {len(photos)} 张照片...")

            count = 0
            for photo_path in photos:
                faces = detector.detect_faces(photo_path)
                faces = [f for f in faces if not matcher._is_false_face(f["encoding"])]
                if faces:
                    # persons 省略 = 保留已有标注（重新检测不该抹掉人名）
                    photo_manager.add_photo(photo_path, faces)
                    count += 1

            invalidate_snapshot_caches()
            st.success(f"扫描完成！其中 {count} 张包含人脸")
            st.rerun()
        else:
            st.error("文件夹无效")
