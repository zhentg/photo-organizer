# pages/1_📁_导入照片.py
import streamlit as st
import os
import hashlib
from pathlib import Path

from core.shared import (
    page_config, render_sidebar, get_detector, get_matcher, get_photo_manager,
    BASE_DIR, invalidate_snapshot_caches
)

# 页面配置
page_config("导入照片 - 人脸照片分类工具", "📁")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()

# 主内容
st.title("📁 导入照片")
st.markdown("---")

# 输入方式选择
input_method = st.radio("选择输入方式", ["文件夹路径", "上传图片"], horizontal=True)

photos_to_process = []

if input_method == "文件夹路径":
    folder_path = st.text_input("输入照片文件夹路径", placeholder="例如: D:\\Photos\\2024")

    if folder_path and os.path.isdir(folder_path):
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        photos_to_process = []
        subfolders = []
        for root, dirs, files in os.walk(folder_path):
            if root != folder_path and root not in subfolders:
                subfolders.append(root)
            for f in files:
                if Path(f).suffix.lower() in image_extensions:
                    photos_to_process.append(os.path.join(root, f))
        st.info(f"找到 {len(photos_to_process)} 张照片，扫描了 {len(subfolders)} 个子文件夹")
        if subfolders:
            st.caption("子文件夹: " + " | ".join(subfolders[:10]) + (" ..." if len(subfolders) > 10 else ""))

        # 显示前几张预览
        if photos_to_process:
            st.markdown("**预览（前6张）:**")
            cols = st.columns(6)
            for i, p in enumerate(photos_to_process[:6]):
                with cols[i]:
                    try:
                        st.image(p, use_container_width=True)
                    except:
                        pass
    elif folder_path:
        st.error("路径不存在或无法访问")

else:  # 上传图片
    uploaded_files = st.file_uploader(
        "拖拽或选择图片文件",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True
    )
    if uploaded_files:
        upload_dir = BASE_DIR / "upload"
        upload_dir.mkdir(exist_ok=True)
        for f in uploaded_files:
            save_path = upload_dir / f.name
            with open(save_path, 'wb') as out:
                out.write(f.read())
            photos_to_process.append(str(save_path))
        st.info(f"已上传 {len(photos_to_process)} 张照片到 upload 目录")

# 处理按钮
if photos_to_process and st.button("🚀 开始处理人脸", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = {
        "total": len(photos_to_process),
        "processed": 0,
        "skipped": 0,
        "faces_found": 0,
        "faces_filtered_false": 0,
        "photos_with_faces": 0,
        "auto_matched": 0,
    }

    def calc_md5(path: str) -> str:
        """计算文件 MD5"""
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    for i, photo_path in enumerate(photos_to_process):
        status_text.text(f"处理中: {Path(photo_path).name}")

        # MD5 去重
        md5 = calc_md5(photo_path)
        existing = photo_manager.check_md5_exists(md5)
        if existing:
            results["skipped"] += 1
            progress_bar.progress((i + 1) / len(photos_to_process))
            continue

        # 检测人脸
        faces = detector.detect_faces(photo_path)
        results["processed"] += 1

        # 过滤已知假人脸
        real_faces = []
        for face in faces:
            if matcher._is_false_face(face["encoding"]):
                results["faces_filtered_false"] += 1
            else:
                real_faces.append(face)
        faces = real_faces

        results["faces_found"] += len(faces)
        if faces:
            results["photos_with_faces"] += 1

        # 自动匹配
        persons = []
        for face in faces:
            matched_name, dist = matcher.find_match(face["encoding"])
            if matched_name is not None and dist <= params["match_threshold"]:
                persons.append(matched_name)
                results["auto_matched"] += 1
            else:
                persons.append(None)

        known_persons = [p for p in persons if p is not None]

        # 只要有任意一张脸没自动匹配上，整张照片都必须留在标注队列里。
        # get_unlabeled_photos() 只捞 persons 为空的照片——旧实现把已匹配的
        # 少数名字写进 persons 后，同照片里其余未匹配的脸就被永久藏起来了
        # （实测库里 271 张这样的照片）。匹配上的脸已经写进 face_db，
        # 标注页会通过 find_match 自动预填、确认时按 bbox 去重，不会重复入库。
        all_matched = bool(faces) and len(known_persons) == len(faces)
        persons_for_index = known_persons if all_matched else []

        # 保存到索引
        photo_manager.add_photo(photo_path, faces, persons_for_index, md5=md5)

        # 同步到 face_db
        for j, face in enumerate(faces):
            if persons[j] is not None:
                matcher.add_person(
                    name=persons[j],
                    encoding=face["encoding"],
                    photo_path=photo_path,
                    face_location=face["location"],
                    bbox=face.get("bbox", (0, 0, 0, 0)),
                )

        progress_bar.progress((i + 1) / len(photos_to_process))

    # 写库后必须刷新进程级快照（坑9）：否则「人脸标注」的未标注列表、
    # sidebar 的人物/照片计数还是导入前的旧值，直到重启进程才更新。
    invalidate_snapshot_caches()

    status_text.text("处理完成！")

    # 显示结果
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("处理照片", results["processed"])
    col2.metric("跳过重复", results["skipped"])
    col3.metric("检测人脸", results["faces_found"])
    col4.metric("含人脸照片", results["photos_with_faces"])
    col5.metric("自动识别", results["auto_matched"])
    col6.metric("排除假人脸", results["faces_filtered_false"])

    if results["faces_filtered_false"] > 0:
        st.success(f"🛡️ 已自动排除 {results['faces_filtered_false']} 个已知假人脸")

    if results["skipped"] > 0:
        st.warning(f"⚠️ 跳过了 {results['skipped']} 张重复照片")
    if results["auto_matched"] > 0:
        st.success(f"✅ 自动识别了 {results['auto_matched']} 个人脸，无需手动标注！")
    elif results["faces_found"] > 0:
        st.info(f"请到「人脸标注」页面为 {results['faces_found']} 个人脸指定姓名")
    elif results["processed"] > 0:
        st.info("未检测到任何人脸，请尝试其他照片")
