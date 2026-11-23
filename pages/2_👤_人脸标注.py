# pages/2_👤_人脸标注.py
import re
import streamlit as st
from pathlib import Path
import cv2

from core.image_utils import upright_photo_view
from core.rotation import label_of
from core.rotate_ui import (render_rotate_buttons, render_rotate_flash,
                            clear_photo_session_keys)
from core.shared import (
    page_config, render_sidebar, get_matcher, get_photo_manager,
    extract_face_image, draw_faces_on_image,
    clean_missing_photos,
    get_persons_snapshot, get_unlabeled_photos_cached, invalidate_snapshot_caches,
    render_name_combobox, STRANGER_NAME, mark_photo_as_strangers
)

# 会话键命名空间：前缀 + f"{照片下标}_{人脸下标}"
# （sb_/cb_/rd_ 的约定见 core/name_combobox.py；px_=排除标记；ti_/q_ 为旧版残留）
_KEY_PREFIXES = ("sb_", "cb_", "rd_", "ti_", "q_", "px_")
_KEY_RE = re.compile(r"^(sb_|cb_|rd_|ti_|q_|px_)(\d+)_(\d+)$")


def _remap_keys_after_removal(photos, removed_idxs):
    """照片被确认/移出队列后，把其余照片的会话键按新队列位置重映射。

    Streamlit 每轮全量重跑页面脚本：队列里少一张照片，它后面的照片
    下标全部前移一位——不重映射的话，用户填给照片 B 的姓名会串到
    照片 C 的框里（键还是旧下标），排除标记也会串。批量确认和
    「👤 本张全是陌生人」移出照片后都必须调用。
    """
    removed = set(removed_idxs)
    remaining = [i for i in range(len(photos)) if i not in removed]
    # 1) 取出全部队列相关键（含历史残留），清空
    saved = {}
    for key in list(st.session_state.keys()):
        if _KEY_RE.match(key):
            saved[key] = st.session_state.pop(key)
    # 2) 按新队列位置写回：旧下标 → 新位置
    for new_pos, old_idx in enumerate(remaining):
        for j in range(len(photos[old_idx]["faces"])):
            for pfx in _KEY_PREFIXES:
                old_key = pfx + str(old_idx) + "_" + str(j)
                if old_key in saved:
                    st.session_state[pfx + str(new_pos) + "_" + str(j)] = saved[old_key]


# 页面配置
page_config("人脸标注 - 人脸照片分类工具", "👤")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()
THRESHOLD = float(params["match_threshold"])

# 主内容
st.title("👤 人脸标注")

# ─────────────────────────────────────────────
# 全库匹配按钮（用已知人脸匹配所有未标注照片）
# ─────────────────────────────────────────────
col_match, col_match_info = st.columns([1, 3])
with col_match:
    if st.button("🎯 全库匹配", type="secondary", use_container_width=True,
                 help="用已标注人物匹配所有未标注照片"):
        # 获取所有未标注照片（不限 display_count）
        all_unlabeled = photo_manager.get_unlabeled_photos()
        
        if not all_unlabeled:
            st.warning("没有需要匹配的照片")
        else:
            progress_bar = st.progress(0, text="正在匹配...")
            matched_count = 0
            face_matched_count = 0
            total_photos = len(all_unlabeled)
            
            for i, photo in enumerate(all_unlabeled):
                photo_path = photo["path"]
                faces = photo.get("faces", [])
                
                if not faces:
                    continue
                
                matched_persons = []
                for face in faces:
                    enc = face.get("encoding")
                    if enc:
                        name, dist = matcher.find_match(enc)
                        if name and dist <= THRESHOLD:
                            # 匹配成功，添加到人脸库
                            matcher.add_person(
                                name=name,
                                encoding=enc,
                                photo_path=photo_path,
                                face_location=face.get("location"),
                                bbox=face.get("bbox", (0, 0, 0, 0))
                            )
                            matched_persons.append(name)
                            face_matched_count += 1
                
                if matched_persons:
                    # 更新照片的人物列表
                    photo_manager.update_photo_persons(photo_path, matched_persons)
                    matched_count += 1
                
                # 更新进度条
                progress = (i + 1) / total_photos
                progress_bar.progress(
                    progress,
                    text=f"匹配中... {i+1}/{total_photos} 已匹配 {face_matched_count} 个人脸"
                )
            
            progress_bar.empty()
            if face_matched_count > 0:
                st.success(f"✅ 匹配完成！{total_photos} 张照片中 {matched_count} 张有匹配，共 {face_matched_count} 个人脸")
                invalidate_snapshot_caches()
                st.rerun()
            else:
                st.info(f"匹配完成，{total_photos} 张照片中无匹配结果")

with col_match_info:
    st.caption("点击后，用已标注人物匹配所有未标注照片并自动保存")

st.markdown("---")

# 页面内设置
col_set, col_info = st.columns([1, 3])
with col_set:
    display_count = st.number_input(
        "每页显示", min_value=1, max_value=50, value=3,
        help="每次显示的照片数量"
    )
with col_info:
    st.caption("调整后自动刷新")

st.markdown("---")

# 获取未标注照片
# 清理不存在的文件（进程级去重：F5 新建 session 也不会重扫全库，见 shared.py）
missing = clean_missing_photos(photo_manager)
if missing:
    st.warning(f"已清理 {len(missing)} 条失效记录")

all_photos = get_unlabeled_photos_cached()
photos_with_faces = [{"path": p["path"], "faces": p["faces"], "face_count": len(p["faces"]),
                      "manual_rotate": p.get("manual_rotate", 0) or 0} for p in all_photos]

# ─────────────────────────────────────────────
# 上一轮手动旋转的结果提示（rerun 之后只剩这条消息）
# ─────────────────────────────────────────────
render_rotate_flash()

# ─────────────────────────────────────────────
# 手动旋转待复查：转过但一张脸都没检出，默认不出现在任何队列里
# ─────────────────────────────────────────────
_review = photo_manager.get_rotate_review_photos(limit=12)
if _review:
    with st.expander(f"🧭 手动旋转待复查（{len(_review)} 张：转过但没检出人脸）",
                     expanded=len(_review) <= 4):
        st.caption("这些照片转到了没有脸的方向（或本来就没人）。继续点方向按钮直到摆正，"
                   "或点「⟲ 重置」回到自动判断。角度记录在 photos.manual_rotate。")
        for k, item in enumerate(_review):
            _p = item["path"]
            _rr = int(item["manual_rotate"] or 0)
            c_img, c_side = st.columns([1, 2])
            with c_img:
                try:
                    _im, _ = upright_photo_view(_p, 420, roll=_rr)
                    st.image(_im, width=180)
                except Exception:
                    st.write("（无法加载）")
            with c_side:
                st.markdown(f"**{Path(_p).name}**  ·  {label_of(_rr)}")
                render_rotate_buttons(detector, matcher, photo_manager, _p, _rr,
                                      key_prefix=f"rot_rev_{k}")
                st.markdown("---")

if not photos_with_faces:
    st.success("✅ 所有已检测人脸都已标注完成！")
else:
    # 按文件名排序，只显示前N张
    photos_with_faces.sort(key=lambda x: Path(x["path"]).name.lower())
    total_count = len(photos_with_faces)
    photos_with_faces = photos_with_faces[:display_count]

    # 自动匹配 - 预填姓名到 UI 组件
    # 关键修复：只在 sb_key 尚未设置时才写入预填值，
    # 避免覆盖用户手动修改的选择
    total_faces = sum(p["face_count"] for p in photos_with_faces)
    auto_matched_count = 0
    for idx, photo_data in enumerate(photos_with_faces):
        for i, face in enumerate(photo_data["faces"]):
            sb_key = "sb_" + str(idx) + "_" + str(i)
            cb_key = "cb_" + str(idx) + "_" + str(i)

            # 已有值 = 用户手动选过或之前已预填，不要覆盖
            if sb_key in st.session_state:
                current = st.session_state[sb_key]
                # 老会话状态里可能没有 cb_（combobox 之前的版本），补上；
                # 已存在（用户正在打字或已选）则保持不动——
                # 绝不能再往下走重新预填，否则会擦掉用户输入中的拼音
                st.session_state.setdefault(cb_key, current)
                if current:
                    auto_matched_count += 1
                continue

            # 首次加载：自动匹配并预填（sb=已确认姓名，cb=框内显示文本）
            enc = face["encoding"]
            matched_name, dist = matcher.find_match(enc)
            if matched_name is not None and dist <= THRESHOLD:
                st.session_state[sb_key] = matched_name
                st.session_state[cb_key] = matched_name
                auto_matched_count += 1
            else:
                st.session_state[sb_key] = ""
                st.session_state[cb_key] = ""

    st.info(f"待标注 {total_count} 张照片（当前显示前 {display_count} 张），"
            f"共 {total_faces} 个人脸，其中 **{auto_matched_count} 个已自动匹配并预填姓名**")

    st.markdown("---")

    # 预计算人物列表（避免循环内重复查询）
    all_persons = get_persons_snapshot()
    person_names = sorted(all_persons.keys(),
                          key=lambda n: -len(all_persons[n].get("photos", [])))

    # 照片列表
    for idx, photo_data in enumerate(photos_with_faces):
        photo_path = photo_data["path"]
        faces = photo_data["faces"]

        with st.container():
            # 统计已预填姓名的数量（sb_key 有值且不为空）
            auto_in_photo = sum(
                1 for i in range(len(faces))
                if st.session_state.get("sb_" + str(idx) + "_" + str(i), "")
            )

            st.markdown(f"### {Path(photo_path).name}（{len(faces)} 人，已预填 {auto_in_photo} 个姓名）")

            col_left, col_right = st.columns([1, 2])
            with col_left:
                img_drawn = draw_faces_on_image(photo_path, faces, max_width=600)
                if img_drawn is not None:
                    st.image(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB),
                             caption="检测到的人脸", width=300)
                # 手动旋转：点一次转 90° → 用新方向重检测人脸 → 可连续点
                render_rotate_buttons(detector, matcher, photo_manager, photo_path,
                                      photo_data.get("manual_rotate", 0),
                                      key_prefix=f"rot_{idx}", idx=idx)

            with col_right:

                for i, face in enumerate(faces):
                    prefix = str(idx) + "_" + str(i)

                    col_face, col_name, col_btn = st.columns([1, 3, 1])
                    with col_face:
                        try:
                            st.image(extract_face_image(photo_path, face["location"], max_width=200,
                                                   roll=face.get("roll", 0.0)), width=80)
                        except Exception:
                            st.write("?")
                    with col_name:
                        # 单框 combobox：输入拼音首字母/姓名即筛选，
                        # 点选确认；无匹配可直接把输入作为新姓名
                        render_name_combobox(prefix, person_names, label=f"人脸 {i+1}")
                    with col_btn:
                        false_key = "px_" + str(idx) + "_" + str(i)
                        is_excluded = st.session_state.get(false_key, False)
                        if is_excluded:
                            st.markdown("❌ 已排除")
                            if st.button("↩", key="btn_" + str(idx) + "_" + str(i),
                                        help="取消排除"):
                                st.session_state[false_key] = False
                        else:
                            if st.button("X", key="btn_" + str(idx) + "_" + str(i),
                                        help="标记为非人脸（纳入批量确认）"):
                                st.session_state[false_key] = True

                # 状态提示
                excluded_set = set()
                pending_labels = {}
                for i in range(len(faces)):
                    false_key = "px_" + str(idx) + "_" + str(i)
                    if st.session_state.get(false_key, False):
                        excluded_set.add(i)
                    else:
                        name_val = st.session_state.get("sb_" + str(idx) + "_" + str(i), "")
                        if name_val and name_val.strip():
                            pending_labels[i] = name_val.strip()

                non_excluded = len(faces) - len(excluded_set)
                all_filled = len(pending_labels) == non_excluded and non_excluded > 0
                excl_info = f"，{len(excluded_set)} 个已排除" if excluded_set else ""

                if all_filled:
                    st.success(f"✓ 已命名 {len(pending_labels)} 人{excl_info} — 可批量确认")
                elif pending_labels:
                    unnamed = non_excluded - len(pending_labels)
                    st.info(f"已命名 {len(pending_labels)} 人{excl_info}，{unnamed} 人待命名")
                else:
                    st.caption(f"待命名 {non_excluded} 人{excl_info}")

                # 一键标为陌生人：本张所有未排除人脸 → 「陌生人」，
                # 整张照片移出标注队列（已排除的仍走假脸流程）
                if st.button("👤 本张全是陌生人", key="stranger_" + str(idx),
                             use_container_width=True,
                             help="把本张照片所有未排除的人脸一键标为「陌生人」，"
                                  "整张照片移出标注队列（路人照不用逐个人命名）"):
                    excluded = {i for i in range(len(faces))
                                if st.session_state.get("px_" + str(idx) + "_" + str(i), False)}
                    n = mark_photo_as_strangers(photo_manager, matcher,
                                                photo_path, faces, excluded)
                    _remap_keys_after_removal(photos_with_faces, [idx])
                    st.success(f"已将《{Path(photo_path).name}》的 {n} 个人脸"
                               f"标为「{STRANGER_NAME}」")
                    invalidate_snapshot_caches()
                    st.rerun()

            st.markdown("---")

    # ─────────────────────────────────────────────
    # 批量确认栏（放在页面最底部：改完姓名顺手点，不用回页首）
    # ─────────────────────────────────────────────
    # 批量确认栏样式
    st.markdown("""
    <style>
    .batch-bar {
        background: #f0f2f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border: 1px solid #e0e2e6;
    }
    </style>
    """, unsafe_allow_html=True)

    # 收集待确认
    all_pending = {}
    all_excluded = {}
    for idx, photo_data in enumerate(photos_with_faces):
        photo_path = photo_data["path"]
        faces = photo_data["faces"]
        photo_pending = {}
        photo_excluded = []
        for i in range(len(faces)):
            false_key = "px_" + str(idx) + "_" + str(i)
            if st.session_state.get(false_key, False):
                photo_excluded.append(i)
            else:
                # combobox 的已确认姓名即最终姓名（可以是人物库中没有的新名字）
                name_val = st.session_state.get("sb_" + str(idx) + "_" + str(i), "")
                if name_val and name_val.strip():
                    photo_pending[i] = name_val.strip()
        if photo_pending:
            all_pending[photo_path] = photo_pending
        if photo_excluded:
            all_excluded[photo_path] = photo_excluded

    # 统计可批量确认
    fully_ready = []
    for idx, photo_data in enumerate(photos_with_faces):
        photo_path = photo_data["path"]
        excluded = set(all_excluded.get(photo_path, []))
        labeled = set(all_pending.get(photo_path, {}).keys())
        if excluded | labeled == set(range(len(photo_data["faces"]))):
            fully_ready.append((idx, photo_path))

    total_excluded = sum(len(v) for v in all_excluded.values())

    with st.container():
        col_info, col_batch, col_refresh = st.columns([2, 1, 1])
        with col_info:
            ready_str = f"{len(fully_ready)} 张可确认"
            excluded_str = f"{total_excluded} 个已排除" if total_excluded else ""
            modified = len(all_pending) + len(all_excluded)
            st.markdown(f"**批量：** {modified} 张已处理（{ready_str}" +
                       (f"，{excluded_str}" if excluded_str else "") + "）")
        with col_batch:
            if fully_ready and st.button(f"✅ 批量确认 ({len(fully_ready)} 张)",
                                            type="primary", use_container_width=True):
                for (idx, photo_path) in fully_ready:
                    photo_data = photos_with_faces[idx]
                    faces = photo_data["faces"]
                    excluded = set(all_excluded.get(photo_path, []))
                    pending = all_pending.get(photo_path, {})
                    # 如果所有人脸都被排除，清空照片的 faces
                    if excluded and len(excluded) == len(faces):
                        # 所有人脸都是假的，清空照片记录
                        photo_manager.clear_photo_faces(photo_path)
                        for i, face in enumerate(faces):
                            bbox = face.get("bbox", (0, 0, 0, 0))
                            matcher.add_false_face(
                                encoding=face["encoding"],
                                photo_path=photo_path,
                                face_location=face["location"],
                                bbox=bbox
                            )
                    else:
                        # 有命名的，正常处理
                        persons = [pending.get(i, "") for i in range(len(faces))]
                        # 过滤空字符串
                        valid_persons = [p for p in persons if p]
                        photo_manager.update_photo_persons(photo_path, valid_persons)

                        for i, face in enumerate(faces):
                            if i in pending:
                                matcher.add_person(
                                    name=pending[i],
                                    encoding=face["encoding"],
                                    photo_path=photo_path,
                                    face_location=face["location"],
                                    bbox=face.get("bbox", (0, 0, 0, 0))
                                )
                            elif i in excluded:
                                bbox = face.get("bbox", (0, 0, 0, 0))
                                matcher.add_false_face(
                                    encoding=face["encoding"],
                                    photo_path=photo_path,
                                    face_location=face["location"],
                                    bbox=bbox
                                )
                                photo_manager.remove_face_from_photo(photo_path, bbox)

                # 移出队列后重映射其余照片的会话键，防止状态串到别的照片上
                # （被确认照片的键不写回，等效于旧的逐张 pop）
                _remap_keys_after_removal(
                    photos_with_faces, [idx for idx, _ in fully_ready])
                st.success(f"已批量确认 {len(fully_ready)} 张（其中排除 {total_excluded} 个）")
                invalidate_snapshot_caches()
                st.rerun()
        with col_refresh:
            if st.button("🔄 重新匹配", use_container_width=True):
                # ── 重新检测人脸（用当前检测器参数）──
                pm = get_photo_manager()
                photo_paths = [p["path"] for p in photos_with_faces]
                with st.spinner(f"正在重新检测 {len(photo_paths)} 张照片的人脸..."):
                    results = []
                    for photo_path in photo_paths:
                        # 清除旧人脸数据
                        pm.clear_photo_faces(photo_path)
                        # 用当前检测器参数重新检测
                        new_faces = detector.detect_faces(photo_path)
                        pm.add_photo(photo_path, new_faces)
                        old_count = next(
                            (len(p["faces"]) for p in photos_with_faces if p["path"] == photo_path), 0)
                        results.append((Path(photo_path).name, old_count, len(new_faces)))
                st.info(
                    "检测完成：" +
                    "  |  ".join(f"{name}: {old}→{new}张" for name, old, new in results)
                )
                # ── 清除 UI 缓存并刷新页面 ──
                for idx2, photo_data in enumerate(photos_with_faces):
                    for i in range(len(photo_data["faces"])):
                        for pfx in ("sb_", "cb_", "ti_", "q_", "px_"):
                            st.session_state.pop(pfx + str(idx2) + "_" + str(i), None)
                invalidate_snapshot_caches()
                st.rerun()
