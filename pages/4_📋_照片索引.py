# pages/4_📋_照片索引.py
import streamlit as st
import os
import pandas as pd
from pathlib import Path

from core.rotation import rotate_photo, label_of
from core.shared import (
    page_config, render_sidebar, get_matcher, get_photo_manager,
    clean_missing_photos, get_persons_snapshot, invalidate_snapshot_caches,
    get_all_photos_cached, get_index_export_json, get_rotate_log_cached
)

# 页面配置
page_config("照片索引 - 人脸照片分类工具", "📋")

# 渲染侧边栏
detector, matcher, photo_manager, params = render_sidebar()


def _rotate_selected(paths, delta):
    """批量手动旋转（+90 左 / -90 右），转完立刻重检测人脸并刷新索引。"""
    prog = st.progress(0.0, text="正在旋转并重新检测人脸…")
    ok = no_face = 0
    failed = []
    for n, p in enumerate(paths, 1):
        try:
            res = rotate_photo(photo_manager, detector, p, delta, matcher=matcher)
            if res["status"] == "ok":
                ok += 1
            elif res["status"] == "no_face":
                no_face += 1
            else:
                failed.append(f'{Path(p).name}:{res["status"]}')
        except Exception as e:                       # noqa: BLE001
            failed.append(f"{Path(p).name}:{e}")
        prog.progress(n / max(len(paths), 1), text=f"旋转 {n}/{len(paths)}…")
    prog.empty()
    invalidate_snapshot_caches()
    msg = f"已旋转 {len(paths)} 张：{ok} 张正常，{no_face} 张没检出人脸"
    if failed:
        st.session_state["_rot_flash4"] = ("warning",
                                           msg + f"；{len(failed)} 张需人工看：" + "，".join(failed[:6]))
    else:
        st.session_state["_rot_flash4"] = ("success", msg)
    st.rerun()

# 主内容
st.title("📋 照片索引")
_flash4 = st.session_state.pop("_rot_flash4", None)
if _flash4:
    (st.success if _flash4[0] == "success" else st.warning)("🧭 " + _flash4[1])
st.markdown("---")

# 清理不存在的文件
missing = clean_missing_photos(photo_manager)
if missing:
    st.warning(f"已清理 {len(missing)} 条失效记录")

all_photos = get_all_photos_cached()

if not all_photos:
    st.info("暂无已索引的照片")
else:
    # 搜索框
    search_query = st.text_input("🔍 搜索文件名或路径", placeholder="输入关键词...", key="photo_search")

    # 筛选选项
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_person = st.selectbox("按人物筛选", ["全部"] + list(get_persons_snapshot().keys()))
    with col2:
        filter_faces = st.selectbox("按人脸数筛选", ["全部", "无人脸", "有人脸"])
    with col3:
        filter_exists = st.selectbox("按文件状态筛选", ["全部", "文件存在", "文件缺失"])
    with col4:
        filter_rotate = st.selectbox(
            "按手动旋转筛选", ["全部", "手动旋转过", "未手动旋转"],
            help="手动旋转记录在 photos.manual_rotate（逆时针为正），用来集中排查"
                 "「图片被转歪/转错」的问题")

    # 构建表格数据
    table_data = []
    for path, data in all_photos.items():
        faces = data.get("faces", [])
        persons = data.get("persons", [])
        md5 = data.get("md5", "")
        indexed_at = data.get("indexed_at", "")
        man_rot = int(data.get("manual_rotate") or 0)

        # 搜索过滤
        if search_query:
            if search_query.lower() not in path.lower() and search_query.lower() not in Path(path).name.lower():
                continue

        # 应用筛选
        if filter_person != "全部":
            if filter_person not in persons:
                continue

        if filter_faces == "无人脸" and len(faces) > 0:
            continue
        if filter_faces == "有人脸" and len(faces) == 0:
            continue

        if filter_rotate == "手动旋转过" and man_rot == 0:
            continue
        if filter_rotate == "未手动旋转" and man_rot != 0:
            continue

        # 页首 clean_missing_photos() 已保证索引里剩下的路径都存在，
        # 只有用户真的按「文件状态」筛选时才逐条 stat（省 8867 次 os.path.exists）。
        if filter_exists == "全部":
            file_exists = True
        else:
            file_exists = os.path.exists(path)
        if filter_exists == "文件存在" and not file_exists:
            continue
        if filter_exists == "文件缺失" and file_exists:
            continue

        table_data.append({
            "选择": False,
            "路径": path,
            "文件名": Path(path).name,
            "MD5": md5[:8] + "..." if md5 else "-",
            "人脸数": len(faces),
            "标注人物": ", ".join(persons) if persons else "-",
            "手动旋转": label_of(man_rot) if man_rot else "-",
            "索引时间": indexed_at[:16] if indexed_at else "-",
            "文件状态": "✅" if file_exists else "❌"
        })

    st.markdown(f"**共 {len(table_data)} 张照片**")

    if table_data:
        df = pd.DataFrame(table_data)

        # 使用 data_editor 支持选择
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
        selected_paths = edited_df[edited_df["选择"] == True]["路径"].tolist()

        # ── 手动旋转选中照片（无需进标注页也能转，含"没人脸"的照片）──
        rot1, rot2, rot3 = st.columns([1, 1, 2])
        with rot1:
            if st.button(f"↺ 选中左转90°（{len(selected_paths)}）",
                         disabled=not selected_paths, use_container_width=True,
                         help="逆时针转 90°，转完立刻重新检测人脸；可反复点"):
                _rotate_selected(selected_paths, +90)
        with rot2:
            if st.button(f"↻ 选中右转90°（{len(selected_paths)}）",
                         disabled=not selected_paths, use_container_width=True,
                         help="顺时针转 90°，转完立刻重新检测人脸；可反复点"):
                _rotate_selected(selected_paths, -90)
        with rot3:
            st.caption("旋转角写进 photos.manual_rotate（逆时针为正），"
                       "每次操作都记到 manual_rotate_log，可用上方筛选集中排查")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"🗑️ 删除选中 ({len(selected_paths)})", type="primary", disabled=len(selected_paths) == 0):
                for path in selected_paths:
                    photo_manager.remove_photo(path)
                    # 否则 face_db 里还留着这张照片的编码，find_match 会继续拿它
                    # 去匹配新照片（删除记录却污染识别结果）。
                    matcher.remove_photo_encodings(path)
                invalidate_snapshot_caches()
                st.success(f"已删除 {len(selected_paths)} 条记录")
                st.rerun()

        with col2:
            csv = df.drop(columns=["选择"]).to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 导出 CSV", csv, "photo_index.csv", "text/csv")

        with col3:
            # 90MB 的 JSON 备份串只在用户真要导出时才生成（旧实现每轮 rerun
            # 都 json.dumps 一次，冷启动白白多花 2.5s）。生成后走进程级缓存。
            if st.button("📥 准备 JSON 导出", use_container_width=True):
                st.session_state["_p4_json_ready"] = True
            if st.session_state.get("_p4_json_ready"):
                st.download_button("⬇️ 下载 photo_index.json",
                                   get_index_export_json(),
                                   "photo_index.json", "application/json",
                                   use_container_width=True)

    # ── 手动旋转记录（集中排查：谁转的、转前转后几张脸、名字有没有丢）──
    _log = get_rotate_log_cached(300)
    with st.expander(f"🧭 手动旋转记录（{len(_log)} 条）", expanded=False):
        if not _log:
            st.caption("还没有手动旋转记录。在标注页/分类浏览页点「↺ ↻」后会记在这里。")
        else:
            log_df = pd.DataFrame([{
                "时间": r["created_at"][:19].replace("T", " "),
                "文件名": Path(r["path"]).name,
                "原角度": label_of(r["from_deg"]) if r["from_deg"] else "-",
                "新角度": label_of(r["to_deg"]) if r["to_deg"] else "-",
                "脸(前→后)": f'{r["faces_before"]}→{r["faces_after"]}',
                "名字(前)": ", ".join(r["names_before"]) or "-",
                "名字(后)": ", ".join(r["names_after"]) or "-",
                "结果": r["status"],
                "备注": r["note"] or "",
                "路径": r["path"],
            } for r in _log])
            st.dataframe(log_df, hide_index=True, use_container_width=True, height=260)
            st.download_button(
                "📥 导出手动旋转记录 CSV",
                log_df.to_csv(index=False).encode("utf-8-sig"),
                "manual_rotate_log.csv", "text/csv")
            bad = [r for r in _log if r["status"] not in ("ok",)]
            if bad:
                st.caption(f"⚠️ 其中 {len(bad)} 条非 ok（names_lost / no_face / "
                           f"too_many_faces / encoding_partial），按「结果」列筛一下")

