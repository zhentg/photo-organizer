# shared.py - 共享代码（初始化、辅助函数、sidebar）
import streamlit as st
import os
import sys
import json
import shutil
import hashlib
import time
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import FaceDetector, FaceMatcher, PhotoManager, extract_face_image, draw_faces_on_image, export_photos_by_person
from core.name_search import filter_persons
from core.name_combobox import render_name_combobox
from core.stranger import STRANGER_NAME, mark_photo_as_strangers

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
BASE_DIR = PROJECT_ROOT
MODELS_DIR = BASE_DIR / "models"
DATABASES_DIR = BASE_DIR / "databases"
FACE_DB_PATH = DATABASES_DIR / "face_db.json"
PHOTO_INDEX_PATH = DATABASES_DIR / "photo_index.json"
PHOTO_INDEX_DB = DATABASES_DIR / "photo_index.db"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# 确保目录存在
MODELS_DIR.mkdir(exist_ok=True)
DATABASES_DIR.mkdir(exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 缓存组件
# ─────────────────────────────────────────────

@st.cache_resource
def get_detector(yunet_threshold: float, yunet_nms: float, yunet_min_face: int,
                 eye_validation: bool, strict_mode: bool,
                 opencv_dnn_conf: float, opencv_dnn_min_face: int,
                 prefer_opencv_dnn: bool = False,
                 yolo_conf: float = 0.5,
                 fallback_dnn_conf: float = 0.12):
    return FaceDetector(
        yunet_threshold=yunet_threshold,
        yunet_nms_threshold=yunet_nms,
        yunet_min_face=yunet_min_face,
        eye_validation=eye_validation,
        strict_mode=strict_mode,
        opencv_dnn_conf=opencv_dnn_conf,
        opencv_dnn_min_face=opencv_dnn_min_face,
        prefer_opencv_dnn=prefer_opencv_dnn,
        yolo_conf=yolo_conf,
        fallback_dnn_conf=fallback_dnn_conf,
        # 手动旋转角存在照片索引库：给检测器路径，detect_faces 才会自动
        # 应用用户点过的方向（不给 = 每次重检测都会把人工纠正抹掉）
        photo_index_path=str(PHOTO_INDEX_DB),
    )

@st.cache_resource
def get_matcher(match_threshold: float):
    return FaceMatcher(str(FACE_DB_PATH), match_threshold=match_threshold)


# ─────────────────────────────────────────────
# 失效记录清理（2026-11-11 修复标注页刷新变慢）
#
# 旧实现：页面用 st.session_state["_missing_cleaned"] 做「每会话一次」，
# 但 F5 会新建 session → 每次刷新都重跑 get_all_photos()（8867 行
# faces/persons JSON 全解析 ~2s）+ 8867 次 os.path.exists（~0.2-1.3s），
# 标注页刷新实测 2.9s（进程冷启动 7.3s）。
# 现在：只查 path 列；去重提到进程级（DB 签名 + TTL），F5 不再重扫。
# ─────────────────────────────────────────────
_MISSING_CHECK = {"sig": None, "at": 0.0, "db": None}
_MISSING_CHECK_TTL = 60.0


def _photo_db_signature(pm: PhotoManager):
    """索引库文件的 (mtime_ns, size)；外部脚本改过库就能被发现。"""
    try:
        stt = os.stat(pm.db_path)
        return (stt.st_mtime_ns, stt.st_size)
    except OSError:
        return None


def clean_missing_photos(pm: PhotoManager, batch_size: int = 200,
                         force: bool = False) -> list:
    """删除索引里源文件已不存在的记录，返回被删路径列表。

    - 只查 path 列，不解析任何 JSON；
    - 进程级去重：DB 未变且距上次检查 < 60s 直接跳过，
      因此 F5 / 切页面 / 交互 rerun 都不会重复全盘扫描；
    - force=True 强制执行真实清理（「缺失文件」页「全部清理」按钮用）。
    """
    global _MISSING_CHECK
    now = time.time()
    sig = _photo_db_signature(pm)
    if (not force and sig is not None
            and _MISSING_CHECK["db"] == pm.db_path
            and _MISSING_CHECK["sig"] == sig
            and now - _MISSING_CHECK["at"] < _MISSING_CHECK_TTL):
        return []

    removed = [p for p in pm.get_all_photo_paths() if not os.path.exists(p)]
    for photo_path in removed:
        pm.remove_photo(photo_path)

    # 删除会改 mtime，签名要在删完之后重取，否则下轮会白扫一次
    _MISSING_CHECK = {"sig": _photo_db_signature(pm), "at": now,
                      "db": pm.db_path}
    if removed:
        # 快照缓存是进程级的，删了记录必须失效（旧实现漏了这步）
        invalidate_snapshot_caches()
    return removed


@st.cache_resource
def get_photo_manager():
    return PhotoManager(str(PHOTO_INDEX_PATH), str(DEFAULT_OUTPUT_DIR))

# ─────────────────────────────────────────────
# 快照缓存（2026-10-18 性能优化）
#
# 背景：Streamlit 每次交互都全量重跑页面脚本，sidebar + 页面主体合计要调
# 2-3 次 get_all_persons（~180ms/次）和 get_unlabeled_photos（~670ms/次）。
#
# 用版本号做缓存键（st.cache_data 是进程级缓存，跨会话/标签页共享）：
# 任何写操作后调用 invalidate_snapshot_caches()，所有页面立即刷新。
# 注意：返回结构只读，页面代码不要修改它。
# ─────────────────────────────────────────────
_SNAPSHOT_VERSION = 0


@st.cache_data
def _persons_snapshot(_version: int):
    return FaceMatcher(str(FACE_DB_PATH)).get_all_persons()


@st.cache_data
def _unlabeled_photos_snapshot(_version: int):
    return get_photo_manager().get_unlabeled_photos()


# ── 2026-11-11 全页体检新增：页面 3/4/6/7/8 的全表扫描统一进快照 ──
# 实测（8867 张照片 / 3664 编码）：
#   get_all_photos 2030ms、get_duplicate_photos 2146ms、get_missing_photos 438ms、
#   get_photos_by_person 1579ms、get_false_faces 269ms、索引 JSON 导出 2769ms。
# 这些原本每轮 rerun 都重跑一遍；进 st.cache_data 后只剩一次 pickle 拷贝
# （all_photos 26MB ≈ 320ms，其余 <10ms）。
@st.cache_data
def _all_photos_snapshot(_version: int):
    return get_photo_manager().get_all_photos()


@st.cache_data
def _duplicate_photos_snapshot(_version: int):
    return get_photo_manager().get_duplicate_photos()


@st.cache_data
def _missing_photos_snapshot(_version: int):
    return get_photo_manager().get_missing_photos()


@st.cache_data
def _false_faces_snapshot(_version: int):
    return FaceMatcher(str(FACE_DB_PATH)).get_false_faces()


@st.cache_data
def _photos_by_person_snapshot(_version: int, name: str):
    return get_photo_manager().get_photos_by_person(name)


@st.cache_data
def _stranger_count_snapshot(_version: int):
    return get_photo_manager().get_stranger_photo_count()


@st.cache_data
def _rotate_log_snapshot(_version: int, limit: int):
    return get_photo_manager().get_manual_rotation_log(limit=limit)


@st.cache_data
def _index_export_json(_version: int) -> str:
    """照片索引导出串（~84MB）。只在快照版本变化时才重新序列化。

    顺带修正格式：export_index()/import_index() 用的是 {"photos": {...}}，
    页面 4 原来导出的是裸 dict，导入会得到 0 张照片。
    """
    return json.dumps({"photos": get_all_photos_cached()},
                      ensure_ascii=False, indent=2)


def get_persons_snapshot():
    """人物列表缓存快照（只读）。替代直接调 matcher.get_all_persons()。"""
    return _persons_snapshot(_SNAPSHOT_VERSION)


def get_unlabeled_photos_cached():
    """未标注照片列表缓存快照（只读）。"""
    return _unlabeled_photos_snapshot(_SNAPSHOT_VERSION)


def get_all_photos_cached():
    return _all_photos_snapshot(_SNAPSHOT_VERSION)


def get_duplicate_photos_cached():
    return _duplicate_photos_snapshot(_SNAPSHOT_VERSION)


def get_missing_photos_cached():
    return _missing_photos_snapshot(_SNAPSHOT_VERSION)


def get_false_faces_cached():
    return _false_faces_snapshot(_SNAPSHOT_VERSION)


def get_photos_by_person_cached(name: str):
    return _photos_by_person_snapshot(_SNAPSHOT_VERSION, name)


def get_stranger_count_cached():
    return _stranger_count_snapshot(_SNAPSHOT_VERSION)


def get_rotate_log_cached(limit: int = 300):
    return _rotate_log_snapshot(_SNAPSHOT_VERSION, limit)


def get_index_export_json():
    return _index_export_json(_SNAPSHOT_VERSION)


_SNAPSHOT_FUNCS = (
    _persons_snapshot, _unlabeled_photos_snapshot, _all_photos_snapshot,
    _duplicate_photos_snapshot, _missing_photos_snapshot, _false_faces_snapshot,
    _photos_by_person_snapshot, _stranger_count_snapshot,
    _rotate_log_snapshot, _index_export_json,
)


def invalidate_snapshot_caches():
    """页面写操作（标注确认/匹配/重置/删除/导入等）后调用，刷新全部快照。

    坑9：任何写数据库的代码路径都必须调本函数，否则用户看到过期数据。
    """
    global _SNAPSHOT_VERSION
    _SNAPSHOT_VERSION += 1
    for fn in _SNAPSHOT_FUNCS:
        fn.clear()
    # 写库会改 photo_index.db 的 mtime，失效清理的签名也同步到当前值，
    # 否则紧接着的那轮 rerun 会白扫 8867 个路径（2026-11-11 C1）。
    try:
        gp = get_photo_manager()
        # 只同步「当前应用这个库」的签名；测试里用的临时库不能误覆盖，
        # 否则 temp PM 的去重会被全局库的签名搅乱。
        if _MISSING_CHECK.get("db") == gp.db_path:
            _MISSING_CHECK["sig"] = _photo_db_signature(gp)
            _MISSING_CHECK["at"] = time.time()
    except Exception:
        pass

# ─────────────────────────────────────────────
# 参数转换
# ─────────────────────────────────────────────
YUNET_MAP = {1: 0.7, 2: 0.65, 3: 0.6, 4: 0.55, 5: 0.5,
             6: 0.45, 7: 0.4, 8: 0.35, 9: 0.32, 10: 0.3}

def sensitivity_to_threshold(sensitivity: int) -> float:
    return YUNET_MAP.get(sensitivity, 0.5)

# ─────────────────────────────────────────────
# Sidebar 渲染（每个页面调用）
# ─────────────────────────────────────────────

def render_sidebar():
    """渲染侧边栏，返回 (detector, matcher, photo_manager, params_dict)"""
    with st.sidebar:
        st.header("⚙️ 检测参数")

        # ── 检测器参数 ──
        with st.expander("🔍 人脸检测", expanded=True):
            sensitivity = st.slider(
                "检测灵敏度", 1, 10, 5,
                help="值越高越容易检测到人脸（召回率高），但误检可能增加"
            )
            yunet_threshold = sensitivity_to_threshold(sensitivity)
            yunet_nms = st.slider("NMS阈值（合并重叠框）", 0.1, 0.7, 0.3, 0.05,
                                  help="合并重叠检测框的阈值，越低越严格")
            yunet_min_face = st.slider("最小人脸(px)", 30, 150, 60, 10,
                                       help="低于此尺寸的人脸将被忽略")
            opencv_dnn_conf = st.slider(
                "OpenCV DNN 置信度", 0.05, 0.5, 0.1, 0.05,
                help="OpenCV DNN 检测器的最低置信度阈值。调低可检测更多小人脸，但误检可能增加。YuNet/YOLO 作为备选。"
            )
            opencv_dnn_min_face = st.slider(
                "OpenCV DNN 最小人脸(px)", 20, 100, 30, 5,
                help="OpenCV DNN 检测器的最小人脸尺寸（独立于 YuNet）。调低可检测更多小人脸。"
            )
            yolo_conf = st.slider(
                "YOLO 置信度", 0.1, 0.95, 0.5, 0.05,
                help="YOLOv8n-Face 的置信度阈值。值越低召回越高（但误检可能增加）。推荐 0.5。"
            )
            eye_validation = st.checkbox("✅ 眼睛验证", value=True,
                                          help="检测后额外做眼睛验证，进一步过滤误检")
            strict_mode = st.checkbox("🔒 严格模式", value=True,
                                       help="严格模式：必须检测到眼睛才算有效人脸")

        # ── 匹配参数 ──
        with st.expander("🎯 人脸匹配"):
            match_threshold = st.slider(
                "匹配阈值（余弦距离）", 0.30, 0.80, 0.60, 0.05,
                help="余弦距离小于此值视为同一人。越小越严格（0.6 ≈ cosine similarity > 0.4）")
            st.caption(f"当前阈值: {match_threshold:.2f} (余弦距离)")

        # ── 检测器状态（负责定位人脸） ──
        _dnn_ok = (os.path.exists(FaceDetector.OPENCV_DNN_PROTO) and
                   os.path.exists(FaceDetector.OPENCV_DNN_WEIGHTS))
        _yunet_ok = os.path.exists(FaceDetector.YUNET_MODEL)
        _yolo_ok = os.path.exists(FaceDetector.YOLO_MODEL)
        _sface_ok = os.path.exists(FaceDetector.SFACE_MODEL)
        _arcface_ok = FaceDetector.ARCFACE_MODEL is not None and os.path.exists(FaceDetector.ARCFACE_MODEL)
        st.markdown("**🔍 检测器（定位人脸）:**")
        st.caption(f"- YOLOv8n: {'✅ 首选' if _yolo_ok else '⚠️ 未下载'}")
        st.caption(f"- OpenCV DNN: {'✅ 兜底' if _dnn_ok else '❌ 缺少模型文件'}")
        st.caption(f"- YuNet: {'✅ 备选' if _yunet_ok else '⚠️ 未下载'}")
        st.caption(f"- Haar: ✅ 始终可用")
        st.caption("优先级: YOLO → DNN兜底 → YuNet → Haar")

        # ── 编码器状态（负责提取特征向量） ──
        st.markdown("**🎭 编码器（提取特征向量）:**")
        if _arcface_ok:
            st.caption(f"- ArcFace: ✅ 512维（推荐）")
        else:
            st.caption(f"- ArcFace: ❌ 未下载（128维 SFace 备选）")
        if _sface_ok and not _arcface_ok:
            st.caption(f"- SFace: ✅ 128维（当前使用）")
        elif _sface_ok:
            st.caption(f"- SFace: ✅ 128维（备用）")
        else:
            st.caption(f"- SFace: ❌ 未下载")

        # ── 编码升级 ──
        if _sface_ok or _arcface_ok:
            st.markdown("---")
            st.header("🔄 编码升级")
            enc_desc = "ArcFace 512维" if _arcface_ok else "SFace 128维"
            st.caption(f"将旧编码升级为 {enc_desc} 编码")
            if st.button("🔄 重新编码全部", use_container_width=True):
                detector = get_detector(
                    yunet_threshold=sensitivity_to_threshold(sensitivity),
                    yunet_nms=yunet_nms, yunet_min_face=yunet_min_face,
                    eye_validation=eye_validation, strict_mode=strict_mode,
                    opencv_dnn_conf=opencv_dnn_conf,
                    opencv_dnn_min_face=opencv_dnn_min_face,
                    prefer_opencv_dnn=False,
                    yolo_conf=yolo_conf,
                    fallback_dnn_conf=0.12,
                )
                matcher = get_matcher(match_threshold=float(match_threshold))
                pm = get_photo_manager()
                with st.spinner("正在重新编码人脸库..."):
                    r1 = matcher.re_encode_all(detector)
                with st.spinner("正在重新编码假人脸库..."):
                    r2 = matcher.re_encode_false_faces(detector)
                with st.spinner("正在重新编码照片索引..."):
                    r3 = pm.re_encode_all_photos(detector)
                invalidate_snapshot_caches()
                st.success(
                    f"人脸库: {r1['re_encoded']} 成功 / {r1['failed']} 失败 / {r1['skipped']} 跳过\n\n"
                    f"假人脸: {r2['re_encoded']} 成功 / {r2['failed']} 失败 / {r2['skipped']} 跳过\n\n"
                    f"照片索引: {r3['re_encoded']} 成功 / {r3['failed']} 失败 / {r3['skipped']} 跳过"
                )
                st.rerun()

            # 横持照片专项：2026-10-19 EXIF 摆正改造前的历史编码是侧躺裁剪的
            st.caption("横持拍摄的照片（相机转 90°）旧编码是侧躺裁剪的，匹配效果差")
            if st.button("🔄 重编码横持照片", use_container_width=True):
                detector = get_detector(
                    yunet_threshold=sensitivity_to_threshold(sensitivity),
                    yunet_nms=yunet_nms, yunet_min_face=yunet_min_face,
                    eye_validation=eye_validation, strict_mode=strict_mode,
                    opencv_dnn_conf=opencv_dnn_conf,
                    opencv_dnn_min_face=opencv_dnn_min_face,
                    prefer_opencv_dnn=False,
                    yolo_conf=yolo_conf,
                    fallback_dnn_conf=0.12,
                )
                matcher = get_matcher(match_threshold=float(match_threshold))
                pm = get_photo_manager()
                with st.spinner("正在重编码横持照片的人脸库..."):
                    r1 = matcher.re_encode_all(detector, force_rotated=True)
                with st.spinner("正在重编码横持照片的假人脸库..."):
                    r2 = matcher.re_encode_false_faces(detector, force_rotated=True)
                with st.spinner("正在重编码横持照片的照片索引..."):
                    r3 = pm.re_encode_all_photos(detector, force_rotated=True)
                invalidate_snapshot_caches()
                st.success(
                    f"人脸库: {r1['re_encoded']} 成功 / {r1['failed']} 失败 / {r1['skipped']} 跳过\n\n"
                    f"假人脸: {r2['re_encoded']} 成功 / {r2['failed']} 失败 / {r2['skipped']} 跳过\n\n"
                    f"照片索引: {r3['re_encoded']} 成功 / {r3['failed']} 失败 / {r3['skipped']} 跳过"
                )
                st.rerun()

        st.markdown("---")
        st.header("👤 已标注人物")

        matcher = get_matcher(match_threshold=float(match_threshold))
        persons = get_persons_snapshot()

        if persons:
            for person_name, data in persons.items():
                photo_count = len(data.get("photos", []))
                st.markdown(f"- **{data.get('name', person_name)}** ({photo_count} 张)")

            st.markdown("---")
            if st.button("📥 导出人脸库", use_container_width=True):
                st.session_state['show_export'] = True
            if st.button("📤 导入人脸库", use_container_width=True):
                st.session_state['show_import'] = True
        else:
            st.info("暂无已标注人物")

        # 数据库管理
        st.markdown("---")
        st.header("🔧 数据库管理")

        pm = get_photo_manager()
        idx_count = pm.get_photo_count()
        db_persons = persons   # 复用上方快照（同一作用域，只读）
        person_count = len(db_persons)
        encoding_count = sum(len(p["photos"]) for p in db_persons.values())
        st.caption(f"📋 索引: {idx_count} 张照片  |  🗃️ 人脸库: {person_count} 人 / {encoding_count} 编码")

        reset_target = st.radio(
            "重置范围",
            ["📋 仅清空索引", "🗃️ 仅清空人脸库", "💥 全部清空 (索引+人脸库)"],
            index=0,
            key="reset_target"
        )

        confirm = st.checkbox(
            "⚠️ 确认清空（不可恢复）",
            value=False,
            key="reset_confirm"
        )

        if st.button("🔄 执行重置", type="primary" if confirm else "secondary", 
                     use_container_width=True, disabled=not confirm):
            target = st.session_state.get('reset_target', '')
            results = []

            if '仅清空索引' in target or '全部清空' in target:
                pm.reset_db()
                results.append("📋 索引已清空")

            if '仅清空人脸库' in target or '全部清空' in target:
                matcher.reset_db()
                results.append("🗃️ 人脸库已清空")

            invalidate_snapshot_caches()
            st.success("  \n".join(results))
            st.rerun()

    # 返回组件和参数
    detector = get_detector(
        yunet_threshold=yunet_threshold,
        yunet_nms=yunet_nms,
        yunet_min_face=yunet_min_face,
        eye_validation=eye_validation,
        strict_mode=strict_mode,
        opencv_dnn_conf=opencv_dnn_conf,
        opencv_dnn_min_face=opencv_dnn_min_face,
        prefer_opencv_dnn=False,
        yolo_conf=yolo_conf,
        fallback_dnn_conf=0.12,
    )

    params = {
        "yunet_threshold": yunet_threshold,
        "yunet_nms": yunet_nms,
        "yunet_min_face": yunet_min_face,
        "eye_validation": eye_validation,
        "strict_mode": strict_mode,
        "match_threshold": match_threshold,
        "sensitivity": sensitivity,
        "opencv_dnn_conf": opencv_dnn_conf,
        "opencv_dnn_min_face": opencv_dnn_min_face,
        "prefer_opencv_dnn": False,
        "yolo_conf": yolo_conf,
        "fallback_dnn_conf": 0.12,
    }
    
    return detector, matcher, pm, params


def page_config(title: str, icon: str = "📸"):
    """设置页面配置（每个页面开头调用）"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide"
    )
