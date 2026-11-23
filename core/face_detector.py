"""
Face Photo Sort - 人脸识别核心模块 v4
- OpenCV DNN ResNet-SSD（首选，大脸特写最佳，泛化能力强）
- YuNet 深度学习检测（备选，误检率低但召回稍弱）
- YOLOv8n-Face 检测（备选，对极端角度更鲁棒）
- Haar Cascade 回退（所有模型均不可用时的最后防线）
- 眼睛验证（对检测结果二次过滤，降低误检）
- 所有检测参数均可配置
"""
import json
import math
import os
import shutil
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Any, List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

try:
    from core.face_index import (get_face_index, note_encoding_added,
                                 note_false_encoding_added, invalidate_face_index)
    from core.image_utils import (extract_face_image as _extract_face_image,
                                  draw_faces_on_image as _draw_faces_on_image,
                                  open_upright, get_orientation,
                                  map_box_to_upright, map_box_to_original,
                                  rotate_expand, map_box_through_affine)
    from core.stranger import STRANGER_NAME
except ImportError:  # 兼容旧脚本以顶层模块方式导入本文件
    from face_index import (get_face_index, note_encoding_added,
                            note_false_encoding_added, invalidate_face_index)
    from image_utils import (extract_face_image as _extract_face_image,
                             draw_faces_on_image as _draw_faces_on_image,
                             open_upright, get_orientation,
                             map_box_to_upright, map_box_to_original,
                             rotate_expand, map_box_through_affine)
    from stranger import STRANGER_NAME

# 假脸黑名单判定阈值（余弦距离，越小越严格）。
# 2026-11-11 前这里直接复用 match_threshold（默认 0.60）——那是「是不是同一个人」
# 的宽松阈值，用它判「是不是同一张误检」会把同一个人的正常照片全过滤掉。
# 黑名单只该拦「几乎一模一样的那个误检框」，故单独用严格阈值。
FALSE_FACE_THRESHOLD = 0.30

# ─────────────────────────────────────────────
# ArcFace ONNX 加载（延迟 import，避免 photo 环境缺少 onnxruntime 时影响 SFace）
# ─────────────────────────────────────────────
_onnxrt = None  # 全局缓存 onnxruntime Session


def _get_onnx_session(model_path: str, providers=None):
    """获取 onnxruntime InferenceSession（全局缓存，避免重复加载）"""
    global _onnxrt
    if _onnxrt is None:
        try:
            import onnxruntime as _ort
            _onnxrt = _ort
        except ImportError:
            return None
    if providers is None:
        providers = ['CPUExecutionProvider']
    try:
        sess = _onnxrt.InferenceSession(model_path, providers=providers)
        return sess
    except Exception as e:
        print(f"[ArcFace] Failed to load ONNX model {model_path}: {e}")
        return None

# ─────────────────────────────────────────────
# 检测器配置（可在实例化后修改）
# ─────────────────────────────────────────────
DEFAULT_YUNET_THRESHOLD = 0.5          # YuNet 置信度阈值
DEFAULT_YUNET_NMS_THRESHOLD = 0.3       # YuNet NMS 阈值
DEFAULT_YUNET_MIN_FACE = 60              # 最小人脸尺寸（像素）
DEFAULT_YOLO_CONF = 0.5                  # YOLOv8n-Face 置信度阈值（首选检测器时）
DEFAULT_FALLBACK_DNN_CONF = 0.12         # OpenCV DNN 兜底阈值（YOLO 失败后召回小脸用）
DEFAULT_MATCH_THRESHOLD = 0.60          # 人脸匹配余弦距离阈值（越小越严格，0.6 ≈ cosine similarity > 0.4）
DEFAULT_EYE_VALIDATION = True            # 是否启用眼睛验证
DEFAULT_DESKEW = True                    # 人脸对齐（编码时按脸裁出来单独转正，不动整张照片）
DEFAULT_DESKEW_THRESHOLD = 12.0          # 双眼连线倾斜超过该角度视为"歪了"（实测 15° 起开始认错人）
DEFAULT_CARDINAL_MIN_SCORE = 0.85        # YuNet 置信度低于该值视为"方向可疑"，走四向探索（正脸实测 0.91+）


def _json_str(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _normalize_rotate(deg: float) -> float:
    """把旋转角归一化到 (-180, 180]，并让 0/±90/±180 保持整值。

    roll 与 manual_rotate 共用这个约定：**逆时针为正**（与 OpenCV/rotate_expand
    一致）。归一是为了让"右转 3 次"（-270）与"左转 1 次"（+90）落到同一个值，
    否则库里会同时出现 +270 和 -90 两种表示，排查时容易看错。
    """
    d = (float(deg) + 180.0) % 360.0 - 180.0
    # 90 的整数倍不该带 1e-14 的尾数（warpAffine 的 cos/sin 有浮点误差）
    if abs(d - round(d)) < 1e-6:
        d = round(d)
    return 0.0 if abs(d) < 1e-6 else float(d)


def _eye_roll(info) -> float:
    """由 YuNet 5 点算双眼连线角（度，逆时针为正，归一化到 ±90）。

    `info` 是 YuNet 的一行：[x, y, w, h, 右眼x, 右眼y, 左眼x, 左眼y, 鼻, 嘴..., score]。
    脸倒 180° 时 YuNet 会把左右眼标签互换，先纠正再算角度，否则会把
    "倒立"误算成 0°（这也是当初整图四向探索会误转 180° 的原因之一）。
    """
    rx, ry = float(info[4]), float(info[5])
    lx, ly = float(info[6]), float(info[7])
    if lx < rx:
        rx, ry, lx, ly = lx, ly, rx, ry
    ang = math.degrees(math.atan2(ly - ry, max(lx - rx, 1e-6)))
    while ang <= -90.0:
        ang += 180.0
    while ang > 90.0:
        ang -= 180.0
    return ang


def _compose_affine(M2, M1):
    """仿射矩阵复合：返回 (M2 ∘ M1)，即先 M1 后 M2（2x3，与 cv2 同约定）。"""
    A2 = np.hstack([M2, np.zeros((2, 1), dtype=np.float64)])
    A1 = np.vstack([M1, [0.0, 0.0, 1.0]])
    return (A2 @ A1)[:2, :]


def _json_loads(s) -> Optional[Any]:
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# SQLite 工具
# ─────────────────────────────────────────────

@contextmanager
def _db_conn(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _db_conn(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL)""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS encodings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                encoding TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                location TEXT,
                bbox TEXT,
                added_at TEXT NOT NULL)""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS photos(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                md5 TEXT,
                faces TEXT,
                persons TEXT DEFAULT '[]',
                indexed_at TEXT NOT NULL)""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS false_faces(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encoding TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                location TEXT,
                bbox TEXT,
                marked_at TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_false_faces_photo ON false_faces(photo_path)")
        # person_id 索引：按人查询（get_all_persons/get_person_photos）否则全表扫
        # 22MB 编码文本，实测无索引时 49 人 ~2s → 有索引 ~50ms（2026-10-18）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_encodings_person ON encodings(person_id)")


def _ensure_indexes(db_path: str):
    """给既有库补建索引（CREATE INDEX IF NOT EXISTS 幂等，已存在时 ~0.1ms）。"""
    with _db_conn(db_path) as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_encodings_person ON encodings(person_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_false_faces_photo ON false_faces(photo_path)")


def _init_photo_index_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _db_conn(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS photos(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                md5 TEXT,
                faces TEXT,
                persons TEXT DEFAULT '[]',
                indexed_at TEXT NOT NULL,
                manual_rotate INTEGER NOT NULL DEFAULT 0,
                manual_rotate_at TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_md5 ON photos(md5)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_manual_rotate "
                     "ON photos(manual_rotate)")
        conn.execute("""CREATE TABLE IF NOT EXISTS manual_rotate_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                from_deg REAL NOT NULL,
                to_deg REAL NOT NULL,
                faces_before INTEGER,
                faces_after INTEGER,
                names_before TEXT,
                names_after TEXT,
                status TEXT,
                note TEXT,
                created_at TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rotate_log_path "
                     "ON manual_rotate_log(path)")


# ─────────────────────────────────────────────
# FaceDetector — 人脸检测（YuNet + YOLO + Haar 三合一）
# ─────────────────────────────────────────────

class FaceDetector:
    """
    多检测器融合人脸检测：

    优先级：OpenCV DNN ResNet-SSD > YuNet > YOLOv8n-Face > Haar Cascade
    OpenCV DNN 为首选检测器（泛化能力强，对测试图片验证正确）。
    眼睛验证：对检测结果进行二次过滤，进一步降低误检率。

    可配置参数（实例化后可直接修改）：
        yunet_threshold      — OpenCV DNN / YuNet 置信度阈值（默认 0.5）
        yunet_nms_threshold — NMS 阈值（默认 0.3）
        yunet_min_face      — 最小人脸尺寸（默认 60px）
        eye_validation      — 是否启用眼睛验证（默认 True）
        strict_mode         — 严格模式：必须有人眼才能通过（默认 False）
        prefer_opencv_dnn   — 优先使用 OpenCV DNN（默认 True，泛化能力最强）
    """

    # OpenCV DNN ResNet-SSD 模型（首选，大脸特写最佳）
    OPENCV_DNN_PROTO = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'models', 'opencv_dnn_face', 'deploy.prototxt')
    OPENCV_DNN_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       'models', 'opencv_dnn_face', 'res10_300x300_ssd_iter_140000.caffemodel')

    YUNET_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'models', 'face_detection_yunet_2023mar.onnx')
    SFACE_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'models', 'face_recognition_sface_2021dec.onnx')
    # ArcFace ONNX 模型（优先使用 w600k_mbf，轻量且效果好）
    # 自动检测多个可能的文件名
    _arcface_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'arcface')
    _arcface_candidates = [
        os.path.join(_arcface_dir, 'w600k_mbf.onnx'),
        os.path.join(_arcface_dir, 'buffalo_l.onnx'),
        os.path.join(_arcface_dir, 'w600k_r50.onnx'),
        os.path.join(_arcface_dir, 'arcface_w600k_r50.onnx'),
        os.path.join(_arcface_dir, 'r100_arcface_ms1m.onnx'),
        os.path.join(_arcface_dir, 'arcface_r100.onnx'),
    ]
    _arcface_detected = None
    for _p in _arcface_candidates:
        if os.path.exists(_p):
            _arcface_detected = _p
            break
    ARCFACE_MODEL = _arcface_detected
    # 自动检测可用 YOLO 模型（.pt 优先于 .onnx）
    YOLO_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'yolov8n-face')
    _yolo_files2 = ['face_yolov8n.pt', 'face_yolov8n_v2.pt', 'yolov8n-face.onnx', 'yolov8n-face.pt']
    _yolo_detected = os.path.join(YOLO_BASE, 'face_yolov8n.pt')  # 默认
    for _f2 in _yolo_files2:
        _p = os.path.join(YOLO_BASE, _f2)
        if os.path.exists(_p):
            _yolo_detected = _p
            break
    YOLO_MODEL = _yolo_detected

    def __init__(self,
                 yunet_threshold: float = DEFAULT_YUNET_THRESHOLD,
                 yunet_nms_threshold: float = DEFAULT_YUNET_NMS_THRESHOLD,
                 yunet_min_face: int = DEFAULT_YUNET_MIN_FACE,
                 eye_validation: bool = DEFAULT_EYE_VALIDATION,
                 strict_mode: bool = False,
                 prefer_opencv_dnn: bool = False,   # 默认 YOLO 优先
                 opencv_dnn_conf: float = 0.12,   # 兜底阈值（YOLO 失败后用）
                 yolo_conf: float = DEFAULT_YOLO_CONF,
                 fallback_dnn_conf: float = DEFAULT_FALLBACK_DNN_CONF,
                 opencv_dnn_min_face: int = 30,
                 deskew: bool = DEFAULT_DESKEW,
                 deskew_threshold: float = DEFAULT_DESKEW_THRESHOLD,
                 cardinal_min_score: float = DEFAULT_CARDINAL_MIN_SCORE,
                 photo_index_path: Optional[str] = None):
        self.yunet_threshold = yunet_threshold
        self.yunet_nms_threshold = yunet_nms_threshold
        self.yunet_min_face = yunet_min_face
        self.eye_validation = eye_validation
        self.strict_mode = strict_mode
        self.prefer_opencv_dnn = prefer_opencv_dnn
        self.opencv_dnn_conf = opencv_dnn_conf   # 主阈值（OpenCV DNN 优先模式下使用）
        self.yolo_conf = yolo_conf               # YOLO 专用 conf
        self.fallback_dnn_conf = fallback_dnn_conf  # 兜底 conf（YOLO 失败后用 OpenCV DNN 时使用）
        self.opencv_dnn_min_face = opencv_dnn_min_face
        self.deskew = deskew                         # 人脸对齐开关（只转裁剪，不动整图）
        self.deskew_threshold = deskew_threshold     # 超过该角度算"歪"（度，仅用于诊断）
        self.cardinal_min_score = cardinal_min_score  # 低于该置信度就试四个方向
        # 照片索引库路径：给了就能读到"用户手动旋转过多少度"（detect_faces 自动应用）。
        # 不给（独立实验脚本/单测）则忽略手动旋转——绝不静默依赖全局配置。
        self.photo_index_path = photo_index_path
        self._rotate_col_ok = None       # photos.manual_rotate 列是否存在（惰性探测）

        # 检测器可用性
        self._opencv_dnn_available = (os.path.exists(self.OPENCV_DNN_PROTO) and
                                       os.path.exists(self.OPENCV_DNN_WEIGHTS))
        self._yunet_available = os.path.exists(self.YUNET_MODEL)
        self._yolo_available = os.path.exists(self.YOLO_MODEL)

        # OpenCV DNN 模型（延迟加载）
        self._opencv_dnn_net = None

        # SFace 人脸识别模型（延迟加载）
        self._sface_recognizer = None
        self._sface_available = os.path.exists(self.SFACE_MODEL)

        # ArcFace ONNX 模型（延迟加载，优先于 SFace）
        self._arcface_session = None
        self._arcface_available = self.ARCFACE_MODEL is not None and os.path.exists(self.ARCFACE_MODEL)
        if self._arcface_available:
            print(f"[FaceDetector] ArcFace model found: {self.ARCFACE_MODEL} (will be used automatically)")
        else:
            print("[FaceDetector] ArcFace model not found — will use SFace (128d). "
                  "See ARCFACE_README.md to enable ArcFace (512d).")

        # Haar 回退用
        self.cascades = {
            'alt':   cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            'alt2':  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
        }
        if not strict_mode:
            self.cascades['profile'] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # YOLO 模型（延迟加载）
        self._yolo_model = None

        # 统计（调试用）
        self._stats = {"opencv_dnn": 0, "yunet": 0, "yolo": 0, "haar": 0,
                       "eye_rejected": 0, "cardinal": 0, "manual": 0, "aligned": 0}

    @property
    def yolo_loaded(self) -> bool:
        return self._yolo_available

    @property
    def stats(self) -> Dict:
        return dict(self._stats)

    def _load_yolo(self):
        """延迟加载 YOLO 模型（节省启动时间）"""
        if self._yolo_model is None and self._yolo_available:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(self.YOLO_MODEL)
                self._yolo_model.overrides['verbose'] = False
            except Exception as e:
                self._yolo_available = False
                print(f"[FaceDetector] YOLO load failed: {e}")

    def _manual_rotate_for(self, image_path: str) -> float:
        """读该照片的"手动旋转角"（度，逆时针为正，与 roll 同向）。

        只在配置了 photo_index_path 时读库；读不到/读失败一律返回 0 ——
        宁可少转，也不要因为库被锁住就把整批照片判歪。
        """
        if not self.photo_index_path:
            return 0.0
        try:
            conn = sqlite3.connect(self.photo_index_path, timeout=5.0)
            try:
                if self._rotate_col_ok is None:
                    cols = [r[1] for r in conn.execute(
                        "PRAGMA table_info(photos)").fetchall()]
                    self._rotate_col_ok = "manual_rotate" in cols
                if not self._rotate_col_ok:
                    return 0.0
                row = conn.execute(
                    "SELECT manual_rotate FROM photos WHERE path=?",
                    (image_path,)).fetchone()
            finally:
                conn.close()
        except Exception:
            return 0.0
        try:
            return _normalize_rotate(float(row[0])) if row and row[0] else 0.0
        except (TypeError, ValueError):
            return 0.0

    def detect_faces(self, image_path: str,
                     manual_rotate: Optional[float] = None,
                     allow_cardinal: Optional[bool] = None) -> List[Dict]:
        """返回检测到的人脸列表

        照片方向只按 **90° 的整数倍** 纠正，顺序固定（后者建立在前者坐标系上）：
        0. **手动旋转**（用户在界面上点按钮决定的角度，必为 ±90/180）——
           最先应用、优先级最高；
        1. EXIF 摆正（Orientation≠1，横持拍摄）——在摆正图上检测/编码；
        2. 四向探索（90/270）——无 EXIF 的躺倒照片，选"脸能立住"的方向。

        脸上残余的倾斜（十几度、不是 90 的倍数）**不转整张照片**：在编码
        阶段按脸裁出来单独对齐（`_encode_face_aligned`）。存库的 "roll"
        因此永远是 90 的整数倍，预览/缩略图不会出现黑角。

        跑完后 bbox 一律映射回**原始像素坐标系**存库（与旧数据、
        extract_face_image、画框全部兼容）。

        Args:
            manual_rotate: 手动旋转角。None = 从照片索引库读（本实例配了
                photo_index_path 时）；显式传 0 表示忽略库里的值。
            allow_cardinal: 是否允许四向探索。None = 手动旋转过就不探索
                （方向由用户拍板，否则点 4 次会被自动逻辑又转回去）。
        """
        try:
            img, orientation, _orig_size = open_upright(image_path)
            arr = np.array(img)
        except Exception as e:
            print(f"[FaceDetector] Cannot open {image_path}: {e}")
            return []

        # BGR 图像（SFace 需要）
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        exif_h, exif_w = arr.shape[:2]   # EXIF 摆正后、手动旋转前的尺寸

        # ── 第 0 级：手动旋转 ──
        manual = (self._manual_rotate_for(image_path)
                  if manual_rotate is None else _normalize_rotate(manual_rotate))
        M_manual = None                  # 手动旋转矩阵（原 EXIF 空间 -> 检测空间）
        if manual:
            bgr, M_manual = rotate_expand(bgr, manual)
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._stats["manual"] = self._stats.get("manual", 0) + 1

        up_h, up_w = arr.shape[:2]       # 当前检测空间的尺寸（手动旋转之后）

        faces, face_infos, backend = self._detect_chain(arr)

        # ── 第 1 级：整图旋转只允许 90° 的整数倍 ──
        # 2026-10-21 用户要求：照片旋转必须是 90 的整数倍。
        # 以前"脸歪了就把整张图转 -15°"会让预览出现黑角、框也跟着偏移；
        # 现在整图只做方向纠正（四向探索只留 90/270），脸上的倾斜改在
        # 编码阶段按脸裁出来对齐（见 _encode_face_aligned），照片本身不动。
        roll = 0.0
        rot_M = None
        probe_cache = {}          # bbox -> (roll, yunet分)，量过就别再量第二次
        if self.deskew and faces:
            largest = max(faces, key=lambda b: b[2] * b[3])
            probe = self._probe_face(arr, largest)      # (roll, yunet_score) | None
            probe_cache[tuple(int(v) for v in largest)] = probe
            base_score = probe[1] if probe else 0.0

            # 侧躺照片的另一个铁证：脸框"横着"（宽 > 高）。
            # 正立的人脸框永远竖着（宽高比 ~0.75-0.9），横框基本只有侧躺才会出现。
            # 这条闸门挡住 180° 翻转误触发（正立脸被转 180 后 YuNet 仍可能给出
            # 高分，实测 67de00.jpg 就被误转过）。
            sideways_box = largest[2] >= largest[3] * 1.05
            # 两个入口：(a) 主检测器 yolo 没命中、掉到 dnn/haar 兜底 —— 方向多半不对，
            # 兜底出来的框（实测 563040 出 46 个屏幕小脸）不能信；
            # (b) yolo 命中了但也没把握、且脸框是横的。
            # 关键点立不住（probe=None）本身就是强信号：侧躺的脸上 YuNet
            # 常常一个点都给不出来（b18c40.jpg 就是这种：YOLO 检出、YuNet
            # 无点、框恰好方方正正，之前两道闸门都放过了它）。
            # 2026-10-20：yunet 命中也算可疑。101bcc0.jpg 侧躺、YOLO+DNN 全漏、
            # YuNet 只在草地上捡到一个 62×68 的竖框（分 0.82）—— 框是竖的，
            # "横框"闸门不触发，于是一整张"两个人站着"的照片被判成 1 张脸。
            # 首选检测器（YOLO）缺席本身就是方向不可信的证据，不必再看框形。
            suspicious = (probe is None) or (backend != "yolo") or (
                base_score < self.cardinal_min_score and sideways_box)
            # 手动旋转过就不再四向探索：用户已经拍板方向，自动逻辑再插一脚
            # 会让"点四次回到原点"，也没法解释给用户听
            if allow_cardinal is not None:
                cardinal_on = allow_cardinal
            else:
                cardinal_on = not manual
            hit = None
            if suspicious and cardinal_on:
                hit = self._try_cardinal(bgr, base_score)
            if hit is not None:
                bgr, arr, faces, face_infos, rot_M, roll = hit
                probe_cache = {}      # 换了坐标系，旧框缓存全部作废
                self._stats["cardinal"] = self._stats.get("cardinal", 0) + 1

        # 检测空间 -> EXIF 摆正空间：先撤整图旋转，再撤手动旋转
        M_space = rot_M
        if M_manual is not None:
            M_space = (M_manual if M_space is None
                       else _compose_affine(M_space, M_manual))

        # 存库 roll = 手动 + 自动（两者都是"逆时针转多少度才正"）
        total_roll = _normalize_rotate(manual + roll)

        # ── 构建结果 ──
        has_yunet_infos = len(face_infos) == len(faces)
        Minv = cv2.invertAffineTransform(M_space) if M_space is not None else None
        result = []
        for idx, (x, y, w, h) in enumerate(faces):
            # 编码用当前（已定向）图里的框
            up_bbox = (int(x), int(y), int(w), int(h))

            # 这张脸歪不歪？歪了才走"裁剪级对齐"，正脸保持原来的编码路径
            # （8k 张老照片的编码都是老路径算的，动了会整体漂移）
            tilted = False
            if self.deskew:
                tb = (int(x), int(y), int(w), int(h))
                if tb in probe_cache:
                    pr = probe_cache[tb]
                else:
                    pr = self._probe_face(arr, up_bbox)
                    probe_cache[tb] = pr
                tilted = pr is not None and abs(pr[0]) >= self.deskew_threshold

            if tilted:
                encoding = self._encode_face_aligned(bgr, up_bbox)
            elif has_yunet_infos:
                encoding = self._arcface_encoding(bgr, face_info=face_infos[idx], bbox=up_bbox)
                if encoding is None:
                    encoding = self._sface_encoding(bgr, face_info=face_infos[idx], bbox=up_bbox)
            else:
                encoding = self._arcface_encoding(bgr, face_info=None, bbox=up_bbox)
                if encoding is None:
                    encoding = self._sface_encoding(bgr, face_info=None, bbox=up_bbox)
            if encoding is None:
                continue

            # 检测空间 -> EXIF 摆正空间（再映射回原图像素空间存库）
            src_bbox = up_bbox
            if Minv is not None:
                cx, cy, cw, ch = map_box_through_affine(up_bbox, Minv)
                cx, cy = max(0, cx), max(0, cy)
                cw = min(cw, exif_w - cx)
                ch = min(ch, exif_h - cy)
                src_bbox = (cx, cy, cw, ch) if cw > 0 and ch > 0 else up_bbox

            bx, by, bw, bh = map_box_to_original(src_bbox, orientation, exif_w, exif_h)
            location = (int(by), int(bx + bw), int(by + bh), int(bx))
            item = {
                "location": location,
                "encoding": encoding,
                "image_path": image_path,
                "bbox": (int(bx), int(by), int(bw), int(bh))
            }
            if total_roll:
                item["roll"] = round(float(total_roll), 1)
            if manual:
                # 排查用：这张脸的角度里有多少是"人手动转的"（不是算法判断的）
                item["manual_rotate"] = round(float(manual), 1)
            result.append(item)
        return result

    def _try_cardinal(self, bgr: np.ndarray, base_score: float = 0.0):
        """四向探索：照片躺着时挑一个"脸能立住"的方向。

        返回 (bgr, arr, faces, infos, M, roll)；都不合格返回 None（保持原图）。

        判据三条一起看（都不可少）：
        1. **关键点要能立起来**（_probe_face 不为 None）且残余角 |roll| < 45°——
           倒着的脸 YuNet 有时仍能给出高分（实测 7d2940 倒 180° 也有 0.868），
           但关键点会明显歪掉，这是最可靠的区分；
        2. **裁剪单脸置信度**要够高，且比原方向明显更高（≥ max(0.86, 原方向+0.08)）；
           注意必须用"裁出这张脸再量"的分——整图缩到 640 时小脸直接漏检，
           实测 7d2940 正确方向整图只有 0.0、错误方向反而 0.868，整图分不可用；
        3. 优先选检出脸更多的方向（组照躺倒时少数方向会漏人）。

        2026-10-20：**不再试 180°**。倒着的脸 YuNet 照样给 0.93~0.95 的检测分
        （86d400.jpg 七个人正着站的合影，转 180° 后抽到的脸分 0.938 > 0.86 阈值），
        而 `_probe_face` 会把左右眼标签互换、把 180° 归一到 0°，|roll|<45 这条
        也就拦不住了 —— 结果是把三张正立合影（86d400 / 10f7440 / fe87c0）转成了
        倒立。倒立照片极罕见，转错一张废一张，收益远小于风险；真要修只能人工。
        """
        best = None            # (score, n_faces, deg, rot, arr, fs, infos, M)
        for deg in (90, 270):
            rot, M = rotate_expand(bgr, deg)
            arr = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
            fs, infos, _bk = self._detect_chain(arr)
            if not fs:
                continue
            # 探最大的前两张脸：组照里最大那张可能是侧脸/被挡住，YuNet
            # 给不出关键点（75f800 正确方向上最大脸 probe=None），此时只看
            # 最大脸会误判"这个方向不行"。
            best_hit = None
            for b in sorted(fs, key=lambda v: v[2] * v[3], reverse=True)[:2]:
                hit = self._probe_face(arr, b)
                if hit is None or abs(hit[0]) >= 45:
                    continue
                if best_hit is None or hit[1] > best_hit[0]:
                    best_hit = (hit[1], hit[0])
            if best_hit is None:
                continue
            score, r = best_hit
            if score < max(0.86, base_score + 0.08):
                continue
            cand = (score, len(fs), deg, rot, arr, fs, infos, M)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        if best is None:
            return None
        _sc, _n, deg, rot, arr, fs, infos, M = best
        # roll 存"要把照片转多少度才正"（270 -> -90），不是转正后的残余角
        roll = deg if deg <= 180 else deg - 360
        return rot, arr, fs, infos, M, roll

    def estimate_roll(self, arr: np.ndarray, bbox: tuple) -> float:
        """量单张脸的 roll 角（度）。正 = 画面里脸顺时针歪，0 = 正。

        量不到关键点时返回 0.0（宁可不转，也不要凭猜测旋转）。
        需要区分"量不到"和"角度为 0"时用 `_measure_roll()`。
        """
        r = self._measure_roll(arr, bbox)
        return 0.0 if r is None else r

    def _measure_roll(self, arr: np.ndarray, bbox: tuple):
        """只取 roll 角；YuNet 在这张脸上找不到 5 点时返回 None。"""
        hit = self._probe_face(arr, bbox)
        return None if hit is None else hit[0]

    def _probe_face(self, arr: np.ndarray, bbox: tuple):
        """裁出单张脸量 (roll, YuNet置信度)；找不到关键点返回 None。

        **置信度才是判断"照片是不是躺着"的主判据**：正常正脸 YuNet 稳定在
        0.91~0.95（抽 12 张实测 min 0.912）；侧躺 90° 的脸只有 0.6 左右，
        而且关键点会"假装"水平（7d2940 在 0° 上量出 -5.7°，单看角度会漏判）。
        """
        try:
            x, y, w, h = [int(v) for v in bbox]
            H, W = arr.shape[:2]
            pad = int(max(w, h) * 0.35)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = arr[y0:y1, x0:x1]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]
            # 统一缩放到长边 640：大脸裁剪（1400px）直接喂 YuNet 很慢，
            # 小脸放大到 640 反而提精度；两端都收益
            s = min(4.0, 640.0 / max(ch, cw))
            if abs(s - 1.0) > 0.05:
                crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))))
            _, infos = self._detect_yunet(crop)
            if not infos:
                return None
            f = max(infos, key=lambda v: v[2] * v[3])
            return _eye_roll(f), float(f[-1])
        except Exception:
            return None

    def _upright_face_crop(self, bgr_img, bbox, max_iter: int = 3):
        """裁出单张脸并**只在这块裁剪里**逐步转正，返回 (crop, boxes, infos)。

        整张照片不会因为脸歪而被旋转（用户要求：照片旋转必须是 90° 的整数倍），
        倾斜在编码阶段按脸消化：裁出来 → 量双眼角 → 转正 → 再量，直到 |角|<5°
        或者没有关键点。全程只影响这张脸的 112×112 输入，与照片方向无关。
        """
        x, y, w, h = [int(v) for v in bbox]
        H, W = bgr_img.shape[:2]
        pad = int(max(w, h) * 0.35)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = bgr_img[y0:y1, x0:x1]
        if crop.size == 0:
            return None, [], []
        ch, cw = crop.shape[:2]
        # 长边统一到 640：大脸裁剪（1400px）直接喂 YuNet 很慢，小脸放大反而更准
        s = min(4.0, 640.0 / max(ch, cw))
        if abs(s - 1.0) > 0.05:
            crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))))

        boxes, infos = [], []
        for _ in range(max(1, max_iter)):
            boxes, infos = self._detect_yunet(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not infos:
                break
            f = max(infos, key=lambda v: v[2] * v[3])
            ang = _eye_roll(f)
            if abs(ang) < 5.0:
                break
            crop, _M = rotate_expand(crop, ang)
        else:
            # 最后一轮是"转完就退出"，框和 crop 已经错位 → 再检一次对齐
            boxes, infos = self._detect_yunet(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return crop, boxes, infos

    def _encode_face_aligned(self, bgr_img, bbox) -> Optional[List[float]]:
        """按 bbox 裁出这张脸 → 摆正 → ArcFace 编码（照片本身不动）。

        先用 YuNet 关键点做 5 点相似变换（alignCrop，最准），拿不到关键点就
        退回原来的"bbox 直接缩放"，再不行交给 SFace。
        """
        if self.deskew:
            try:
                crop, boxes, infos = self._upright_face_crop(bgr_img, bbox)
            except Exception:
                crop, boxes, infos = None, [], []
            if crop is not None and infos:
                i = max(range(len(boxes)), key=lambda k: boxes[k][2] * boxes[k][3])
                enc = self._arcface_encoding(crop, face_info=infos[i], bbox=boxes[i])
                if enc is None:
                    enc = self._sface_encoding(crop, face_info=infos[i], bbox=boxes[i])
                if enc is not None:
                    self._stats["aligned"] = self._stats.get("aligned", 0) + 1
                    return enc
        enc = self._arcface_encoding(bgr_img, face_info=None, bbox=bbox)
        return enc if enc is not None else self._sface_encoding(bgr_img, None, bbox)

    def encode_region(self, bgr_img, bbox, roll: float = 0.0) -> Optional[List[float]]:
        """按 bbox 编一张已存在的人脸（重编码用）。

        roll 现在只应该是 90 的整数倍（照片方向）。裁剪内部再按脸对齐，
        与 detect_faces 产出的编码走的是同一条路径，互相可比值。
        """
        if not roll or abs(roll) < 1.0:
            return self._encode_face_aligned(bgr_img, bbox)

        x, y, w, h = [int(v) for v in bbox]
        H, W = bgr_img.shape[:2]
        pad = int(max(w, h) * 0.35)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = bgr_img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        rot, M = rotate_expand(crop, roll)
        bx = map_box_through_affine((x - x0, y - y0, w, h), M)
        enc = self._encode_face_aligned(rot, bx)
        return enc if enc is not None else self._sface_encoding(rot, None, bx)

    def _detect_chain(self, arr: np.ndarray) -> Tuple[List[Tuple], List, str]:
        """按优先级跑检测器 + 眼睛验证，返回 (boxes, yunet_infos, backend)。

        backend 是真正命中的检测器名（yolo/opencv_dnn/yunet/haar/none）——
        **它不是细节**：yolo 命中说明画面方向基本正常；掉到 dnn/haar 往往
        意味着方向不对（侧躺），此时该去看四个方向而不是信这些框。
        """
        # 优先级：YOLO > OpenCV DNN > YuNet > DNN 低阈值 > Haar
        face_infos = []  # YuNet 原始数据（用于 alignCrop）
        faces = []
        backend = "none"

        # 1. YOLOv8n-Face（首选，对多张小脸/侧脸最鲁棒，速度最快）
        if not self.prefer_opencv_dnn and self._yolo_available:
            faces = self._detect_yolo(arr)
            if faces:
                self._stats["yolo"] += 1
                backend = "yolo"

        # 2. OpenCV DNN ResNet-SSD（YOLO 失败时的兜底，或优先模式下的首选）
        if not faces and self._opencv_dnn_available:
            faces = self._detect_opencv_dnn(arr)
            if faces:
                self._stats["opencv_dnn"] += 1
                backend = "opencv_dnn"

        # 3. YuNet（进一步兜底，误检率极低）
        if not faces and self._yunet_available:
            faces, face_infos = self._detect_yunet(arr)
            if faces:
                self._stats["yunet"] += 1
                backend = "yunet"

        # 4. YOLO 失败后用 OpenCV DNN 低阈值兜底（召回更多小脸）
        if not faces and self._opencv_dnn_available and not self.prefer_opencv_dnn:
            original_conf = self.opencv_dnn_conf
            self.opencv_dnn_conf = self.fallback_dnn_conf   # 临时用兜底阈值
            faces = self._detect_opencv_dnn(arr)
            self.opencv_dnn_conf = original_conf             # 恢复原阈值
            if faces:
                self._stats["opencv_dnn"] += 1
                backend = "opencv_dnn"

        # 5. Haar Cascade（最后的回退）
        if not faces:
            faces = self._detect_haar(arr)
            if faces:
                self._stats["haar"] += 1
                backend = "haar"

        # ── 眼睛验证：过滤误检 ──
        if faces and self.eye_validation:
            original_count = len(faces)
            faces, face_infos = self._eye_validate(arr, faces, face_infos)
            self._stats["eye_rejected"] += original_count - len(faces)
        return faces, face_infos, backend

    def _load_opencv_dnn(self):
        """延迟加载 OpenCV DNN 模型"""
        if self._opencv_dnn_net is None and self._opencv_dnn_available:
            try:
                self._opencv_dnn_net = cv2.dnn.readNetFromCaffe(
                    self.OPENCV_DNN_PROTO, self.OPENCV_DNN_WEIGHTS
                )
            except Exception as e:
                self._opencv_dnn_available = False
                print(f"[FaceDetector] OpenCV DNN load failed: {e}")

    def _detect_opencv_dnn(self, arr: np.ndarray) -> List[Tuple]:
        """OpenCV DNN ResNet-SSD 人脸检测（首选，大脸特写最佳）

        Returns:
            bboxes — (x,y,w,h) 列表
        """
        self._load_opencv_dnn()
        if self._opencv_dnn_net is None:
            return []

        h, w = arr.shape[:2]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # 创建 blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0),
            swapRB=False, crop=False
        )

        self._opencv_dnn_net.setInput(blob)
        detections = self._opencv_dnn_net.forward()

        # 第一步：收集所有置信度 >= opencv_dnn_conf 的检测框，
        # 用 cv2.dnn.NMSBoxes 做 IOR-then-min-size 级联过滤
        # 这样即使大脸旁有小脸，只要各自的置信度够高就不会互相抑制
        raw_boxes, raw_confs = [], []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.opencv_dnn_conf:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bw, bh = x2 - x1, y2 - y1
            raw_boxes.append((x1, y1, bw, bh))
            raw_confs.append(conf)

        if not raw_boxes:
            return []

        # cv2.dnn.NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold)
        # — 先按 scoreThreshold 过滤（已在上面做）
        # — 再按 nmsThreshold 做 IOR 重叠过滤
        # — 最后返回 minSize 过滤后的框（由 cv2 内部实现）
        indices = cv2.dnn.NMSBoxes(
            raw_boxes, raw_confs,
            score_threshold=self.opencv_dnn_conf,
            nms_threshold=0.3,          # 宽松 IOR threshold
        )

        if isinstance(indices, np.ndarray) and len(indices):
            result = [raw_boxes[i] for i in indices.flatten()]
        else:
            result = []

        # 额外按 min_face 做尺寸过滤（允许比 yunet_min_face 更小的检测框）
        min_size = self.opencv_dnn_min_face  # 独立参数，默认 30px
        result = [(x, y, bw, bh) for x, y, bw, bh in result
                   if bw >= min_size and bh >= min_size]

        return result

    def _detect_yunet(self, arr: np.ndarray) -> Tuple[List[Tuple], List]:
        """YuNet 深度学习检测

        Returns:
            (bboxes, face_infos) — bboxes 为 (x,y,w,h) 列表；face_infos 为 YuNet 原始数组列表
        """
        h, w = arr.shape[:2]
        # 复用 YuNet 实例：每次 create 都要重新加载 onnx（~300ms），
        # deskew 每张照片都要量 roll，不缓存代价太高
        det = getattr(self, '_yunet_det', None)
        key = (self.yunet_threshold, self.yunet_nms_threshold)
        if det is None or getattr(self, '_yunet_key', None) != key:
            det = cv2.FaceDetectorYN.create(
                self.YUNET_MODEL, '',
                (w, h),
                score_threshold=self.yunet_threshold,
                nms_threshold=self.yunet_nms_threshold,
                top_k=5000
            )
            self._yunet_det = det
            self._yunet_key = key
        else:
            det.setInputSize((w, h))
        detector = det
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(bgr)
        if faces is None:
            return [], []
        bboxes = []
        face_infos = []
        for f in faces:
            if int(f[2]) >= self.yunet_min_face and int(f[3]) >= self.yunet_min_face:
                bboxes.append((int(f[0]), int(f[1]), int(f[2]), int(f[3])))
                face_infos.append(f)
        return bboxes, face_infos

    def _detect_yolo(self, arr: np.ndarray) -> List[Tuple]:
        """YOLOv8n-Face 检测（延迟加载模型，含 pre-resize 优化）"""
        self._load_yolo()
        if self._yolo_model is None:
            return []
        h_img, w_img = arr.shape[:2]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Pre-resize to 640px max edge for faster inference
        scale = min(640 / w_img, 640 / h_img)
        if scale < 1.0:
            small = cv2.resize(bgr, (int(w_img * scale), int(h_img * scale)))
        else:
            small = bgr
            scale = 1.0

        results = self._yolo_model(small, verbose=False, conf=self.yolo_conf, imgsz=640)
        faces = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Scale back to original image coordinates
                x, y = int(x1 / scale), int(y1 / scale)
                w, h = int((x2 - x1) / scale), int((y2 - y1) / scale)
                if w >= self.yunet_min_face and h >= self.yunet_min_face:
                    faces.append((x, y, w, h))
        return self._nms(faces, iou_threshold=0.4)

    def _detect_haar(self, arr: np.ndarray) -> List[Tuple]:
        """Haar Cascade 回退检测"""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h_img, w_img = gray.shape[:2]
        min_face = max(40, int(min(h_img, w_img) * 0.05))

        raw_boxes = set()
        for cascade in self.cascades.values():
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_face, min_face))
            for (x, y, w, h) in faces:
                raw_boxes.add((x, y, w, h))

        if not raw_boxes:
            return []

        # 宽高比过滤
        filtered = [(x, y, w, h) for (x, y, w, h) in raw_boxes
                    if 0.5 < w / max(h, 1) < 1.6]
        if not filtered:
            return []

        # 眼睛验证
        verified = []
        for (x, y, w, h) in filtered:
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=2,
                minSize=(max(5, w//8), max(5, h//8)))
            has_eyes = len(eyes) > 0
            if self.strict_mode:
                if has_eyes:
                    verified.append((x, y, w, h))
            else:
                is_large = (w * h) >= (min_face * min_face * 4)
                if has_eyes or is_large:
                    verified.append((x, y, w, h))

        if not self.strict_mode and len(verified) < len(filtered) * 0.3:
            verified = filtered

        return self._nms(verified, iou_threshold=0.4)

    def _eye_validate(self, arr: np.ndarray, boxes: List[Tuple],
                      face_infos: List = None) -> Tuple[List[Tuple], List]:
        """眼睛验证：对检测框进行二次过滤

        Returns:
            (validated_boxes, validated_infos) — 过滤后的 bbox 列表和对应的 face_info 列表
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h_img, w_img = gray.shape[:2]
        min_face = max(40, int(min(h_img, w_img) * 0.05))
        validated = []
        validated_infos = []

        for idx, (x, y, w, h) in enumerate(boxes):
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=2,
                minSize=(max(5, w//8), max(5, h//8)))
            has_eyes = len(eyes) > 0

            if self.strict_mode:
                # 严格模式：必须有一双眼睛
                if has_eyes:
                    validated.append((x, y, w, h))
                    if face_infos and idx < len(face_infos):
                        validated_infos.append(face_infos[idx])
            else:
                # 普通模式：大脸无需眼睛，小脸需要眼睛验证
                is_large = (w * h) >= (min_face * min_face * 4)
                if has_eyes or is_large:
                    validated.append((x, y, w, h))
                    if face_infos and idx < len(face_infos):
                        validated_infos.append(face_infos[idx])

        # 如果过滤掉太多，回退到原始结果
        if len(validated) < len(boxes) * 0.3 and not self.strict_mode:
            validated = list(boxes)
            validated_infos = list(face_infos) if face_infos else []
        return validated, validated_infos

    def _eye_validate_keypoints(self, faces: List[Tuple],
                                 keypoints_list: List,
                                 arr: np.ndarray) -> Tuple[List[Tuple], List]:
        """使用 YOLO keypoints 做眼睛验证（替代 Haar Cascade，无需额外推理）

        YOLOv8n-Face 输出 5 个关键点：左眼、右眼、鼻尖、左嘴角、右嘴角
        通过检查双眼位置合理性来过滤误检。

        Args:
            faces: (x, y, w, h) 列表
            keypoints_list: 每个 face 对应的 keypoints 数组 (N, 2)
            arr: 原始 RGB 图像（仅用于获取尺寸信息）
        Returns:
            (validated_faces, validated_infos) — 过滤后的结果
        """
        h_img, w_img = arr.shape[:2]
        min_face = max(40, int(min(h_img, w_img) * 0.05))
        validated = []

        for idx, ((x, y, w, h), kpts) in enumerate(zip(faces, keypoints_list)):
            if kpts is None or len(kpts) < 2:
                # 没有 keypoint 数据，大脸直接通过，小脸拒绝
                is_large = (w * h) >= (min_face * min_face * 4)
                if self.strict_mode:
                    if not is_large:  # strict: 无 keypoints 一律拒绝
                        continue
                else:
                    if not is_large:
                        continue
                validated.append((x, y, w, h))
                continue

            # keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
            left_eye = kpts[0]
            right_eye = kpts[1]

            # 验证眼睛位置合理性
            face_cx = x + w / 2
            face_cy = y + h / 2

            # 1. 眼睛应在脸部上半部分
            eyes_above_center = (left_eye[1] < face_cy) and (right_eye[1] < face_cy)

            # 2. 双眼间距应合理（至少是脸宽的 10%，不超过脸宽的 95%）
            eye_dist = abs(left_eye[0] - right_eye[0])
            eye_dist_reasonable = (w * 0.1) < eye_dist < (w * 0.95)

            # 3. 眼睛应在脸部水平范围内
            eyes_in_face = (x < left_eye[0] < x + w) and (x < right_eye[0] < x + w)

            has_valid_eyes = eyes_above_center and eye_dist_reasonable and eyes_in_face

            if self.strict_mode:
                if has_valid_eyes:
                    validated.append((x, y, w, h))
            else:
                is_large = (w * h) >= (min_face * min_face * 4)
                if has_valid_eyes or is_large:
                    validated.append((x, y, w, h))

        # 如果过滤掉太多，回退
        if len(validated) < len(faces) * 0.3 and not self.strict_mode:
            validated = list(faces)

        return validated, []  # no face_infos for YOLO path

    def _nms(self, boxes: List, iou_threshold: float = 0.4) -> List:
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]
        order = np.argsort(areas)[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return [tuple(boxes[i].astype(int)) for i in keep]

    def _load_sface(self):
        """延迟加载 SFace 人脸识别模型"""
        if self._sface_recognizer is None and self._sface_available:
            try:
                self._sface_recognizer = cv2.FaceRecognizerSF.create(
                    self.SFACE_MODEL, '')
            except Exception as e:
                self._sface_available = False
                print(f"[FaceDetector] SFace load failed: {e}")

    def _arcface_encoding(self, bgr_img, face_info=None, bbox=None) -> Optional[List[float]]:
        """使用 ArcFace ONNX 模型提取 512 维人脸特征向量

        ArcFace 相比 SFace（128d）精度更高，尤其在极端光照/角度场景下。
        模型自动检测（models/arcface/ 目录下任一 .onnx 文件即可）。
        """
        if not self._arcface_available:
            return None

        if self._arcface_session is None:
            self._arcface_session = _get_onnx_session(self.ARCFACE_MODEL)
            if self._arcface_session is None:
                self._arcface_available = False
                return None

        try:
            # 准备输入图像（对齐裁剪为 112x112）
            if face_info is not None:
                # 使用 YuNet landmarks 做对齐裁剪（最佳质量）
                if self._sface_recognizer is None and self._sface_available:
                    if self._sface_recognizer is None:
                        self._sface_recognizer = cv2.FaceRecognizerSF.create(self.SFACE_MODEL, '')
                if self._sface_recognizer is not None:
                    aligned = self._sface_recognizer.alignCrop(bgr_img, face_info)
                else:
                    aligned = self._arcface_align(bgr_img, face_info, bbox)
            elif bbox is not None:
                # 从 bbox 裁剪并缩放
                x, y, w, h = bbox
                face_crop = bgr_img[y:y+h, x:x+w]
                if face_crop.size == 0:
                    return None
                aligned = cv2.resize(face_crop, (112, 112))
            else:
                return None

            # ArcFace 预处理：标准化到 [-1, 1]
            blob = aligned.astype(np.float32) / 255.0
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            blob = (blob - mean) / std
            blob = blob.transpose(2, 0, 1)  # HWC -> CHW
            blob = blob[np.newaxis, ...]    # Add batch dim

            # ONNX 推理（自动获取输入节点名）
            input_name = self._arcface_session.get_inputs()[0].name
            feature = self._arcface_session.run(None, {input_name: blob})[0]
            return feature.flatten().tolist()
        except Exception as e:
            print(f"[FaceDetector] ArcFace encoding failed: {e}")
            return None

    def _arcface_align(self, bgr_img, face_info, bbox) -> np.ndarray:
        """使用 5 个关键点做相似变换对齐人脸（ArcFace 标准对齐）

        face_info 格式（YuNet）：[x, y, w, h, l5_x0, l5_y0, l5_x1, l5_y1, ..., score]
        ArcFace 标准 5 关键点（112x112）：左眼、右眼、鼻尖、左嘴角、右嘴角
        """
        # 提取 5 个 landmarks
        kps = face_info[4:14].reshape(5, 2)  # 5 landmarks

        # 目标关键点（ArcFace 112x112 标准）
        dst = np.array([
            [30.2946, 51.6968],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ], dtype=np.float32)

        # 源关键点（如果人脸框太小，缩放）
        src = kps.astype(np.float32)

        # 相似变换（只做缩放+旋转+平移，不做仿射）
        tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if tform is None:
            # 回退：直接 resize
            x, y = int(face_info[0]), int(face_info[1])
            w, h = int(face_info[2]), int(face_info[3])
            return cv2.resize(bgr_img[y:y+h, x:x+w], (112, 112))

        aligned = cv2.warpAffine(bgr_img, tform, (112, 112), borderValue=0.0)
        return aligned

    def _sface_encoding(self, bgr_img, face_info=None, bbox=None) -> Optional[List[float]]:
        """使用 SFace 提取 128 维人脸特征向量

        Args:
            bgr_img: BGR 格式原图（numpy array）
            face_info: YuNet 返回的完整人脸数据（含 landmarks），用于 alignCrop
            bbox: 人脸边界框 (x, y, w, h)，face_info 不可用时的回退
        """
        self._load_sface()
        if self._sface_recognizer is None:
            # SFace 不可用，回退到简单编码
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            if face_info is not None:
                x, y, w, h = int(face_info[0]), int(face_info[1]), int(face_info[2]), int(face_info[3])
            elif bbox is not None:
                x, y, w, h = bbox
            else:
                return None
            face_img = gray[y:y+h, x:x+w]
            if face_img.size == 0:
                return None
            return self._simple_encoding(face_img)

        try:
            if face_info is not None:
                # 使用 YuNet 完整数据做 alignCrop（最佳质量）
                aligned = self._sface_recognizer.alignCrop(bgr_img, face_info)
            elif bbox is not None:
                # 回退：从 bbox 裁剪并缩放到 112x112
                x, y, w, h = bbox
                face_crop = bgr_img[y:y+h, x:x+w]
                if face_crop.size == 0:
                    return None
                aligned = cv2.resize(face_crop, (112, 112))
            else:
                return None

            feature = self._sface_recognizer.feature(aligned)
            return feature.flatten().tolist()
        except Exception as e:
            print(f"[FaceDetector] SFace encoding failed: {e}")
            return None

    def _simple_encoding(self, face_img) -> Optional[List[float]]:
        """灰度直方图编码（回退方案，32 维）"""
        if face_img is None or face_img.size == 0:
            return None
        face_resized = cv2.resize(face_img, (64, 64))
        hist = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        return [float(v) for v in hist]


# ─────────────────────────────────────────────
# FaceMatcher — 人脸库
# ─────────────────────────────────────────────

def _to_db_path(old_path: str) -> str:
    if old_path.endswith('.json'):
        return old_path[:-5] + '.db'
    if old_path.endswith('.db'):
        return old_path  # 已经是 .db 路径，直接返回
    return old_path + '.db'


class FaceMatcher:
    """人脸匹配与聚类（SQLite 持久化）"""

    def __init__(self, face_db_path: str, match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self._db_path = _to_db_path(face_db_path)
        self.match_threshold = match_threshold   # ← 可在实例化后修改
        # 仅当传入旧版 .json 路径时才尝试迁移；直接传 .db 时若走迁移逻辑，
        # 会把 .db 当 JSON 解析失败并改名成 .broken（数据事故），2026-10-18 修复
        if face_db_path.endswith('.json'):
            self._maybe_migrate_from_json(face_db_path)
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persons'").fetchone()
        conn.close()
        if not cur:
            _init_db(self._db_path)
        _ensure_indexes(self._db_path)   # 既有库补建 idx_encodings_person（幂等）

    def face_distance(self, enc1: List[float], enc2: List[float]) -> float:
        """计算两个人脸编码之间的余弦距离（1 - cosine_similarity）

        值越小表示越相似：0 = 完全相同，2 = 完全相反
        对于 SFace 128 维编码，阈值约 0.6（即 cosine similarity > 0.4）
        """
        a = np.array(enc1)
        b = np.array(enc2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return 1.0 - cos_sim

    def find_match(self, encoding: List[float]) -> Tuple[Optional[str], float]:
        """在已标注人物中查找与 encoding 最相似的人，返回 (name, dist)。

        2026-10-18：改走 core/face_index.py 的向量索引（numpy 矩阵运算 + 缓存），
        原实现每次全量 json.loads 全部编码后 Python 循环，3213 条时 ~3.5s/次。
        返回语义完全不变：假脸 → (None, inf)；无人脸库 → (None, inf)。
        """
        if encoding is None:
            return None, float('inf')
        return get_face_index(self._db_path).find_match(
            encoding, false_face_threshold=FALSE_FACE_THRESHOLD)

    def _is_false_face(self, encoding: List[float]) -> bool:
        """假脸黑名单判定。

        旧实现每调用一次都 SELECT 全部 false_faces 并逐条 json.loads +
        face_distance（215 条 × 每张脸）——批量导入时是主要瓶颈，且与
        find_match 用的阈值不一致。现在复用编码索引里已归一化的假脸矩阵
        （numpy 点积），语义不变但快 ~100 倍。
        """
        return get_face_index(self._db_path).is_false_face(
            encoding, FALSE_FACE_THRESHOLD)

    def add_false_face(self, encoding: List[float], photo_path: str,
                       face_location: tuple, bbox: tuple):
        with _db_conn(self._db_path) as conn:
            conn.execute(
                "INSERT INTO false_faces(encoding,photo_path,location,bbox,marked_at) VALUES(?,?,?,?,?)",
                (_json_str(encoding), photo_path,
                 _json_str(face_location), _json_str(bbox), datetime.now().isoformat())
            )
        note_false_encoding_added(self._db_path, encoding)

    def get_false_faces(self) -> List[Dict]:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT encoding,photo_path,location,bbox,marked_at FROM false_faces "
                "ORDER BY marked_at DESC").fetchall()
        return [{"encoding": json.loads(r[0]), "photo_path": r[1],
                 "location": _json_loads(r[2]), "bbox": _json_loads(r[3]), "marked_at": r[4]}
                for r in rows]

    def remove_false_face(self, photo_path: str, bbox: tuple):
        with _db_conn(self._db_path) as conn:
            conn.execute(
                "DELETE FROM false_faces WHERE photo_path=? AND bbox=?",
                (photo_path, _json_str(bbox))
            )
        invalidate_face_index(self._db_path)

    def add_person(self, name: str, encoding: List[float], photo_path: str,
                   face_location: tuple, bbox: tuple) -> str:
        with _db_conn(self._db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO persons(name) VALUES(?)", (name,))
            pid = conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()[0]
            existing = conn.execute(
                "SELECT id FROM encodings WHERE person_id=? AND photo_path=? AND bbox=?",
                (pid, photo_path, _json_str(bbox))
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO encodings(person_id,encoding,photo_path,location,bbox,added_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (pid, _json_str(encoding), photo_path,
                     _json_str(face_location), _json_str(bbox), datetime.now().isoformat())
                )
                # 写入穿透编码索引（INSERT 不重建整库，见 face_index 模块文档）
                note_encoding_added(self._db_path, name, encoding)
        return name

    def get_person_encodings(self, name: str) -> List[List[float]]:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT e.encoding FROM encodings e JOIN persons p ON p.id=e.person_id "
                "WHERE p.name=?", (name,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def update_person_name(self, old_name: str, new_name: str):
        with _db_conn(self._db_path) as conn:
            conn.execute("UPDATE persons SET name=? WHERE name=?", (new_name, old_name))
        invalidate_face_index(self._db_path)  # 原地 UPDATE 不改变指纹，必须显式失效

    def delete_person(self, name: str):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM persons WHERE name=?", (name,))
        invalidate_face_index(self._db_path)

    def get_duplicate_encodings(self, threshold: float = 0.4) -> List[Dict]:
        """找出彼此过于相似的编码（余弦距离 < threshold）。

        2026-11-11 前这里只保留 `len(enc) < 128` 的「32维直方图」编码，
        而库里 3664 条全是 512 维 ArcFace → 结果恒为空，这个功能等于没开。
        同时旧实现用未归一化向量的 L2 距离，和全应用的余弦距离语义不一致。
        现在改为分块 numpy 矩阵（3664×3664 不整体落地），按维度分组比较。
        """
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT e.id, e.person_id, p.name, e.encoding, e.photo_path, e.bbox "
                "FROM encodings e JOIN persons p ON p.id=e.person_id"
            ).fetchall()

        by_dim = {}
        for r in rows:
            enc = _json_loads(r[3])
            if not enc:
                continue
            meta = {"id": r[0], "person": r[2], "photo_path": r[4],
                    "bbox": _json_loads(r[5])}
            by_dim.setdefault(len(enc), []).append((meta, enc))

        duplicates = []
        MAX_RESULTS = 500          # 页面只显示前 20 条，不必枚举全部组合
        for items in by_dim.values():
            if len(items) < 2:
                continue
            metas = [m for m, _ in items]
            M = np.asarray([v for _, v in items], dtype=np.float64)
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            Q = M / norms                       # 预归一化 → 余弦距离 = 1 - 点积
            n = len(Q)
            for start in range(0, n, 256):
                block = 1.0 - (Q[start:start + 256] @ Q.T)
                ai, bi = np.where(block < threshold)
                for a, b in zip(ai, bi):
                    i, j = start + int(a), int(b)
                    if i >= j:                  # 只要上三角，排除自身
                        continue
                    duplicates.append({
                        "encoding_id": metas[i]["id"], "person": metas[i]["person"],
                        "photo_path": metas[i]["photo_path"], "bbox": metas[i]["bbox"],
                        "duplicate_of": {
                            "encoding_id": metas[j]["id"], "person": metas[j]["person"],
                            "photo_path": metas[j]["photo_path"], "bbox": metas[j]["bbox"]},
                        "distance": round(float(block[a, b]), 4),
                    })
                    if len(duplicates) >= MAX_RESULTS:
                        return duplicates
        return duplicates

    def remove_encoding(self, encoding_id: int):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM encodings WHERE id=?", (encoding_id,))
        invalidate_face_index(self._db_path)

    def remove_photo_encodings(self, photo_path: str):
        """Remove all face encodings for a specific photo."""
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM encodings WHERE photo_path=?", (photo_path,))
        invalidate_face_index(self._db_path)

    def reset_db(self):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM encodings")
            conn.execute("DELETE FROM persons")
            conn.execute("DELETE FROM photos")
            conn.execute("DELETE FROM false_faces")
            conn.commit()
        invalidate_face_index(self._db_path)

    def get_all_persons(self) -> Dict:
        """所有人及其编码照片列表。

        2026-10-18：改为单条 JOIN 查询。原实现每人一次 get_person_photos
        （各开一个连接 + 全表扫 encodings——person_id 无索引），49 人 ~2s；
        配合 idx_encodings_person 索引后 ~50ms。返回结构不变。
        """
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT p.name, e.photo_path, e.location, e.bbox, e.added_at "
                "FROM persons p LEFT JOIN encodings e ON e.person_id = p.id "
                "ORDER BY p.name, e.id"
            ).fetchall()
        result = {}
        for r in rows:
            entry = result.get(r[0])
            if entry is None:
                entry = {"name": r[0], "photos": []}
                result[r[0]] = entry
            if r[1] is not None:   # LEFT JOIN：无编码的人物 photos 为空
                entry["photos"].append({
                    "path": r[1],
                    "location": _json_loads(r[2]),
                    "bbox": _json_loads(r[3]),
                    "added_at": r[4],
                })
        return result

    def get_person_photos(self, name: str) -> List[Dict]:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT e.photo_path,e.location,e.bbox,e.added_at "
                "FROM encodings e JOIN persons p ON p.id=e.person_id WHERE p.name=?",
                (name,)
            ).fetchall()
        return [{"path": r[0], "location": _json_loads(r[1]),
                 "bbox": _json_loads(r[2]), "added_at": r[3]} for r in rows]

    def export_db(self) -> Dict:
        persons = {}
        for name in self.get_all_persons():
            with _db_conn(self._db_path) as conn:
                enc_rows = conn.execute(
                    "SELECT encoding FROM encodings e JOIN persons p ON p.id=e.person_id "
                    "WHERE p.name=?", (name,)
                ).fetchall()
            photos = self.get_person_photos(name)
            persons[name] = {"name": name,
                              "encodings": [json.loads(r[0]) for r in enc_rows],
                              "photos": photos}
        return {"persons": persons, "unknown_clusters": {}}

    def import_db(self, data: Dict):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM encodings")
            conn.execute("DELETE FROM persons")
        for name, pdata in data.get("persons", {}).items():
            for enc, photo_info in zip(
                    pdata.get("encodings", []),
                    pdata.get("photos", [])):
                self.add_person(name, enc, photo_info.get("path", ""),
                                photo_info.get("location"), photo_info.get("bbox"))
        invalidate_face_index(self._db_path)

    def re_encode_all(self, detector, force_rotated: bool = False) -> Dict:
        """使用新编码模型重新编码所有已标注人脸

        Args:
            detector: FaceDetector 实例（用于提取新编码）
            force_rotated: True 时，横持照片（EXIF Orientation≠1）的编码
                即使维度已达标也强制重编码——它们的旧编码是侧躺裁剪出来
                的，匹配质量差（2026-10-19 EXIF 摆正改造前的历史数据）
        Returns:
            {"re_encoded": 成功数, "failed": 失败数, "skipped": 跳过数}
        """
        # 目标维度：ArcFace 优先（512d），否则 SFace（128d）
        target_dim = 512 if detector._arcface_available else 128

        stats = {"re_encoded": 0, "failed": 0, "skipped": 0}
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT e.id, e.photo_path, e.bbox, e.encoding, p.name "
                "FROM encodings e JOIN persons p ON p.id=e.person_id"
            ).fetchall()

        for row in rows:
            enc_id, photo_path, bbox_str, old_enc_str, person_name = row
            old_enc = json.loads(old_enc_str) if old_enc_str else []

            # 检查是否需要重新编码（已达到目标维度则跳过；
            # force_rotated 时横持照片例外，见 docstring）
            if len(old_enc) == target_dim and not (
                    force_rotated and get_orientation(photo_path) != 1):
                stats["skipped"] += 1
                continue

            if not os.path.exists(photo_path):
                stats["failed"] += 1
                continue

            try:
                # 读取摆正后的图（横持照片的旧编码是侧躺裁剪的，必须重编码）
                img, orientation, (orig_w, orig_h) = open_upright(photo_path)
                arr = np.array(img)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                bbox = json.loads(bbox_str) if bbox_str else None
                if bbox is None:
                    stats["failed"] += 1
                    continue

                x, y, w, h = bbox
                if orientation != 1:
                    x, y, w, h = map_box_to_upright((x, y, w, h), orientation,
                                                    orig_w, orig_h)
                # 优先用 ArcFace 编码，失败则回退 SFace
                new_enc = detector._arcface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
                if new_enc is None:
                    new_enc = detector._sface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
                if new_enc is None:
                    stats["failed"] += 1
                    continue

                # 更新数据库
                with _db_conn(self._db_path) as conn:
                    conn.execute(
                        "UPDATE encodings SET encoding=? WHERE id=?",
                        (_json_str(new_enc), enc_id)
                    )
                stats["re_encoded"] += 1
            except Exception as e:
                print(f"[re_encode] Failed for {photo_path}: {e}")
                stats["failed"] += 1

        # 原地 UPDATE 不改变库指纹，必须显式失效编码索引
        invalidate_face_index(self._db_path)
        return stats

    def re_encode_false_faces(self, detector, force_rotated: bool = False) -> Dict:
        """使用新编码模型重新编码所有假人脸

        Args:
            detector: FaceDetector 实例
            force_rotated: True 时横持照片的假脸编码也强制重编码
                （旧编码是侧躺裁剪的，黑名单匹配不上摆正后的复检脸）
        Returns:
            {"re_encoded": 成功数, "failed": 失败数, "skipped": 跳过数}
        """
        target_dim = 512 if detector._arcface_available else 128

        stats = {"re_encoded": 0, "failed": 0, "skipped": 0}
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, photo_path, bbox, encoding FROM false_faces"
            ).fetchall()

        for row in rows:
            ff_id, photo_path, bbox_str, old_enc_str = row
            old_enc = json.loads(old_enc_str) if old_enc_str else []

            if len(old_enc) == target_dim and not (
                    force_rotated and get_orientation(photo_path) != 1):
                stats["skipped"] += 1
                continue

            if not os.path.exists(photo_path):
                stats["failed"] += 1
                continue

            try:
                img, orientation, (orig_w, orig_h) = open_upright(photo_path)
                arr = np.array(img)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                bbox = json.loads(bbox_str) if bbox_str else None
                if bbox is None:
                    stats["failed"] += 1
                    continue

                x, y, w, h = bbox
                if orientation != 1:
                    x, y, w, h = map_box_to_upright((x, y, w, h), orientation,
                                                    orig_w, orig_h)
                new_enc = detector._arcface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
                if new_enc is None:
                    new_enc = detector._sface_encoding(bgr, face_info=None, bbox=(x, y, w, h))
                if new_enc is None:
                    stats["failed"] += 1
                    continue

                with _db_conn(self._db_path) as conn:
                    conn.execute(
                        "UPDATE false_faces SET encoding=? WHERE id=?",
                        (_json_str(new_enc), ff_id)
                    )
                stats["re_encoded"] += 1
            except Exception as e:
                print(f"[re_encode_false] Failed for {photo_path}: {e}")
                stats["failed"] += 1

        # 原地 UPDATE 不改变库指纹，必须显式失效编码索引
        invalidate_face_index(self._db_path)
        return stats


# ─────────────────────────────────────────────
# PhotoManager — 照片索引
# ─────────────────────────────────────────────

class PhotoManager:
    def __init__(self, index_path: str, output_dir: str):
        self.output_dir = output_dir
        self._db_path = _to_db_path(index_path)
        # 仅当传入旧版 .json 路径时才尝试迁移（原因同 FaceMatcher，2026-10-18 修复）
        if index_path.endswith('.json'):
            self._maybe_migrate_from_json(index_path)
        self._ensure_tables()
        self._ensure_md5_column()
        self._ensure_rotate_columns()
        os.makedirs(output_dir, exist_ok=True)

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='photos'").fetchone()
        conn.close()
        if not cur:
            _init_photo_index_db(self._db_path)

    def _ensure_md5_column(self):
        with _db_conn(self._db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()]
            if "md5" not in cols:
                conn.execute("ALTER TABLE photos ADD COLUMN md5 TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_md5 ON photos(md5)")

    def _ensure_rotate_columns(self):
        """给既有库补"手动旋转"标记列 + 操作日志表（幂等，启动时 ~1ms）。

        manual_rotate 语义与 roll 一致：**逆时针为正**，range (-180, 180]，
        正常照片为 0。它记录的是"用户点了几次、往哪边点"，与算法判断的
        roll 分开存 —— 排查时要能一眼看出"这张是谁转歪的"。
        """
        with _db_conn(self._db_path) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()]
            if "manual_rotate" not in cols:
                conn.execute("ALTER TABLE photos ADD COLUMN "
                             "manual_rotate INTEGER NOT NULL DEFAULT 0")
            if "manual_rotate_at" not in cols:
                conn.execute("ALTER TABLE photos ADD COLUMN manual_rotate_at TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_manual_rotate "
                         "ON photos(manual_rotate)")
            conn.execute("""CREATE TABLE IF NOT EXISTS manual_rotate_log(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    from_deg REAL NOT NULL,
                    to_deg REAL NOT NULL,
                    faces_before INTEGER,
                    faces_after INTEGER,
                    names_before TEXT,
                    names_after TEXT,
                    status TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rotate_log_path "
                         "ON manual_rotate_log(path)")

    # ── 手动旋转 ──

    def get_manual_rotate(self, photo_path: str) -> int:
        with _db_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT manual_rotate FROM photos WHERE path=?", (photo_path,)).fetchone()
        try:
            return int(row[0]) if row and row[0] else 0
        except (TypeError, ValueError):
            return 0

    def get_photo_faces_persons(self, photo_path: str):
        """单张照片的 (faces, persons)，不存在时返回 ([], [])。"""
        with _db_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT faces, persons FROM photos WHERE path=?", (photo_path,)).fetchone()
        if not row:
            return [], []
        return _json_loads(row[0]) or [], _json_loads(row[1]) or []

    def photo_exists(self, photo_path: str) -> bool:
        """该路径是否已在索引里（**精确匹配**：路径写法不同就是另一行）。"""
        with _db_conn(self._db_path) as conn:
            return conn.execute(
                "SELECT 1 FROM photos WHERE path=?", (photo_path,)).fetchone() is not None

    def set_manual_rotate(self, photo_path: str, deg: float) -> int:
        """写入手动旋转角（归一化到 (-180,180]），返回写入值。"""
        deg = _normalize_rotate(deg)
        with _db_conn(self._db_path) as conn:
            conn.execute(
                "UPDATE photos SET manual_rotate=?, manual_rotate_at=? WHERE path=?",
                (int(deg), datetime.now().isoformat(), photo_path))
        return int(deg)

    def log_manual_rotate(self, photo_path: str, from_deg: float, to_deg: float,
                          faces_before: int, faces_after: int,
                          names_before=None, names_after=None,
                          status: str = "ok", note: str = ""):
        """记一条旋转日志（排查用：谁转的、转前转后几张脸、名字有没有丢）。"""
        with _db_conn(self._db_path) as conn:
            conn.execute(
                "INSERT INTO manual_rotate_log(path,from_deg,to_deg,faces_before,"
                "faces_after,names_before,names_after,status,note,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (photo_path, float(from_deg), float(to_deg), int(faces_before),
                 int(faces_after), _json_str(names_before or []),
                 _json_str(names_after or []), status, note,
                 datetime.now().isoformat()))

    def get_manual_rotation_log(self, limit: int = 300) -> List[Dict]:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,from_deg,to_deg,faces_before,faces_after,"
                "names_before,names_after,status,note,created_at "
                "FROM manual_rotate_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"path": r[0], "from_deg": r[1], "to_deg": r[2],
                 "faces_before": r[3], "faces_after": r[4],
                 "names_before": _json_loads(r[5]) or [],
                 "names_after": _json_loads(r[6]) or [],
                 "status": r[7], "note": r[8], "created_at": r[9]} for r in rows]

    def get_manual_rotated_photos(self) -> List[Dict]:
        """所有手动旋转过的照片（含旋转角），排查入口。"""
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,manual_rotate,manual_rotate_at,faces,persons,indexed_at "
                "FROM photos WHERE manual_rotate IS NOT NULL AND manual_rotate != 0 "
                "ORDER BY manual_rotate_at DESC").fetchall()
        return [{"path": r[0], "manual_rotate": r[1], "manual_rotate_at": r[2],
                 "faces": _json_loads(r[3]) or [], "persons": _json_loads(r[4]) or [],
                 "indexed_at": r[5]} for r in rows]

    def get_rotate_review_photos(self, limit: int = 30) -> List[Dict]:
        """手动旋转过、但一张脸都没检出的照片（默认不在任何页面出现，专用复查）。"""
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,manual_rotate,manual_rotate_at,indexed_at FROM photos "
                "WHERE manual_rotate != 0 AND (faces IS NULL OR faces='[]') "
                "ORDER BY manual_rotate_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"path": r[0], "manual_rotate": r[1], "manual_rotate_at": r[2],
                 "indexed_at": r[3]} for r in rows]

    def check_md5_exists(self, md5: str) -> Optional[str]:
        with _db_conn(self._db_path) as conn:
            row = conn.execute(
                "SELECT path FROM photos WHERE md5=?", (md5,)).fetchone()
            return row[0] if row else None

    def add_photo(self, photo_path: str, faces: List[Dict],
                  persons: List[str] = None, md5: str = None):
        with _db_conn(self._db_path) as conn:
            existing = conn.execute(
                "SELECT persons FROM photos WHERE path=?", (photo_path,)).fetchone()
            if existing:
                # persons 显式传入时才覆盖（None = 保留原标注，[] = 清空）。
                # 旧实现 UPDATE 分支完全忽略 persons，导致重新扫描 / 导入索引
                # 时新脸换了、旧人名却还挂着。
                conn.execute(
                    "UPDATE photos SET faces=?, persons=COALESCE(?,persons), "
                    "md5=COALESCE(?,md5), indexed_at=? WHERE path=?",
                    (_json_str(faces),
                     _json_str(persons) if persons is not None else None,
                     md5, datetime.now().isoformat(), photo_path)
                )
            else:
                conn.execute(
                    "INSERT INTO photos(path,md5,faces,persons,indexed_at) VALUES(?,?,?,?,?)",
                    (photo_path, md5, _json_str(faces), _json_str(persons or []),
                     datetime.now().isoformat())
                )

    def get_photos_by_person(self, person_name: str) -> List[Dict]:
        # 2026-11-11：原来的 persons LIKE '%"name"%' 有两个坑——
        #   ① 名字里的 % / _ 会被当通配符；② SQLite LIKE 对 ASCII 不区分大小写，
        #      "tom" 会把 ["Tom"] 也算进来，导出时把别人的照片拷进自己目录。
        # 改用 json_each 做精确值匹配。
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT p.path,p.faces,p.md5,p.manual_rotate FROM photos p "
                "WHERE json_valid(p.persons) AND EXISTS ("
                "  SELECT 1 FROM json_each(p.persons) j WHERE j.value=?)",
                (person_name,)
            ).fetchall()
        return [{"path": r[0], "faces": _json_loads(r[1]) or [], "md5": r[2],
                 "manual_rotate": r[3] or 0} for r in rows]

    def get_missing_photos(self) -> List[Dict]:
        """索引里源文件已不存在的记录。

        2026-11-11：先只取 path 做存在性判断，再**仅对缺失的那几行**解析
        faces/persons JSON。旧实现无条件把 8867 行 ×2 个 JSON 全 loads 一遍
        （~440ms，且绝大多数是白做），缺失文件为 0 时尤其浪费。
        """
        with _db_conn(self._db_path) as conn:
            gone = [r[0] for r in conn.execute("SELECT path FROM photos").fetchall()
                    if not os.path.exists(r[0])]
            if not gone:
                return []
            marks = ",".join("?" for _ in gone)
            rows = conn.execute(
                f"SELECT path,md5,faces,persons,indexed_at FROM photos "
                f"WHERE path IN ({marks})", gone).fetchall()
        return [{"path": r[0], "md5": r[1],
                 "faces": _json_loads(r[2]) or [],
                 "persons": _json_loads(r[3]) or [],
                 "indexed_at": r[4]} for r in rows]

    def get_duplicate_photos(self) -> Dict[str, List[Dict]]:
        duplicates = {"by_md5": {}}
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,md5,faces,persons,indexed_at FROM photos "
                "WHERE md5 IS NOT NULL AND md5 != ''").fetchall()
        md5_groups = {}
        for r in rows:
            md5 = r[1]
            if md5 not in md5_groups:
                md5_groups[md5] = []
            md5_groups[md5].append({"path": r[0], "md5": md5,
                                     "faces": _json_loads(r[2]) or [],
                                     "persons": _json_loads(r[3]) or [],
                                     "indexed_at": r[4]})
        for md5, photos in md5_groups.items():
            if len(photos) > 1:
                duplicates["by_md5"][md5] = photos
        return duplicates

    def remove_photo(self, photo_path: str):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM photos WHERE path=?", (photo_path,))

    def get_unlabeled_photos(self) -> List[Dict]:
        """只返回有 faces 但没有 persons 的照片（标注页专用，避免全表扫描）。"""
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,faces,manual_rotate FROM photos "
                "WHERE faces IS NOT NULL AND faces!='[]' AND (persons IS NULL OR persons='[]' OR persons='')"
            ).fetchall()
        return [{"path": r[0], "faces": _json_loads(r[1]) or [],
                 "manual_rotate": r[2] or 0} for r in rows]

    @property
    def db_path(self) -> str:
        """SQLite 文件路径（供进程级缓存签名用，不必碰私有属性）。"""
        return self._db_path

    def get_all_photo_paths(self) -> List[str]:
        """只取 path 列，不解析 faces/persons JSON。

        清理失效记录时用：get_all_photos() 会把 8867 行的 faces/persons
        全 json.loads 一遍（8867×2 次，实测 ~2s），而这里只需要路径去做
        os.path.exists。2026-11-11 标注页刷新变慢的根因就在这里。
        """
        with _db_conn(self._db_path) as conn:
            return [r[0] for r in conn.execute("SELECT path FROM photos").fetchall()]

    def get_all_photos(self) -> Dict:
        with _db_conn(self._db_path) as conn:
            rows = conn.execute(
                "SELECT path,faces,persons,indexed_at,md5,manual_rotate "
                "FROM photos").fetchall()
        return {r[0]: {"faces": _json_loads(r[1]),
                        "persons": _json_loads(r[2]) or [],
                        "indexed_at": r[3],
                        "md5": r[4],
                        "manual_rotate": r[5] or 0}
                for r in rows}

    def get_photo_count(self) -> int:
        """照片总数（O(1) 计数）。

        sidebar / 首页等只关心数量时使用，避免 get_all_photos() 全表加载
        faces JSON（8867 行实测 ~4.4s）。
        """
        with _db_conn(self._db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]

    def get_stranger_photo_count(self) -> int:
        """标记为「陌生人」的照片数（persons JSON 含该名字）。

        陌生人没有 encodings（见 core/stranger.py），不会出现在
        get_all_persons 里；分类浏览页用它把「陌生人」并入人物列表，
        使这批照片可浏览、可批量重置。
        """
        with _db_conn(self._db_path) as conn:
            # 精确值匹配；GROUP BY 防止一张照片里出现多个「陌生人」时重复计数
            row = conn.execute(
                "SELECT COUNT(*) FROM (SELECT p.id FROM photos p "
                "WHERE json_valid(p.persons) AND EXISTS ("
                "  SELECT 1 FROM json_each(p.persons) j WHERE j.value=?) "
                "GROUP BY p.id)",
                (STRANGER_NAME,)
            ).fetchone()
        return row[0] if row else 0

    def update_photo_persons(self, photo_path: str, persons: List[str]):
        with _db_conn(self._db_path) as conn:
            conn.execute("UPDATE photos SET persons=? WHERE path=?",
                         (_json_str(persons), photo_path))

    def remove_face_from_photo(self, photo_path: str, bbox: tuple):
        """Remove a face from a photo's face list by its bbox."""
        with _db_conn(self._db_path) as conn:
            rows = conn.execute("SELECT faces FROM photos WHERE path=?", (photo_path,)).fetchall()
            if not rows:
                return
            faces = _json_loads(rows[0][0]) or []
            # Remove the face whose bbox matches (compare as strings for safety)
            bbox_str = _json_str(bbox)
            new_faces = [f for f in faces if _json_str(f.get("bbox", ())) != bbox_str]
            conn.execute("UPDATE photos SET faces=? WHERE path=?",
                         (_json_str(new_faces), photo_path))

    def clear_photo_faces(self, photo_path: str):
        """Clear all faces and persons from a photo (used when all detected faces are false)."""
        with _db_conn(self._db_path) as conn:
            conn.execute("UPDATE photos SET faces=?, persons=? WHERE path=?",
                         ("[]", "[]", photo_path))

    def reset_db(self):
        with _db_conn(self._db_path) as conn:
            conn.execute("DELETE FROM photos")
            conn.commit()

    def re_encode_all_photos(self, detector, force_rotated: bool = False) -> Dict:
        """使用新编码模型重新编码照片索引中的所有人脸

        Args:
            detector: FaceDetector 实例
            force_rotated: True 时横持照片的人脸编码也强制重编码
                （旧编码是侧躺裁剪的，标注页预填匹配不上）
        Returns:
            {"re_encoded": 成功数, "failed": 失败数, "skipped": 跳过数}
        """
        target_dim = 512 if detector._arcface_available else 128

        stats = {"re_encoded": 0, "failed": 0, "skipped": 0}
        with _db_conn(self._db_path) as conn:
            rows = conn.execute("SELECT path, faces FROM photos").fetchall()

        for photo_path, faces_str in rows:
            if not faces_str:
                continue
            faces = _json_loads(faces_str) or []
            if not faces:
                continue

            modified = False
            if not os.path.exists(photo_path):
                stats["failed"] += len(faces)
                continue

            try:
                img, orientation, (orig_w, orig_h) = open_upright(photo_path)
                arr = np.array(img)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                stats["failed"] += len(faces)
                continue

            for face in faces:
                enc = face.get("encoding", [])
                if len(enc) == target_dim and not (
                        force_rotated and orientation != 1):
                    stats["skipped"] += 1
                    continue

                bbox = face.get("bbox")
                if not bbox:
                    stats["failed"] += 1
                    continue

                x, y, w, h = tuple(bbox)
                if orientation != 1:
                    x, y, w, h = map_box_to_upright((x, y, w, h), orientation,
                                                    orig_w, orig_h)
                try:
                    new_enc = detector.encode_region(bgr, (x, y, w, h), face.get("roll", 0.0))
                    if new_enc is None:
                        stats["failed"] += 1
                        continue
                    face["encoding"] = new_enc
                    modified = True
                    stats["re_encoded"] += 1
                except Exception:
                    stats["failed"] += 1

            if modified:
                with _db_conn(self._db_path) as conn:
                    conn.execute("UPDATE photos SET faces=? WHERE path=?",
                                 (_json_str(faces), photo_path))

        return stats

    def export_index(self) -> Dict:
        return {"photos": self.get_all_photos()}

    def import_index(self, data: Dict):
        for photo_path, photo_data in data.get("photos", {}).items():
            self.add_photo(photo_path, photo_data.get("faces", []),
                          photo_data.get("persons", []),
                          md5=photo_data.get("md5"))
            # 旧实现只还原 faces/persons，丢掉 md5 和手动旋转角：
            # 导入后 MD5 去重失效、人工摆正过的方向被抹掉。
            rot = int(photo_data.get("manual_rotate") or 0)
            if rot:
                self.set_manual_rotate(photo_path, rot)


# ─────────────────────────────────────────────
# 迁移
# ─────────────────────────────────────────────

def _migrate_json_to_db(json_path: str, db_path: str, table: str, fields: Dict):
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        import traceback; traceback.print_exc()
        try:
            os.rename(json_path, json_path + '.broken')
        except OSError:
            os.remove(json_path)
        print(f"  [迁移] {json_path} 损坏，已处理")
        return
    if os.path.exists(db_path):
        os.rename(json_path, json_path + '.migrated')
        print(f"  [迁移] {json_path} 已迁移，跳过")
        return
    os.rename(json_path, json_path + '.migrated')
    _init_db(db_path)
    with _db_conn(db_path) as conn:
        for key, row_data in data.get(table, {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO photos(path,faces,persons,indexed_at) VALUES(?,?,?,?)",
                (key, _json_str(row_data.get("faces")),
                 _json_str(row_data.get("persons", [])),
                 row_data.get("indexed_at", datetime.now().isoformat()))
            )
    print(f"  [迁移] {json_path} → {db_path}")


def _migrate_face_db_json(json_path: str, db_path: str):
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        try:
            os.rename(json_path, json_path + '.broken')
        except OSError:
            os.remove(json_path)
        print(f"  [迁移] {json_path} 损坏，已处理")
        return
    if os.path.exists(db_path):
        os.rename(json_path, json_path + '.migrated')
        print(f"  [迁移] {json_path} 已迁移，跳过")
        return
    os.rename(json_path, json_path + '.migrated')
    _init_db(db_path)
    with _db_conn(db_path) as conn:
        for name, pdata in data.get("persons", {}).items():
            conn.execute("INSERT OR IGNORE INTO persons(name) VALUES(?)", (name,))
            pid = conn.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()[0]
            for enc, photo_info in zip(pdata.get("encodings", []), pdata.get("photos", [])):
                conn.execute(
                    "INSERT INTO encodings(person_id,encoding,photo_path,location,bbox,added_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (pid, _json_str(enc), photo_info.get("path", ""),
                     _json_str(photo_info.get("location")),
                     _json_str(photo_info.get("bbox")),
                     photo_info.get("added_at", datetime.now().isoformat()))
                )
    print(f"  [迁移] {json_path} → {db_path}")


FaceMatcher._maybe_migrate_from_json = lambda self, old_path: _migrate_face_db_json(old_path, self._db_path)
PhotoManager._maybe_migrate_from_json = lambda self, old_path: _migrate_json_to_db(
    old_path, self._db_path, "photos", {"faces": None, "persons": [], "indexed_at": None})


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def extract_face_image(photo_path: str, location: tuple,
                       max_width: Optional[int] = None,
                       roll: float = 0.0) -> Image.Image:
    """从照片中裁剪一张脸（实现见 core/image_utils.py：draft 降采样 + 缓存）。

    max_width=None 时全分辨率裁剪；传入显示宽度（如 160）则降采样后裁剪，
    用于 st.image 小图显示，解码快约 8 倍。
    """
    return _extract_face_image(photo_path, location, max_width, roll)


def draw_faces_on_image(image_path: str, faces: List[Dict], labels: Dict = None,
                        max_width: Optional[int] = None) -> Optional[np.ndarray]:
    """在照片上画人脸框，返回 BGR numpy 数组（实现见 core/image_utils.py）。"""
    return _draw_faces_on_image(image_path, faces, labels, max_width)


def export_photos_by_person(photo_manager: PhotoManager, matcher: FaceMatcher, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    persons = matcher.get_all_persons()
    exported_count = {}
    for person_name in persons.keys():
        person_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        photos = photo_manager.get_photos_by_person(person_name)
        for photo_info in photos:
            photo_path = photo_info["path"]
            if os.path.exists(photo_path):
                filename = os.path.basename(photo_path)
                target_path = os.path.join(person_dir, filename)
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{datetime.now().strftime('%H%M%S')}{ext}"
                    target_path = os.path.join(person_dir, filename)
                shutil.copy2(photo_path, target_path)
                exported_count[person_name] = exported_count.get(person_name, 0) + 1
    return exported_count
