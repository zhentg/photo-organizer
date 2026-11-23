# core/face_index.py — 人脸编码向量索引（find_match 加速，2026-10-18）
"""
背景：FaceMatcher.find_match 原实现每次调用都全量读取 encodings（3213条）
+ false_faces（77条），逐条 json.loads 后用 Python 循环算余弦距离，
实测 ~3.5s/次，且随人脸库增长线性变慢。

本模块把全部编码加载为 numpy 矩阵并预归一化，查询时一次矩阵乘法完成
（3213×512 ≈ 1ms）。返回语义与原 find_match 完全一致：
(name, dist)；假脸 → (None, inf)；无人脸库 → (None, inf)。

缓存失效策略（按写入类型分流）：
- INSERT（add_person / add_false_face）：写入穿透——直接把新编码追加进
  矩阵（pending 缓冲，下次查询前合并），避免"全库匹配"这类
  find_match 与写入交替的流程反复重建索引（重建一次 ~3.4s）。
- 原地 UPDATE / DELETE（改名、删除、重编码、重置、导入）：整库失效，
  下次查询时从 db 重建。这些都是低频管理操作，重建成本可接受。
  新增写入路径时务必按上面两类之一调用对应函数，否则索引会过期。

纯 core 模块：无 Streamlit 依赖，可独立测试（设计原则不变）。
"""
import json
import sqlite3
import threading

import numpy as np

_LOCK = threading.RLock()
_INDEX_CACHE = {}   # db_path -> FaceEncodingIndex


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    """按行归一化；零向量行保持零（距离=1.0，与 face_distance 语义对齐）。"""
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _load_matrix(rows) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray([json.loads(r[0]) for r in rows], dtype=np.float64)


class FaceEncodingIndex:
    """全部人物编码 + 假脸编码的 numpy 矩阵视图（只读 + 写入穿透）。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._load()

    def _load(self):
        conn = _connect(self.db_path)
        try:
            person_rows = conn.execute(
                "SELECT e.encoding, p.name FROM encodings e "
                "JOIN persons p ON p.id = e.person_id"
            ).fetchall()
            false_rows = conn.execute("SELECT encoding FROM false_faces").fetchall()
        finally:
            conn.close()
        self.names = [r[1] for r in person_rows]
        self._mat = _row_normalize(_load_matrix(person_rows))
        self._fmat = _row_normalize(_load_matrix(false_rows))
        self._pending = []          # [(name, encoding)] 尚未合并的写入穿透
        self._pending_false = []    # [encoding] 尚未合并的假脸写入穿透

    @property
    def size(self) -> int:
        return len(self.names) + len(self._pending)

    def _flush_pending(self):
        """把 pending 写入合并进矩阵（调用方需持有 _LOCK）。"""
        if self._pending:
            rows = np.asarray([enc for _, enc in self._pending], dtype=np.float64)
            rows = _row_normalize(rows)
            self._mat = rows if self._mat.size == 0 else np.vstack([self._mat, rows])
            self.names.extend(name for name, _ in self._pending)
            self._pending.clear()
        if self._pending_false:
            rows = np.asarray(self._pending_false, dtype=np.float64)
            rows = _row_normalize(rows)
            self._fmat = rows if self._fmat.size == 0 else np.vstack([self._fmat, rows])
            self._pending_false.clear()

    # ── 写入穿透（INSERT 类路径调用） ──

    def add_encoding(self, name: str, encoding):
        """追加一条人物编码（不立即合并，下次查询前 flush）。"""
        with _LOCK:
            self._pending.append((name, list(encoding)))

    def add_false_encoding(self, encoding):
        """追加一条假脸编码。"""
        with _LOCK:
            self._pending_false.append(list(encoding))

    # ── 查询 ──

    def find_match(self, encoding, false_face_threshold: float):
        """向量化匹配。false_face_threshold：假脸判定阈值（< 即判假脸）。"""
        with _LOCK:
            self._flush_pending()
            if encoding is None:
                return None, float("inf")
            q = _row_normalize(np.asarray(encoding, dtype=np.float64).reshape(1, -1))[0]

            # 假脸过滤（语义同原 _is_false_face）
            if self._fmat.shape[0] and self._fmat.shape[1] == q.shape[0]:
                fd = 1.0 - (self._fmat @ q)
                if float(fd.min()) < false_face_threshold:
                    return None, float("inf")

            if not self.names:
                return None, float("inf")
            if self._mat.shape[1] != q.shape[0]:
                # 维度不一致（如残留的低维编码）：跳过人物匹配而非崩溃
                print(f"[face_index] dimension mismatch: index={self._mat.shape[1]}, "
                      f"query={q.shape[0]}")
                return None, float("inf")

            d = 1.0 - (self._mat @ q)
            i = int(np.argmin(d))
            return self.names[i], float(d[i])

    def is_false_face(self, encoding, threshold: float) -> bool:
        """向量化假脸判定（替代 FaceMatcher._is_false_face 的 215 次 json.loads 循环）。

        返回 True 表示 encoding 与某个已知假脸足够近（余弦距离 < threshold）。
        """
        with _LOCK:
            self._flush_pending()
            if encoding is None or not self._fmat.shape[0]:
                return False
            q = _row_normalize(np.asarray(encoding, dtype=np.float64).reshape(1, -1))[0]
            if self._fmat.shape[1] != q.shape[0]:
                return False
            return float((1.0 - (self._fmat @ q)).min()) < threshold


def get_face_index(db_path: str) -> FaceEncodingIndex:
    """获取编码索引（不存在时从 db 构建）。线程安全。"""
    with _LOCK:
        idx = _INDEX_CACHE.get(db_path)
        if idx is None:
            idx = FaceEncodingIndex(db_path)
            _INDEX_CACHE[db_path] = idx
        return idx


def note_encoding_added(db_path: str, name: str, encoding):
    """add_person 写入成功后调用：索引已存在则写入穿透；不存在则无需操作
    （首次 find_match 构建时自然包含新数据）。"""
    with _LOCK:
        idx = _INDEX_CACHE.get(db_path)
        if idx is not None:
            idx.add_encoding(name, encoding)


def note_false_encoding_added(db_path: str, encoding):
    """add_false_face 写入成功后调用。语义同 note_encoding_added。"""
    with _LOCK:
        idx = _INDEX_CACHE.get(db_path)
        if idx is not None:
            idx.add_false_encoding(encoding)


def invalidate_face_index(db_path: str = None):
    """原地 UPDATE / DELETE 类写入后调用：丢弃缓存，下次查询时重建。
    db_path=None 时清空全部。"""
    with _LOCK:
        if db_path is None:
            _INDEX_CACHE.clear()
        else:
            _INDEX_CACHE.pop(db_path, None)
