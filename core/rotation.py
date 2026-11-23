# core/rotation.py — 手动旋转（90° 步进）+ 旋转后重检测
"""用户在界面上"把照片转 90°"的完整动作，只在这里实现一次。

为什么单独成模块：这个动作要同时动四张表/四个状态，任何一处漏掉都会
产生"界面看着是正的、库里还是歪的"这种静默不一致（975a80.jpg 就是这样
被漏掉的）。所以把顺序固定下来：

    1. 读旧 manual_rotate，算出新角度（归一化到 (-180,180]，逆时针为正）
    2. 写 manual_rotate（先落库：万一检测崩了，下次重跑也还是这个方向）
    3. 用 detect_faces(manual_rotate=新角度) 重检测——**关掉四向探索**，
       方向完全由用户决定
    4. 已标注照片：旧人名按"框中心最近"搬到新框（数量不一致/太远就不搬，
       记为 names_lost 由复查面板处理）
    5. 写 faces / persons
    6. 已标注照片同步人脸库 encodings（旧框+旧向量必须一起换，否则识别
       永远匹配不上 —— 975a80 的旧编码距离 0.68，新编码 0.04）
    7. 记一条 manual_rotate_log，status ∈ ok / no_face / names_lost

坐标约定：手动旋转只是一次整体旋转，detect_faces 内部会把 bbox 映射回
原始像素坐标，所以库里所有框仍然可以用旧代码裁剪/画框。
"""
import math
from typing import Dict, List, Tuple

from core.face_detector import _normalize_rotate


def _center(bbox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _migrate_names(old_faces: List[Dict], new_faces: List[Dict],
                   old_persons: List[str]) -> Tuple[List[str], List[str]]:
    """把旧人名搬到新框上，返回 (新 persons 列表, 丢失的人名列表)。

    匹配用"框中心最近 + 一对一贪心"。旋转不改人脸在画面里的相对位置，
    所以正确旋转后旧框中心与新框中心应当基本重合；容差取 0.75 倍对角线
    （大于这个距离说明框已经对不上，宁可不搬也不猜）。
    """
    if not old_faces or not new_faces:
        return [""] * len(new_faces), [n for n in old_persons if n]

    pairs = []
    for i, of in enumerate(old_faces):
        oc = _center(of.get("bbox", (0, 0, 1, 1)))
        diag = math.hypot(of.get("bbox", (0, 0, 1, 1))[2],
                          of.get("bbox", (0, 0, 1, 1))[3])
        for j, nf in enumerate(new_faces):
            nc = _center(nf.get("bbox", (0, 0, 1, 1)))
            pairs.append((_dist(oc, nc), i, j, max(diag, 1.0)))
    pairs.sort(key=lambda t: t[0])

    used_i, used_j, mapping = set(), set(), {}
    for d, i, j, diag in pairs:
        if i in used_i or j in used_j:
            continue
        if d > 0.75 * diag:
            continue
        used_i.add(i)
        used_j.add(j)
        mapping[i] = j

    persons_new = [""] * len(new_faces)
    lost = []
    for i, j in mapping.items():
        name = old_persons[i] if i < len(old_persons) else ""
        if name:
            persons_new[j] = name
    # 旧名字里没有被搬过去的 → 丢了
    for i, name in enumerate(old_persons):
        if name and i not in mapping:
            lost.append(name)
    return persons_new, lost


def _sync_encodings(matcher, photo_path: str, faces: List[Dict],
                    persons: List[str]) -> int:
    """按新框/新向量重写该照片在人脸库里的编码行，返回写入条数。

    只删这张照片的行（别的照片不动）。名字为空的人脸没有编码可写。
    """
    if matcher is None:
        return 0
    matcher.remove_photo_encodings(photo_path)
    n = 0
    for face, name in zip(faces, persons):
        if not name:
            continue
        enc = face.get("encoding")
        if not enc:
            continue
        matcher.add_person(name=name, encoding=enc, photo_path=photo_path,
                           face_location=face.get("location"),
                           bbox=face.get("bbox", (0, 0, 0, 0)))
        n += 1
    return n


def rotate_photo(pm, detector, photo_path: str, delta: float,
                 matcher=None, max_faces: int = 20,
                 note: str = "") -> Dict:
    """把照片再手动转 delta 度（逆时针为正；界面右转传 -90）并重检测。

    Args:
        pm: PhotoManager（照片索引库）
        detector: FaceDetector（**必须**带 photo_index_path 或显式传角度，
            否则自动逻辑会忽略手动旋转）
        photo_path: 照片绝对路径
        delta: 本次增量角度（通常 ±90）
        matcher: FaceMatcher；已标注照片用它同步 encodings，None 则跳过同步
        max_faces: 检出超过这个数量视为误检簇，不落库（只记日志）

    Returns:
        dict：old_manual / new_manual / faces_before / faces_after /
        names_before / names_after / lost / status / roll
    """
    old_manual = pm.get_manual_rotate(photo_path)
    new_manual = int(_normalize_rotate(old_manual + delta))

    # 路径必须在索引里精确存在：add_photo 对未知路径是 INSERT，
    # 写法差一个斜杠就会**新增一行重复照片**（实测踩到过），先拦住。
    if not pm.photo_exists(photo_path):
        pm.log_manual_rotate(photo_path, old_manual, new_manual, 0, 0, [], [],
                             "not_indexed", "路径不在照片索引中，未写入")
        return {"old_manual": old_manual, "new_manual": old_manual,
                "faces_before": 0, "faces_after": 0, "names_before": [],
                "names_after": [], "lost": [], "synced": 0,
                "status": "not_indexed"}

    # 先落库：检测失败也不会让"用户点过的方向"丢失
    pm.set_manual_rotate(photo_path, new_manual)

    old_faces, old_persons = pm.get_photo_faces_persons(photo_path)
    old_persons = (list(old_persons) + [""] * len(old_faces))[: len(old_faces)]

    try:
        # allow_cardinal 交给 detect_faces 默认规则：手动角为 0（重置）时
        # 恢复自动四向探索，非 0 时方向由用户拍板、不再自动探索
        new_faces = detector.detect_faces(photo_path, manual_rotate=new_manual)
    except Exception as e:                                    # noqa: BLE001
        pm.log_manual_rotate(photo_path, old_manual, new_manual, len(old_faces), 0,
                             old_persons, [], "error", f"检测异常: {e}")
        return {"old_manual": old_manual, "new_manual": new_manual,
                "faces_before": len(old_faces), "faces_after": 0,
                "names_before": [n for n in old_persons if n], "names_after": [],
                "lost": [n for n in old_persons if n], "status": "error",
                "error": str(e)}

    if len(new_faces) > max_faces:
        pm.log_manual_rotate(photo_path, old_manual, new_manual, len(old_faces),
                             len(new_faces), old_persons, [],
                             "too_many_faces",
                             f"检出 {len(new_faces)} 张脸 > {max_faces}，未落库")
        return {"old_manual": old_manual, "new_manual": new_manual,
                "faces_before": len(old_faces), "faces_after": len(new_faces),
                "names_before": [n for n in old_persons if n], "names_after": [],
                "lost": [n for n in old_persons if n], "status": "too_many_faces"}

    labeled = any(n for n in old_persons)
    if labeled:
        persons_new, lost = _migrate_names(old_faces, new_faces, old_persons)
    else:
        persons_new, lost = [""] * len(new_faces), []

    pm.clear_photo_faces(photo_path)
    pm.add_photo(photo_path, new_faces)
    pm.update_photo_persons(photo_path, [p for p in persons_new if p])

    synced = 0
    if labeled and matcher is not None:
        synced = _sync_encodings(matcher, photo_path, new_faces, persons_new)

    if not new_faces:
        status = "no_face"
    elif lost:
        status = "names_lost"
    elif labeled and matcher is not None and synced < sum(1 for p in persons_new if p):
        # 编码没写全 = 识别库与索引不一致（975a80 那类问题的入口）
        status = "encoding_partial"
    else:
        status = "ok"

    pm.log_manual_rotate(photo_path, old_manual, new_manual, len(old_faces),
                         len(new_faces), old_persons, persons_new, status, note)

    return {"old_manual": old_manual, "new_manual": new_manual,
            "faces_before": len(old_faces), "faces_after": len(new_faces),
            "names_before": [n for n in old_persons if n],
            "names_after": [n for n in persons_new if n],
            "lost": lost, "synced": synced, "status": status,
            "roll": [f.get("roll") for f in new_faces]}


def reset_manual_rotate(pm, detector, photo_path: str, matcher=None) -> Dict:
    """把手动旋转清零并重检测（"⟲ 重置"按钮）。"""
    cur = pm.get_manual_rotate(photo_path)
    return rotate_photo(pm, detector, photo_path, -cur, matcher=matcher,
                        note="reset")


def label_of(deg: float) -> str:
    """把角度写成界面上的说法（排查/展示共用，避免各处自己拼字符串）。"""
    d = _normalize_rotate(deg)
    if not d:
        return "未旋转"
    n = int(abs(d))
    side = "左转" if d > 0 else "右转"
    if n == 90:
        return f"{side} 90°"
    if n == 180:
        return "转 180°"
    if n == 270:
        return f"{side} 90°×3"
    return f"{side} {n}°"
