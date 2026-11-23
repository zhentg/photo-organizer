"""图片解码工具：draft 降采样 + LRU 缓存。

背景：源照片多为 6000x4000 相机原图（Nikon 的不少是 MPO 格式），而标注页
只需要 80px 的脸缩略图和 300px 的总览图。旧实现每取一张脸就全图解码一次
（~350-500ms/次），N 人照片要解码 N+1 次，是交互期最大瓶颈。

本模块两个手段：
1. PIL ``Image.draft()`` —— 让 libjpeg 解码时直接按 1/2^k 缩放（跳过部分
   DCT 系数的反变换/上采样）。实测 6000x4000 MPO：全解码 ~350ms，
   解到 1500x1000 ~135ms（约 2.6 倍）——注意熵解码仍要读全部系数，
   时间收益是 2-4 倍而非像素数的 16 倍。
2. ``lru_cache`` 缓存"已解码的目标尺寸图"，同一次页面渲染里重复取同一张
   照片的脸（或 rerun 后回到同一页）直接命中，零解码。

坐标说明：``location=(top,right,bottom,left)`` 和 ``bbox=(x,y,w,h)``
都是原图坐标系。降采样后坐标按 scale = target_w/orig_w 换算，
输出语义（裁剪出的脸 / 画框的总览图）不变。

**两种 max_width 语义（别混！）：**
- ``extract_face_image(max_width=W)``：W 是**脸的目标宽度**。模块按脸在
  原图中的占比反推照片解码尺寸（脸占 1/8 宽就只解 1/4 的数据），并向上
  对齐到 draft 档位（原图的 1/1、1/2、1/4、1/8），让同一张照片里大小相近
  的脸共享同一缓存条目。比显示尺寸还小的脸才全解码（此时本身就在放大，
  多解也没有更多细节）。
- ``draw_faces_on_image(max_width=W)``：W 是**整张总览图的解码宽度**
  （总览要显示整张照片，按宽度解码、高度同比计算即可）。

**EXIF 摆正（2026-10-19 新增，横持拍摄照片的整体方案）：**
相机横持拍的照片（EXIF Orientation=8）在文件里是"躺着"的。本模块提供：
- ``open_upright(path)``：按 EXIF 摆正后返回 RGB 图——检测和编码都在摆正
  图上跑（YOLO/ArcFace 对正脸的效果远好于侧躺 90° 的脸）；
- ``map_box_to_original()``：摆正图坐标 → 原始像素坐标（存库用，保证
  ``extract_face_image`` 从原文件裁剪时落点正确）；
- ``map_box_to_upright()``：原始像素坐标 → 摆正图坐标（重编码旧库数据用）；
- ``extract_face_image()`` 输出前把裁剪结果按 orientation 旋转——**显示
  出来的脸永远是正的**。
坐标约定：库里存的 bbox/location 一律是**原始像素坐标系**（照片文件躺着的
坐标系），与旋转前完全兼容。
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# draft 只能按 1/1、1/2、1/4、1/8 降采样（PIL JpegImagePlugin 的档位表）
_DRAFT_STEPS = (8, 4, 2, 1)

# 缓存条目数：原来 16 条只够标注页 3 张照片；分类浏览页一次渲染 20 张、
# 假脸管理页一页 24 张，16 条会必然抖动（每条 rerun 全部重新解码）。
# 2026-11-11 提到 32，够覆盖任一单页；（超过 600 万像素的仍不进 LRU）。
_CACHE_SIZE = 32

# 超过这个像素数的解码结果不进 LRU（单条 6000x4000 全解码 = 72MB，
# 缓存它不划算）。6000x4000 的 1/2 档（3000x2000 = 600 万）刚好在界内。
_HUGE_PIXELS = 6_000_000


def _photo_size(path: str) -> Tuple[int, int]:
    """只读图片尺寸（不解码像素，~1ms）。"""
    with Image.open(path) as im:
        return im.size


# ─────────────────────────────────────────────
# EXIF 摆正（横持拍摄照片）
# ─────────────────────────────────────────────

# EXIF Orientation -> PIL transpose 方法（1 = 正常，不需处理）
_TRANSPOSE_METHODS = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_270,
    7: Image.TRANSVERSE,
    8: Image.ROTATE_90,
}


def get_orientation(path: str) -> int:
    """读 EXIF Orientation（1-8；无 EXIF/读失败按 1）。~1ms，不解码像素。"""
    try:
        with Image.open(path) as im:
            o = int(im.getexif().get(274, 1) or 1)
    except Exception:
        return 1
    return o if (o == 1 or o in _TRANSPOSE_METHODS) else 1


def open_upright(path: str) -> Tuple[Image.Image, int, Tuple[int, int]]:
    """打开图片并按 EXIF 摆正。

    返回 (摆正后的 RGB 图, orientation, 原始尺寸)。
    orientation=1 时图就是原图（零拷贝语义，仍需 .convert('RGB')）。
    """
    img = Image.open(path)
    try:
        o = int(img.getexif().get(274, 1) or 1)
    except Exception:
        o = 1
    if o != 1 and o not in _TRANSPOSE_METHODS:
        o = 1
    orig_size = img.size
    if o != 1:
        img = img.transpose(_TRANSPOSE_METHODS[o])
    return img.convert("RGB"), o, orig_size


def _forward_point(x: int, y: int, w: int, h: int, o: int) -> Tuple[int, int]:
    """原图 (w×h) 坐标系里的点 → 摆正图坐标系（与 PIL transpose 语义一致）。"""
    if o == 2:
        return (w - 1 - x, y)
    if o == 3:
        return (w - 1 - x, h - 1 - y)
    if o == 4:
        return (x, h - 1 - y)
    if o == 5:
        return (y, x)
    if o == 6:
        return (h - 1 - y, x)
    if o == 7:
        return (h - 1 - y, w - 1 - x)
    if o == 8:
        return (y, w - 1 - x)
    return (x, y)


def map_box_to_upright(bbox: tuple, orientation: int,
                       orig_w: int, orig_h: int) -> tuple:
    """原始像素坐标 bbox → 摆正图坐标（重编码旧库数据时用）。

    旋转/翻转都是轴对齐的，映射后仍是轴对齐框，取角点包围盒即精确值。
    """
    x, y, w, h = bbox
    if orientation == 1:
        return tuple(bbox)
    pts = [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    mapped = [_forward_point(px, py, orig_w, orig_h, orientation) for px, py in pts]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    nx, ny = min(xs), min(ys)
    return (nx, ny, max(xs) - nx + 1, max(ys) - ny + 1)


def map_box_to_original(bbox: tuple, orientation: int,
                        up_w: int, up_h: int) -> tuple:
    """摆正图坐标 bbox → 原始像素坐标（检测结果存库时用）。

    up_w/up_h 是摆正图的尺寸；原图尺寸按 orientation 是否交换宽高推导。
    """
    x, y, w, h = bbox
    if orientation == 1:
        return tuple(bbox)
    if orientation in (5, 6, 7, 8):
        orig_w, orig_h = up_h, up_w
    else:
        orig_w, orig_h = up_w, up_h
    # 反向点映射（_forward_point 的逆）
    pts = [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    inv = []
    for px, py in pts:
        if orientation == 2:
            inv.append((orig_w - 1 - px, py))
        elif orientation == 3:
            inv.append((orig_w - 1 - px, orig_h - 1 - py))
        elif orientation == 4:
            inv.append((px, orig_h - 1 - py))
        elif orientation == 5:
            inv.append((py, px))
        elif orientation == 6:
            inv.append((py, orig_h - 1 - px))
        elif orientation == 7:
            inv.append((orig_w - 1 - py, orig_h - 1 - px))
        else:  # 8
            inv.append((orig_w - 1 - py, px))
    xs = [p[0] for p in inv]
    ys = [p[1] for p in inv]
    nx, ny = min(xs), min(ys)
    return (nx, ny, max(xs) - nx + 1, max(ys) - ny + 1)


def rotate_expand(arr: np.ndarray, deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """绕图像中心旋转 deg 度，**扩展画布**保证内容不丢失。

    返回 (旋转后的图, M)。M 是 2x3 仿射矩阵，把源图坐标映射到旋转图坐标
    （与 cv2.warpAffine 的约定一致），逆变换用 cv2.invertAffineTransform(M)。

    deg 为负表示逆时针（OpenCV 正角度=逆时针；要"摆正右倾的脸"就传 -roll）。
    """
    h, w = arr.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(w * sin + h * cos))
    M[0, 2] += nw / 2.0 - c[0]
    M[1, 2] += nh / 2.0 - c[1]
    out = cv2.warpAffine(arr, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out, M


def map_box_through_affine(bbox: tuple, M: np.ndarray) -> tuple:
    """把 bbox 的 4 个角点过仿射变换，返回变换后坐标系里的轴对齐包围盒。

    旋转后的脸在目标坐标系里是"斜的"，包围盒会比原框略大——这是刻意的：
    存库坐标必须轴对齐（与旧数据、裁剪、画框全部兼容）。
    """
    x, y, w, h = bbox
    pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    dst = np.hstack([pts, ones]) @ M.T
    xs, ys = dst[:, 0], dst[:, 1]
    nx, ny = float(xs.min()), float(ys.min())
    return (int(round(nx)), int(round(ny)),
            int(round(xs.max() - nx)), int(round(ys.max() - ny)))


def _decode_to_size(path: str, target_w: int, target_h: int) -> Tuple[Image.Image, float]:
    """打开图片并解码到精确的 (target_w, target_h)，返回 (图, scale)。不走缓存。

    - draft 请求 (target_w, target_h)，libjpeg 选"不小于它的最小 1/2^k 档位"。
      **必须同时传宽和高**：档位对两个方向同时生效，只传宽会让高度条件
      把档位顶大（6000x4000 请求宽 1500 高 1501 → 只能解 3000x2000）。
    - 档位粒度粗，解码结果通常略大于请求，最后 resize 到精确尺寸
    - load() 之后 fp 即为 None（图片自包含），无需 copy
    """
    with Image.open(path) as img:
        orig_w, orig_h = img.size
        if orig_w > target_w or orig_h > target_h:
            # draft 必须在 load() 之前调用才对解码器生效；
            # mode 传图片自身模式（硬编码 "RGB" 会让灰度 JPEG 解析错位）
            img.draft(img.mode, (target_w, target_h))
        img.load()
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BILINEAR)
        return img, target_w / orig_w


@lru_cache(maxsize=_CACHE_SIZE)
def _load_scaled(path: str, target_w: int, target_h: int) -> Tuple[Image.Image, float]:
    """_decode_to_size 的缓存版。**返回的图是共享对象，调用方不得修改**"""
    return _decode_to_size(path, target_w, target_h)


def _get_scaled(path: str, target_w: int, target_h: int) -> Tuple[Image.Image, float]:
    """按目标尺寸取图：小结果走 LRU 缓存，超大结果（如全解码 6000x4000）直通。"""
    if target_w * target_h > _HUGE_PIXELS:
        return _decode_to_size(path, target_w, target_h)
    return _load_scaled(path, target_w, target_h)


def _snap_to_draft_step(orig_w: int, orig_h: int,
                        need_w: int, need_h: int) -> Tuple[int, int]:
    """把需要的解码尺寸向上对齐到 draft 档位（原图的 1/1、1/2、1/4、1/8）。

    对齐后 draft 输出即目标尺寸、无需 resize，且同一照片里大小相近的脸
    共享同一缓存条目。档位尺寸取 floor，保证 PIL 的 s*k<=orig 条件成立
    （最多差 1px，由 _load_scaled 的 resize 兜底）。
    """
    for k in _DRAFT_STEPS:
        sw, sh = orig_w // k, orig_h // k
        if sw >= need_w and sh >= need_h:
            return sw, sh
    return orig_w, orig_h


def _scale_bbox(bbox: tuple, scale: float) -> tuple:
    if scale == 1.0:
        return tuple(bbox)
    return tuple(int(round(v * scale)) for v in bbox)


def _rotate_crop(crop: Image.Image, roll: float) -> Image.Image:
    """把裁出来的脸按 roll 转正（roll 正=脸在画面里顺时针歪）。"""
    if not roll or abs(roll) < 1.0:
        return crop
    return crop.rotate(roll, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def extract_face_image(photo_path: str, location: tuple,
                       max_width: Optional[int] = None,
                       roll: float = 0.0) -> Image.Image:
    """从照片中裁剪一张脸。

    max_width 为 None 时保持旧行为（全分辨率裁剪）。
    传入 max_width 时，**max_width 指裁剪出的脸的目标宽度**（不是照片宽度！）：
    按"脸需要多大"反推照片解码尺寸——脸占原图 1/8 宽时，照片解到 1/4 档
    就能得到 max_width 像素的清晰脸图；只有比显示尺寸还小的脸才全解码。

    横持照片（EXIF Orientation≠1）的裁剪结果会按 orientation 旋转后返回，
    **显示出来的脸永远是正的**；调用方无需关心照片朝向。
    姿态歪的（detect_faces 打了 "roll" 标记的脸）传 roll，输出前同样转正。
    """
    orientation = get_orientation(photo_path)
    method = _TRANSPOSE_METHODS.get(orientation)

    if not max_width:
        img = Image.open(photo_path)
        top, right, bottom, left = location
        crop = img.crop((left, top, right, bottom))
        crop = crop.transpose(method) if method else crop
        return _rotate_crop(crop, roll)

    orig_w, orig_h = _photo_size(photo_path)
    top, right, bottom, left = location
    crop_w = max(1, right - left)
    # 脸目标 max_width px -> 照片解码尺寸（宽高同比缩放）
    need_w = round(orig_w * max_width / crop_w)
    need_h = round(orig_h * max_width / crop_w)
    decode_w, decode_h = _snap_to_draft_step(orig_w, orig_h, need_w, need_h)
    img, scale = _get_scaled(photo_path, decode_w, decode_h)
    t, r, b, l = _scale_bbox(location, scale)
    crop = img.crop((l, t, r, b))
    crop = crop.transpose(method) if method else crop
    return _rotate_crop(crop, roll)


def upright_photo_view(photo_path: str, max_width: Optional[int] = None,
                       roll: float = 0.0):
    """打开照片并摆正（EXIF + roll），返回 (PIL RGB 图, 坐标映射函数)。

    mapper(bbox) 把**原始像素坐标**的 (x, y, w, h) 映射到返回图的坐标。
    与 extract_face_image 的摆正方式完全一致（EXIF 用 PIL transpose，
    roll 用 rotate_expand），所以缩略图和整图预览里的脸都是同一个朝向。
    """
    orig_w, orig_h = _photo_size(photo_path)
    if max_width and orig_w > max_width:
        target_w = int(max_width)
        target_h = max(1, round(orig_h * target_w / orig_w))
    else:
        target_w, target_h = orig_w, orig_h
    img, sc = _get_scaled(photo_path, target_w, target_h)

    o = get_orientation(photo_path)
    if o != 1:
        img = img.transpose(_TRANSPOSE_METHODS[o])
    img = img.convert("RGB")

    def mapper(bbox):
        x, y, w, h = _scale_bbox(bbox, sc)
        if o != 1:
            x, y, w, h = map_box_to_upright((x, y, w, h), o, target_w, target_h)
        return (x, y, w, h)

    if not roll or abs(roll) < 1.0:
        return img, mapper

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rot, M = rotate_expand(bgr, roll)

    def mapper_rot(bbox):
        return map_box_through_affine(mapper(bbox), M)

    return Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)), mapper_rot


def draw_faces_on_image(image_path: str, faces: List[Dict],
                        labels: Dict = None,
                        max_width: Optional[int] = None) -> Optional[np.ndarray]:
    """在照片上画出人脸框，返回 BGR numpy 数组（供 cv2/st.image 使用）。

    预览图会先摆正（EXIF 横持 + 脸的 roll），框跟着一起映射——所以
    **预览里的人脸永远是正的**，不会出现"框对的、画面躺着"的割裂感。
    """
    roll = 0.0
    for f in faces or []:
        if f.get("roll"):
            roll = float(f["roll"])
            break
    img, mapper = upright_photo_view(image_path, max_width, roll)
    orig_w = _photo_size(image_path)[0]
    scale = min(1.0, img.width / max(1, orig_w))

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    thickness = max(1, round(2 * max(scale, 0.35)))
    font_scale = max(0.3, 0.5 * max(scale, 0.35))
    pad = max(2, round(10 * max(scale, 0.35)))
    for i, face in enumerate(faces or []):
        x, y, w, h = mapper(face.get("bbox", (0, 0, 0, 0)))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        label = f"Face {i+1}"
        if labels and str(i) in labels:
            label = labels[str(i)]
        cv2.putText(out, label, (x, max(pad, y - pad)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    return out


def clear_image_cache() -> None:
    """清空解码缓存（照片文件被删除/替换后调用）。"""
    _load_scaled.cache_clear()
