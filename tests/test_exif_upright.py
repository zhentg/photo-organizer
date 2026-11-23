"""EXIF 摆正改造回归测试（2026-10-19）。

背景：相机横持拍的照片（EXIF Orientation=8）在文件里是"躺着"的。
改造前 detect_faces 直接在躺着图上检测/编码：
- YOLO 对侧躺 90° 的脸置信度暴跌（4656 实测最高 0.187 < 0.5 阈值），
  触发 DNN 兜底产出 11 个误检；
- ArcFace 从侧躺裁剪出的编码谁都匹配不上（dist=inf）；
- 标注页脸缩略图也是躺着的。

改造后：检测/编码在摆正图上跑，bbox 映射回原始像素空间存库，
extract_face_image 输出前旋转裁剪结果——显示永远是正脸。

本测试锁住：
1. 坐标映射往返（8 种 orientation 随机框，逐像素一致）
2. PIL 实测校验映射公式（每种 orientation 用真旋转验证）
3. detect_faces 在横持照片上：框界内、显示输出==摆正裁剪、编码==摆正编码
4. extract_face_image 对正照片零变化（回归保护）
5. detect_faces 在正照片上：bbox/编码与手工一致（回归保护）

运行：python tests/test_exif_upright.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from PIL import Image

from core.image_utils import (map_box_to_upright, map_box_to_original,
                              open_upright, extract_face_image, get_orientation,
                              _TRANSPOSE_METHODS)
from core.face_detector import FaceDetector

ROTATED_PHOTO = r"D:\photoes\2024春节古装照片\5C3A4654.JPG"   # Orientation=8
UPRIGHT_PHOTO = r"D:\photoes\Nikon相机照片\比特恢复的文件\00 找到的文件   (9173 个文件)\图片\jpg\573140.jpg"  # Orientation=1


def test_roundtrip():
    """摆正→原图→摆正 往返必须逐像素一致（8 种 orientation）"""
    import random
    rng = random.Random(42)
    for o in range(1, 9):
        for _ in range(100):
            W, H = rng.choice([(6000, 4000), (4000, 6000), (800, 600)])
            x = rng.randint(0, W - 40)
            y = rng.randint(0, H - 40)
            w = rng.randint(20, min(600, W - x))
            h = rng.randint(20, min(600, H - y))
            up = map_box_to_upright((x, y, w, h), o, W, H)
            back = map_box_to_original(up, o, H, W) if o in (5, 6, 7, 8) \
                else map_box_to_original(up, o, W, H)
            assert back == (x, y, w, h), f"o={o}: {(x,y,w,h)} -> {up} -> {back}"
    print("✓ 坐标映射往返一致（800 组随机框 x 8 种 orientation）")


def test_pil_ground_truth():
    """映射公式 vs PIL 真旋转：合成图上放一个已知方块，逐 orientation 比对"""
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[100:200, 300:500] = 255          # 原图方块 (300,100,200,100)
    img = Image.fromarray(arr)
    for o, method in _TRANSPOSE_METHODS.items():
        rarr = np.array(img.transpose(method))
        ys, xs = np.where(rarr.sum(axis=2) > 0)
        actual = (int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        predicted = map_box_to_upright((300, 100, 200, 100), o, 600, 400)
        assert actual == predicted, f"o={o}: PIL {actual} != 公式 {predicted}"
    print("✓ 映射公式与 PIL 真旋转一致（8 种 orientation）")


def test_detect_on_rotated():
    """横持照片：detect_faces 的框/编码/显示三者全部来自摆正图"""
    assert get_orientation(ROTATED_PHOTO) == 8
    det = FaceDetector()
    faces = det.detect_faces(ROTATED_PHOTO)
    assert faces, "横持照片应能检测到人脸"
    img_up, o, (orig_w, orig_h) = open_upright(ROTATED_PHOTO)
    assert o == 8 and img_up.size == (orig_h, orig_w)
    bgr = cv2.cvtColor(np.array(img_up), cv2.COLOR_RGB2BGR)

    for f in faces:
        ob = tuple(f["bbox"])
        ub = map_box_to_upright(ob, o, orig_w, orig_h)
        # 摆正框必须在摆正图内
        assert 0 <= ub[0] and ub[0] + ub[2] <= img_up.size[0]
        assert 0 <= ub[1] and ub[1] + ub[3] <= img_up.size[1]
        # 显示输出（原图裁剪+旋转）== 摆正图手工裁剪
        manual = img_up.crop((ub[0], ub[1], ub[0] + ub[2], ub[1] + ub[3]))
        shown = extract_face_image(ROTATED_PHOTO, f["location"], max_width=400)
        manual_r = manual.resize((400, round(400 * manual.size[1] / manual.size[0])),
                                 Image.LANCZOS)
        shown_r = shown.resize(manual_r.size, Image.LANCZOS)
        diff = np.abs(np.array(manual_r.convert("RGB"), int)
                      - np.array(shown_r.convert("RGB"), int)).mean()
        assert diff < 2, f"显示输出与摆正裁剪不一致（像素差 {diff:.1f}）"
        # 编码来自摆正裁剪
        manual_enc = det._arcface_encoding(bgr, face_info=None, bbox=ub)
        assert manual_enc is not None
        enc_diff = float(np.linalg.norm(np.array(f["encoding"]) - np.array(manual_enc)))
        assert enc_diff < 1e-3, f"编码不是来自摆正裁剪（距离 {enc_diff:.3f}）"
    print(f"✓ 横持照片 detect_faces 正确（{len(faces)} 张脸：框界内/显示正脸/编码一致）")


def test_upright_unchanged():
    """正照片零变化：bbox 与摆正前一致、显示不经旋转、编码与手工一致"""
    assert get_orientation(UPRIGHT_PHOTO) == 1
    det = FaceDetector()
    faces = det.detect_faces(UPRIGHT_PHOTO)
    assert faces, "正照片应能检测到人脸"
    img = Image.open(UPRIGHT_PHOTO).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for f in faces:
        # orientation=1：detect_faces 的框就是原图框（与旧行为一致）
        manual_enc = det._arcface_encoding(bgr, face_info=None, bbox=tuple(f["bbox"]))
        assert manual_enc is not None
        enc_diff = float(np.linalg.norm(np.array(f["encoding"]) - np.array(manual_enc)))
        assert enc_diff < 1e-3, f"正照片编码与手工不一致（距离 {enc_diff:.3f}）"
        # extract_face_image 输出 == 原图直接裁剪（未旋转）
        top, right, bottom, left = f["location"]
        direct = img.crop((left, top, right, bottom))
        shown = extract_face_image(UPRIGHT_PHOTO, f["location"], max_width=400)
        direct_r = direct.resize(shown.size, Image.LANCZOS)
        diff = np.abs(np.array(direct_r.convert("RGB"), int)
                      - np.array(shown.convert("RGB"), int)).mean()
        assert diff < 2, f"正照片显示输出被意外变换（像素差 {diff:.1f}）"
    print(f"✓ 正照片 detect_faces/extract_face_image 行为不变（{len(faces)} 张脸）")


if __name__ == "__main__":
    test_roundtrip()
    test_pil_ground_truth()
    test_detect_on_rotated()
    test_upright_unchanged()
    print("\n全部通过 ✅")
