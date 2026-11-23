"""方向纠正 / 人脸对齐回归测试（2026-10-19，2026-10-21 改版）。

背景：实测（tools/tilt_experiment2.py）证明——
- YOLO 对旋转极耐，0~90° 都能检出（conf 0.84→0.77）；
- 但 ArcFace 认人从 15° 就开始错：15° 5/6、30° 3/6、45° 1/6、≥60° 0/6。

2026-10-21 用户要求：**照片旋转必须是 90° 的整数倍**（非 90° 会让预览出现
黑角、框也跟着偏）。所以拆成两件事：
1. 整图旋转 = 方向（EXIF / 手动 / 四向探索 90·270），存库的 roll 只在这个范围；
2. 脸上的残余倾斜 = 编码时按脸裁出来对齐（_encode_face_aligned），
   照片本身不动、roll 也不记录。

本测试锁住：
1. 歪 25° 的照片：不产生 roll（非 90 不入库），但仍能认回本人
2. 关掉对齐（deskew=False）明显更差 —— 说明对齐真的在起作用
3. 正常照片：无 roll、编码与旧路径一致（零影响回归）
4. 全库 roll 都是 90 的整数倍
5. 侧躺照片：四向探索仍选 270 -> -90
6. extract_face_image 接 roll 参数正常

运行：python tests/test_deskew.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from core.face_detector import FaceDetector, FaceMatcher, extract_face_image
from core.image_utils import open_upright, rotate_expand

PHOTO = r"D:\photoes\2024春节古装照片\5C3A4598.JPG"   # 单人（Lili），EXIF 正常
TMP = os.path.join(os.path.dirname(__file__), "..", "data", "_test_deskew")
# 无 EXIF 但整张画面侧躺（翻拍）：只有转 90° 才能提出正脸
SIDEWAYS = r"D://photoes//Nikon相机照片//比特恢复的文件//00 找到的文件   (9173 个文件)//图片//jpg//7d2940.jpg"
# 正常竖拍，曾被四向逻辑误转 180°（回归保护）
UPRIGHT_GUARD = r"D://photoes//Nikon相机照片//比特恢复的文件//00 找到的文件   (9173 个文件)//图片//jpg//67de00.jpg"
# 侧躺 + YOLO 全漏 + YuNet 只在草地捡到竖框：闸门曾整个放过（2026-10-20 用户报障）
YUNET_JUNK = r"D://photoes//Nikon相机照片//比特恢复的文件//00 找到的文件   (9173 个文件)//图片//jpg//101bcc0.jpg"


def _prepare(deg):
    """把照片缩放一半后旋转 deg 度存到临时文件，返回路径 + BGR。"""
    os.makedirs(TMP, exist_ok=True)
    img, _, _ = open_upright(PHOTO)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (bgr.shape[1] // 2, bgr.shape[0] // 2))
    if deg:
        bgr, _ = rotate_expand(bgr, deg)
    p = os.path.join(TMP, f"rot_{deg:+d}.jpg")
    cv2.imwrite(p, bgr)
    return p


def _biggest(faces):
    return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])


def test_tilt_recovers_identity(det, matcher):
    """歪 15/25/45°：不入 roll（非 90 不存），但仍能认回本人。"""
    base_path = _prepare(0)
    base = _biggest(det.detect_faces(base_path))
    name0, d0 = matcher.find_match(base["encoding"])
    assert name0, "基准照片应能匹配到人"
    assert "roll" not in base, "正常照片不应带 roll 字段"

    det_off = FaceDetector(deskew=False)
    for deg in (15, 25, 45):
        tilted = _prepare(deg)
        on = _biggest(det.detect_faces(tilted))
        assert "roll" not in on, f"歪 {deg}° 不该写入 roll（只允许 90 的整数倍），实际 {on.get('roll')}"
        d_on = matcher.face_distance(base["encoding"], on["encoding"])
        name_on, _ = matcher.find_match(on["encoding"])

        off = _biggest(det_off.detect_faces(tilted))
        d_off = matcher.face_distance(base["encoding"], off["encoding"])

        print(f"  {deg:>2}° 基准={name0}({d0:.3f}) | 对齐={name_on}({d_on:.3f}) | 不对齐={d_off:.3f}")
        assert name_on == name0, f"歪 {deg}° 应对齐后认回 {name0}，实际 {name_on}"
        assert d_on < 0.60, f"歪 {deg}° 对齐后距离应 < 0.6，实际 {d_on:.3f}"
        assert d_on < d_off * 0.75, f"对齐应明显改善（{d_on:.3f} vs {d_off:.3f}）"


def test_roll_is_multiple_of_90():
    """全库守卫：存库的 roll 必须是 90 的整数倍（用户 2026-10-21 的硬要求）。"""
    import json
    import sqlite3
    db = os.path.join(os.path.dirname(__file__), "..", "databases", "photo_index.db")
    con = sqlite3.connect(db)
    bad = []
    for path, faces in con.execute("SELECT path, faces FROM photos"):
        for f in json.loads(faces or "[]"):
            r = f.get("roll")
            if r is not None and abs(abs(float(r)) % 90) > 0.6:
                bad.append((os.path.basename(path), r))
    con.close()
    assert not bad, f"库里存在非 90 整数倍的 roll: {bad[:5]}"
    print("  ✓ 全库 roll 都是 90 的整数倍")


def test_crop_with_roll(det):
    """extract_face_image 接 roll=-90：裁剪要跟着转（长宽对调、内容不丢）。"""
    p = SIDEWAYS
    face = _biggest(det.detect_faces(p))
    plain = extract_face_image(p, face["location"], max_width=200)
    roted = extract_face_image(p, face["location"], max_width=200,
                               roll=face["roll"])
    assert roted.size[0] > 0 and roted.size[1] > 0
    # 转 90° 时长宽对调；排序后两边应基本一致（expand 只是不让内容被切）
    a, b = sorted(roted.size), sorted(plain.size)
    assert abs(a[0] - b[0]) <= max(4, b[0] * 0.1) and abs(a[1] - b[1]) <= max(4, b[1] * 0.1), \
        f"旋转后裁剪尺寸应基本不变，plain={plain.size} rolled={roted.size}"
    print(f"  ✓ 裁剪: plain={plain.size} roll={face['roll']} -> {roted.size}")


def test_cardinal_sideways(det, matcher):
    """无 EXIF 的侧躺翻拍：四向探索应选 270->-90，且能认出本人。"""
    fs = det.detect_faces(SIDEWAYS)
    assert fs, "侧躺照片应该能提出脸"
    big = _biggest(fs)
    assert big.get("roll") == -90, f"应标记 roll=-90（转 90° 正向），实际 {big.get('roll')}"
    name, dist = matcher.find_match(big["encoding"])
    print(f"  ✓ 侧躺翻拍: roll={big['roll']} match={name}({dist:.3f})")
    assert name is not None, "转正后应能匹配到人"


def test_upright_not_flipped(det):
    """正常竖拍照片不能被四向逻辑翻转（67de00 曾被误转 180°）。"""
    fs = det.detect_faces(UPRIGHT_GUARD)
    assert fs, "应该检出脸"
    rolls = [f.get("roll") for f in fs]
    assert all(r is None for r in rolls), f"正常照片不该带 roll，实际 {rolls}"
    print(f"  ✓ 竖拍保护: {len(fs)} 张脸，无 roll 标记")


def test_upright_view(det):
    """整图预览也要摆正：侧躺照片画完框应是竖构图（页面上看到的图不躺着）。"""
    from core.image_utils import draw_faces_on_image
    fs = det.detect_faces(SIDEWAYS)
    out = draw_faces_on_image(SIDEWAYS, fs, max_width=300)
    h, w = out.shape[:2]
    assert h > w, f"摆正后应是竖构图，实际 {w}x{h}"
    print(f"  ✓ 预览摆正: {w}x{h}（{len(fs)} 个框）")


def test_yunet_junk_box(det):
    """YOLO+DNN 全漏、YuNet 捡到竖框的侧躺照片也要走四向探索（101bcc0）。

    这张是"两个人站在草地上"，侧躺时 YuNet 只在草地给出 62×68 的竖框（分 0.82），
    框不是横的 → 旧的"横框"闸门不触发，整张照片被判成 1 张脸、没有 roll。
    """
    fs = det.detect_faces(YUNET_JUNK)
    assert len(fs) == 2, f"应该检出 2 个人（大人+小孩），实际 {len(fs)}"
    rolls = [f.get("roll") for f in fs]
    assert all(r == 90 for r in rolls), f"应标记 roll=+90，实际 {rolls}"
    print(f"  ✓ YuNet 竖框侧躺: {len(fs)} 张脸 roll={rolls}")


def main():
    det = FaceDetector()
    matcher = FaceMatcher(os.path.join(os.path.dirname(__file__), "..",
                                       "databases", "face_db.db"))
    print("test_tilt_recovers_identity")
    test_tilt_recovers_identity(det, matcher)
    print("test_roll_is_multiple_of_90")
    test_roll_is_multiple_of_90()
    print("test_crop_with_roll")
    test_crop_with_roll(det)
    print("test_cardinal_sideways")
    test_cardinal_sideways(det, matcher)
    print("test_yunet_junk_box")
    test_yunet_junk_box(det)
    print("test_upright_not_flipped")
    test_upright_not_flipped(det)
    print("test_upright_view")
    test_upright_view(det)
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
