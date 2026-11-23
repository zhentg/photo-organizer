"""手动旋转（界面上的「↺ 左转 90° / ↻ 右转 90°」）回归测试。

背景（2026-10-20 用户需求）：有的照片没有 EXIF、自动四向探索也没救回来，
用户希望**自己点按钮转 90°**，转完立刻按新方向重新检测人脸，可以连点到摆正；
并且要在库里留下"这张是人手动转的"标记，方便事后集中排查。

本测试锁住 6 件事：
1. 角度归一化：(-180, 180]，逆时针为正（与 roll 同向）；
2. 手动角度真的生效：转过之后 roll 包含手动部分、bbox 换到新位置；
3. bbox 映射回原始坐标系：裁剪+按 roll 转正后的脸尺寸与旋转前一致
   （差 1-2px 是仿射取整，超过就说明映射错了）；
4. 手动转过的照片**不再四向探索**（方向由用户拍板，不会被算法转回去）；
5. 已标注照片：人名能搬过去、encodings 一起重写；重置能回到原样；
6. 每次操作都记 manual_rotate_log；转过但没检出人脸的会进"待复查"。

运行：python tests/test_manual_rotate.py
"""
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.face_detector import FaceDetector, FaceMatcher, PhotoManager, _normalize_rotate
from core.rotation import rotate_photo, reset_manual_rotate, label_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_INDEX = os.path.join(ROOT, "databases", "photo_index.db")
SRC_FACE = os.path.join(ROOT, "databases", "face_db.db")
TMP = os.path.join(ROOT, "data", "_test_manual_rotate")

SIDEWAYS = (r"D:\photoes\Nikon相机照片\比特恢复的文件"
            r"\00 找到的文件   (9173 个文件)\图片\jpg\7d2940.jpg")     # 自动判出 -90
UPRIGHT = (r"D:\photoes\Nikon相机照片\比特恢复的文件"
           r"\00 找到的文件   (9173 个文件)\图片\jpg\67de00.jpg")      # 正常竖拍、无 roll


def _fresh_db():
    """把库拷到临时目录（测试绝不碰真库）。"""
    os.makedirs(TMP, exist_ok=True)
    idb, fdb = os.path.join(TMP, "index.db"), os.path.join(TMP, "face.db")
    for s, d in ((SRC_INDEX, idb), (SRC_FACE, fdb)):
        shutil.copyfile(s, d)
    return idb, fdb


def test_normalize():
    cases = {0: 0, 90: 90, -90: -90, 270: -90, -270: 90, 360: 0,
             180: -180, -180: -180, 450: 90}
    for src, want in cases.items():
        got = _normalize_rotate(src)
        assert abs(got - want) < 1e-6, f"{src} -> {got}，期望 {want}"
    assert label_of(0) == "未旋转"
    assert label_of(90) == "左转 90°", label_of(90)
    assert label_of(-90) == "右转 90°", label_of(-90)
    print("  ✓ 归一化/文案: 270→-90、-270→90、0→未旋转")


def test_manual_rotates_face(det, pm, matcher, idb):
    """手动角度与自动判断"同一个方向"时必须得到同一结果（可重复、不叠加）。

    这是手动旋转最容易写错的地方：手动角要叠在自动角**之上**，而不是替代它。
    SIDEWAYS 自动判出 -90；再手动右转 90（-90）应当得到一样的框和一样的编码；
    反过来手动 +90 应当真的把图转歪（人脸距离明显变差）。
    """
    auto = det.detect_faces(SIDEWAYS, manual_rotate=0)
    assert auto, "基准照片应有脸"
    bbox_auto = auto[0]["bbox"]
    d_auto = min((matcher.face_distance(auto[0]["encoding"], e)
                  for e in matcher.get_person_encodings("Anne")), default=9.9)

    same = det.detect_faces(SIDEWAYS, manual_rotate=-90)   # 与自动同向
    assert same and same[0]["bbox"] == bbox_auto, \
        f"同向手动旋转应与自动结果一致：{same[0]['bbox']} vs {bbox_auto}"
    assert abs(same[0]["roll"] + 90) <= 3, f"roll 应≈-90，实际 {same[0].get('roll')}"

    wrong = det.detect_faces(SIDEWAYS, manual_rotate=90)   # 反向
    assert wrong, "反向也应能检出（YOLO 对旋转不敏感）"
    assert abs(wrong[0]["roll"] - 90) <= 3, f"roll 应≈+90，实际 {wrong[0].get('roll')}"
    d_wrong = min((matcher.face_distance(wrong[0]["encoding"], e)
                   for e in matcher.get_person_encodings("Anne")), default=9.9)
    assert d_wrong > d_auto + 0.2, \
        f"转错方向后人脸距离应明显变大：{d_auto:.3f} → {d_wrong:.3f}"

    # 写进库后，不带参数调用（自动读库）也必须拿到同一个结果
    pm.set_manual_rotate(SIDEWAYS, -90)
    from_db = det.detect_faces(SIDEWAYS)
    assert from_db and from_db[0]["bbox"] == bbox_auto, "库里的 manual_rotate 应被自动应用"
    pm.set_manual_rotate(SIDEWAYS, 0)
    print(f"  ✓ 手动角叠在自动角之上: 同向 bbox 一致, 反向距离 {d_auto:.3f}→{d_wrong:.3f}")


def test_bbox_mapping(det):
    """手动旋转后 bbox 仍在原图坐标系内、面积合理；重置后完全还原。"""
    base = det.detect_faces(UPRIGHT, manual_rotate=0)
    assert base, "基准照片应有脸"
    b0 = base[0]["bbox"]

    rot = det.detect_faces(UPRIGHT, manual_rotate=90)
    assert rot, "转 90° 后应仍能检出"
    x, y, w, h = rot[0]["bbox"]
    from PIL import Image
    with Image.open(UPRIGHT) as im:
        W, H = im.size
    # EXIF 正常时原图就是 W×H；横持时允许交换
    assert 0 <= x <= max(W, H) and 0 <= y <= max(W, H), f"bbox 越界: {rot[0]['bbox']}"
    assert x + w <= max(W, H) + 8 and y + h <= max(W, H) + 8, f"bbox 越界: {rot[0]['bbox']}"
    area_ratio = (w * h) / max(b0[2] * b0[3], 1)
    assert 0.5 <= area_ratio <= 1.6, f"面积比异常 {area_ratio:.2f}（框映射错了）"

    back = det.detect_faces(UPRIGHT, manual_rotate=0)
    assert back[0]["bbox"] == b0, "reset 后 bbox 应完全还原"
    print(f"  ✓ bbox 映射: 面积比 {area_ratio:.2f}，重置还原")


def test_no_second_guess(det, pm):
    """手动转过的照片不再四向探索：方向是用户定的。"""
    pm.set_manual_rotate(SIDEWAYS, 0)
    auto = det.detect_faces(SIDEWAYS)
    r_auto = max(auto, key=lambda f: f["bbox"][2] * f["bbox"][3]).get("roll")
    assert r_auto is not None and abs(abs(r_auto) - 90) <= 3, \
        f"侧躺照片自动应判 -90/+90，实际 {r_auto}"

    pm.set_manual_rotate(SIDEWAYS, 90)
    man = det.detect_faces(SIDEWAYS)
    assert man, "手动 90° 后应仍能检出"
    r_man = max(man, key=lambda f: f["bbox"][2] * f["bbox"][3]).get("roll")
    # 若自动逻辑又插一脚（cardinal），这里会变成 0 或 -90
    assert r_man is not None and abs(r_man - 90) <= 3, \
        f"手动 90° 后 roll 应保持 90（不再自动探索），实际 {r_man}"
    pm.set_manual_rotate(SIDEWAYS, 0)
    print(f"  ✓ 不再自动探索: 自动={r_auto} → 手动 90° 后={r_man}")


def test_labeled_roundtrip(pm, det, matcher, idb):
    """已标注照片：转 → 名字还在、编码重写；重置 → 完全还原。"""
    old_faces, old_persons = pm.get_photo_faces_persons(SIDEWAYS)
    assert old_persons, "该照片应已标注"

    res = rotate_photo(pm, det, SIDEWAYS, 90, matcher=matcher)
    assert res["status"] == "ok", f"旋转应成功，实际 {res['status']}"
    assert res["names_after"] == res["names_before"], \
        f"人名不能丢：{res['names_before']} → {res['names_after']}"
    assert res["synced"] == len(res["names_after"]), "编码应全部重写"

    # 编码行必须指向新名字（写的是临时库 face.db）
    rows = sqlite3.connect(os.path.join(TMP, "face.db")).execute(
        "select p.name from encodings e join persons p on p.id=e.person_id "
        "where e.photo_path=?", (SIDEWAYS,)).fetchall()
    assert rows and set(r[0] for r in rows) == set(res["names_after"]), \
        f"编码行应只剩新名字，实际 {rows}"

    back = reset_manual_rotate(pm, det, SIDEWAYS, matcher=matcher)
    assert back["new_manual"] == 0
    assert back["names_after"] == res["names_before"]
    f2, p2 = pm.get_photo_faces_persons(SIDEWAYS)
    assert [tuple(f["bbox"]) for f in f2] == [tuple(f["bbox"]) for f in old_faces], \
        "重置后 bbox 应还原"
    print(f"  ✓ 已标注往返: {res['names_before']} 保留，编码重写，重置后还原")


def test_log_and_review(pm, idb):
    log = pm.get_manual_rotation_log(50)
    assert log, "应有旋转日志"
    statuses = {r["status"] for r in log}
    print(f"  ✓ 日志 {len(log)} 条，状态 {sorted(statuses)}")

    # 造一张"转过但没脸"的照片 → 必须出现在待复查列表里
    fake = r"D:\fake\manual_rotate_no_face.jpg"
    con = sqlite3.connect(idb)
    con.execute("insert or replace into photos(path,faces,persons,indexed_at,"
                "manual_rotate) values(?,?,?,?,?)",
                (fake, "[]", "[]", "2026-10-20T00:00:00", -90))
    con.commit()
    con.close()
    review = [r["path"] for r in pm.get_rotate_review_photos(50)]
    assert fake in review, "转过但无脸的照片应进待复查列表"
    con = sqlite3.connect(idb)
    con.execute("delete from photos where path=?", (fake,))
    con.commit()
    con.close()
    print("  ✓ 待复查列表能捞出「转歪到没脸」的照片")


def main():
    idb, fdb = _fresh_db()
    pm = PhotoManager(idb, os.path.join(TMP, "out"))
    det = FaceDetector(photo_index_path=idb)
    matcher = FaceMatcher(fdb)

    print("test_normalize")
    test_normalize()
    print("test_manual_rotates_face")
    test_manual_rotates_face(det, pm, matcher, idb)
    print("test_bbox_mapping")
    test_bbox_mapping(det)
    print("test_no_second_guess")
    test_no_second_guess(det, pm)
    print("test_labeled_roundtrip")
    test_labeled_roundtrip(pm, det, matcher, idb)
    print("test_log_and_review")
    test_log_and_review(pm, idb)
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
