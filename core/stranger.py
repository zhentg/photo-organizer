# core/stranger.py — 「陌生人」批量标记（2026-10-19）
"""
需求来源：有的照片上所有人都是陌生人（游客照里的路人、背景人物），
逐个人脸命名太慢——要一个按钮把整张照片一键标为「陌生人」。

「陌生人」是**照片级桶标签**，不是人脸库里的"人"：
  - 只写 photo_index.persons（如 ["陌生人", "陌生人"]），**不写 encodings**。
    原因一：find_match 是全局最近邻（core/face_index.py），陌生脸会汇成
    几百上千条的巨型簇，把已知人物的脸错误匹配成「陌生人」——预填和
    「全库匹配」都会被污染。
    原因二：陌生人日后可能想命名，而 update_person_name 会连同所有
    陌生人一起改名；作为独立"人"存进人脸库无法安全拆分。
  - 已排除（X 标记）的人脸仍走假脸流程（false_faces + 移出照片 faces），
    与标注页「批量确认」语义完全一致。

纯 core 模块：photo_manager / matcher 由调用方传入（不 import），
便于无头单测（录音替身，不碰真实库）。
"""

STRANGER_NAME = "陌生人"


def mark_photo_as_strangers(photo_manager, matcher, photo_path: str,
                            faces: list, excluded=()) -> int:
    """把一张照片的所有未排除人脸标为「陌生人」，返回标记的陌生人脸数。

    photo_manager / matcher  由调用方传入（测试可用录音替身）。
    faces                    照片的人脸列表，每项含 encoding/location/bbox。
    excluded                 被排除的人脸下标集合（X 标记的非人脸）。

    分支语义与「批量确认」一致：
      - 全部排除 → clear_photo_faces + 每个脸进 false_faces；
      - 部分排除 → 未排除的写 ["陌生人"]*n，排除的进 false_faces 并移出 faces；
      - 无排除   → 全部写陌生人。
    """
    excluded = set(excluded or ())
    if excluded and len(excluded) == len(faces):
        # 所有人脸都是假的，清空照片记录（同批量确认）
        photo_manager.clear_photo_faces(photo_path)
        for face in faces:
            matcher.add_false_face(
                encoding=face["encoding"],
                photo_path=photo_path,
                face_location=face.get("location"),
                bbox=face.get("bbox", (0, 0, 0, 0)),
            )
        return 0

    n_stranger = len(faces) - len(excluded)
    photo_manager.update_photo_persons(photo_path, [STRANGER_NAME] * n_stranger)
    for i, face in enumerate(faces):
        if i not in excluded:
            continue
        matcher.add_false_face(
            encoding=face["encoding"],
            photo_path=photo_path,
            face_location=face.get("location"),
            bbox=face.get("bbox", (0, 0, 0, 0)),
        )
        photo_manager.remove_face_from_photo(photo_path, face.get("bbox", (0, 0, 0, 0)))
    return n_stranger
