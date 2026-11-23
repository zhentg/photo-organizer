"""扫描 / 匹配准确度回归测试（2026-11-11 全页体检修复项）。

锁住 5 件事：
1. persons 精确匹配：大小写、通配符、子串都不能误命中（原 LIKE '%"name"%'）。
2. 陌生人计数按「照片」去重，不是按 persons 里的重复条目计数。
3. 假脸黑名单用独立的严格阈值，不复用 match_threshold（否则同一个人的
   正常照片会被当假脸过滤）。
4. 重复编码检测对 512 维 ArcFace 编码有效（原 `len(enc)<128` 恒为空）。
5. add_photo 的 UPDATE 分支：persons=None 保留、[] 清空、名单覆盖。

运行：python tests/test_scan_accuracy.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.face_detector import (PhotoManager, FaceMatcher,
                                FALSE_FACE_THRESHOLD)
from core.stranger import STRANGER_NAME

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _vec(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(float)
    return (v / np.linalg.norm(v)).tolist()


def _at_distance(base, dist, seed=5):
    """构造一个与 base 的余弦距离恰好为 dist 的 512 维单位向量。"""
    a = np.asarray(base, dtype=float)
    a = a / np.linalg.norm(a)
    rng = np.random.RandomState(seed)
    w = rng.randn(512)
    w -= w.dot(a) * a
    w /= np.linalg.norm(w)
    cos = 1.0 - dist
    return (cos * a + np.sqrt(max(0.0, 1 - cos * cos)) * w).tolist()


def test_person_matching_exact():
    tmp = tempfile.mkdtemp(prefix="fps_json_")
    try:
        pm = PhotoManager(os.path.join(tmp, "index.db"), os.path.join(tmp, "out"))
        pm.add_photo("anne.jpg", [], ["Anne"])
        pm.add_photo("anne_lower.jpg", [], ["anne"])
        pm.add_photo("tom_pct.jpg", [], ["Tom%"])
        pm.add_photo("multi.jpg", [], [STRANGER_NAME, STRANGER_NAME])
        pm.add_photo("none.jpg", [])

        by_name = lambda n: {p["path"] for p in pm.get_photos_by_person(n)}
        assert by_name("Anne") == {"anne.jpg"}, by_name("Anne")
        assert by_name("anne") == {"anne_lower.jpg"}, by_name("anne")
        assert by_name("Tom%") == {"tom_pct.jpg"}, "名字里的 % 不能被当通配符"
        assert by_name("TomX") == set(), "LIKE 通配符会误命中，json_each 不该"
        assert pm.get_stranger_photo_count() == 1, "一张照片里两个陌生人不该算 2 张"
        print("  ✓ persons 精确匹配（大小写/通配符安全）+ 陌生人按照片去重")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_false_face_threshold_and_duplicates():
    tmp = tempfile.mkdtemp(prefix="fps_match_")
    try:
        fm = FaceMatcher(os.path.join(tmp, "face.db"), match_threshold=0.60)
        assert FALSE_FACE_THRESHOLD < fm.match_threshold, \
            "假脸阈值必须比同人阈值严格"

        junk = _vec(7)
        fm.add_false_face(junk, "junk.jpg", (0, 0, 10, 10), (0, 0, 10, 10))
        assert fm._is_false_face(junk) is True, "同一张误检必须被拦下"

        # 与假脸距离约 0.45（>0.30 但 <0.60）：旧实现会误判成假脸，
        # 修复后应放行；find_match 也不该返回 (None, inf)。
        near = _at_distance(junk, 0.45)
        d = fm.face_distance(junk, near)
        assert 0.30 < d < 0.60, f"构造的样本距离不对: {d:.3f}"
        assert fm._is_false_face(near) is False, \
            f"距离 {d:.3f} 的真脸被当成假脸（复用 0.6 阈值的老 bug）"

        a = _vec(11)
        fm.add_person("Alice", a, "a.jpg", (0, 0, 10, 10), (0, 0, 10, 10))
        fm.add_person("Bob", _at_distance(a, 0.05), "b.jpg",
                      (0, 0, 10, 10), (1, 1, 10, 10))
        dups = fm.get_duplicate_encodings(0.40)
        assert dups, "512 维编码的重复检测仍然是空（len(enc)<128 的老 bug）"
        assert all(x["distance"] < 0.40 for x in dups)
        print(f"  ✓ 假脸阈值解耦（{FALSE_FACE_THRESHOLD}）+ "
              f"512 维重复编码检出 {len(dups)} 对")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_add_photo_persons_on_update():
    tmp = tempfile.mkdtemp(prefix="fps_addphoto_")
    try:
        pm = PhotoManager(os.path.join(tmp, "index.db"), os.path.join(tmp, "out"))
        pm.add_photo("a.jpg", [{"bbox": [0, 0, 1, 1]}], ["Alice"])
        pm.add_photo("a.jpg", [{"bbox": [0, 0, 2, 2]}])          # 省略 = 保留
        assert pm.get_photo_faces_persons("a.jpg")[1] == ["Alice"]
        pm.add_photo("a.jpg", [{"bbox": [0, 0, 3, 3]}], [])      # [] = 清空
        assert pm.get_photo_faces_persons("a.jpg")[1] == []
        pm.add_photo("a.jpg", [{"bbox": [0, 0, 4, 4]}], ["Bob"])  # 名单 = 覆盖
        assert pm.get_photo_faces_persons("a.jpg")[1] == ["Bob"]
        print("  ✓ add_photo UPDATE：保留 / 清空 / 覆盖 persons")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_person_matching_exact()
    test_false_face_threshold_and_duplicates()
    test_add_photo_persons_on_update()
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
