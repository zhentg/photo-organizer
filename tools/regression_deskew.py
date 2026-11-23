"""回归抽样：正常照片（索引里 roll 为空）在新闸门下会不会被误转/误改数量。

用法: python tools/regression_deskew.py [n] [--seed S]
只读，不写库。
"""
import sys, os, json, sqlite3, random
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def work(path):
    from core.face_detector import FaceDetector
    d = FaceDetector()
    try:
        fs = d.detect_faces(path)
    except Exception as e:
        return (path, -1, None, f"error {e}")
    return (path, len(fs), [f.get("roll") for f in fs], d._stats)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seed = 7
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    con = sqlite3.connect(os.path.join(ROOT, "databases", "photo_index.db"))
    rows = con.execute("SELECT path, faces FROM photos").fetchall()
    recent_only = "--recent" in sys.argv
    if recent_only:
        # 只比"当前管线"写过的照片（4 月的旧索引差异太多，测不出闸门改动的影响）
        rows = con.execute(
            "SELECT path, faces FROM photos WHERE indexed_at>='2026-10-19'").fetchall()
    plain = []
    for p, fj in rows:
        fs = json.loads(fj or "[]")
        if not fs or any(f.get("roll") for f in fs):
            continue
        if not os.path.exists(p):
            continue
        plain.append((p, len(fs)))
    random.Random(seed).shuffle(plain)
    sample = plain[:n]
    print(f"抽样 {len(sample)} 张（全库正常照片 {len(plain)} 张），seed={seed}")

    with Pool(4) as pool:
        out = pool.map(work, [p for p, _ in sample])
    old = dict(sample)
    flipped, count_changed, err = [], [], []
    for path, n_new, rolls, stats in out:
        name = os.path.basename(path)
        if n_new < 0:
            err.append((name, stats))
            continue
        if any(rolls):
            flipped.append((name, old[path], n_new, rolls))
        elif n_new != old[path]:
            count_changed.append((name, old[path], n_new))
    print(f"误转（新增 roll）: {len(flipped)}")
    for f in flipped:
        print("   ", f)
    print(f"数量变化: {len(count_changed)}")
    for f in count_changed:
        print("   ", f)
    print(f"异常: {len(err)}")
    for e in err:
        print("   ", e)


if __name__ == "__main__":
    main()
