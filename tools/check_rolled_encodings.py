import sqlite3, json, os
import numpy as np

ROOT = r"D:\CADD\workspace\qclaw\projects\face-photo-sort"
fdb = sqlite3.connect(os.path.join(ROOT, "databases", "face_db.db"))
idx = sqlite3.connect(os.path.join(ROOT, "databases", "photo_index.db"))


def d(a, b):
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 所有已标注且 roll != 0 的照片，逐脸检查：新编码最近的人是不是它自己标的人
rows = idx.execute("SELECT path, faces, persons FROM photos").fetchall()
bad, checked, skipped = [], 0, 0
all_enc = fdb.execute(
    "SELECT p.name, e.encoding, e.photo_path, e.bbox FROM encodings e JOIN persons p ON p.id=e.person_id").fetchall()
parsed = [(n, np.array(json.loads(e)), p, b) for n, e, p, b in all_enc if e]

for path, fj, pj in rows:
    faces, persons = json.loads(fj or "[]"), json.loads(pj or "[]")
    if not any((f.get("roll") or 0) for f in faces):
        continue
    if not persons or len(faces) != len(persons):
        skipped += 1
        continue
    for f, name in zip(faces, persons):
        enc = f.get("encoding")
        if not enc:
            continue
        v = np.array(enc)
        # 该人自己其他照片的编码
        mine = [d(v, e) for n, e, p, b in parsed if n == name and p != path]
        # 其他所有人
        others = [d(v, e) for n, e, p, b in parsed if n != name]
        checked += 1
        self_best = min(mine) if mine else None
        other_best = min(others) if others else None
        ok = self_best is not None and self_best < 0.60 and (
            other_best is None or self_best <= other_best)
        if not ok:
            bad.append((os.path.basename(path), name,
                        None if self_best is None else round(self_best, 3),
                        None if other_best is None else round(other_best, 3)))

print(f"check人脸 {checked} | 对不齐跳过 {skipped} | 可疑 {len(bad)}")
for b in bad:
    print("  ", b)
