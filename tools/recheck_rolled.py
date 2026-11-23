"""重跑 scan 报告里"带 roll"的照片（去掉 180° 之后的最终结果）。"""
import sys, os, json, sqlite3
from concurrent.futures import ProcessPoolExecutor

ROOT = r"D:\CADD\workspace\qclaw\projects\face-photo-sort"
sys.path.insert(0, ROOT)
REPORT = os.path.join(ROOT, "data", "_yolo_miss_sideways.json")


def work(path):
    from core.face_detector import FaceDetector
    d = FaceDetector()
    fs = d.detect_faces(path)
    return {"path": path,
            "n": len(fs),
            "rolls": [f.get("roll") for f in fs],
            "faces": fs}


def main():
    rep = json.load(open(REPORT, encoding="utf-8"))
    rolled = [c for c in rep["changed"] if any(c["rolls"])]
    paths = [c["path"] for c in rolled]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(work, paths, chunksize=2))
    con = sqlite3.connect(os.path.join(ROOT, "databases", "photo_index.db"))
    persons = {p: json.loads(pj or "[]") for p, pj in con.execute("SELECT path, persons FROM photos")}

    final, dropped = [], []
    for r in res:
        p = r["path"]
        if any(r["rolls"]):
            final.append({"path": p, "n": r["n"], "rolls": r["rolls"],
                          "labeled": bool(persons.get(p))})
        else:
            dropped.append(p)
    print(f"重跑 {len(res)} 张：仍带 roll {len(final)}，不再转 {len(dropped)}")
    for d in dropped:
        print("   放弃:", os.path.basename(d))
    for f in final:
        print(f"  {os.path.basename(f['path'])} n={f['n']} rolls={f['rolls']}"
              f"{' [已标注]' if f['labeled'] else ''}")

    unl = [f["path"] for f in final if not f["labeled"]]
    lab = [f["path"] for f in final if f["labeled"]]
    open(os.path.join(ROOT, "data", "_yolo_miss_fix.txt"), "w", encoding="utf-8").write(
        "\n".join(unl) + "\n")
    open(os.path.join(ROOT, "data", "_yolo_miss_labeled.txt"), "w", encoding="utf-8").write(
        "\n".join(lab) + "\n")
    json.dump({"final": final, "dropped": dropped},
              open(os.path.join(ROOT, "data", "_yolo_miss_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"未标注 {len(unl)} → data/_yolo_miss_fix.txt | 已标注 {len(lab)} → data/_yolo_miss_labeled.txt")


if __name__ == "__main__":
    main()
