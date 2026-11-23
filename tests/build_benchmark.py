"""
Build test set by scanning photo directories directly
"""
import os, json, random
from pathlib import Path
from PIL import Image

# Find photo directories
photo_dirs = [r"D:\photoes"]
image_exts = {'.jpg', '.jpeg', '.png'}

candidates = []
for d in photo_dirs:
    if not os.path.isdir(d):
        continue
    count = 0
    for root, dirs, files in os.walk(d):
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                fp = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fp)
                    if sz > 10_000:  # > 10KB
                        candidates.append(fp)
                        count += 1
                except:
                    pass
                if count >= 500:
                    break
        if count >= 500:
            break

print(f"Found {len(candidates)} candidates")

# Categorize by size (proxy for complexity)
small = [p for p in candidates if os.path.getsize(p) < 500_000]    # <500KB
medium = [p for p in candidates if 500_000 <= os.path.getsize(p) < 2_000_000]
large = [p for p in candidates if os.path.getsize(p) >= 2_000_000]

print(f"Small: {len(small)}, Medium: {len(medium)}, Large: {len(large)}")

# Pick 10 diverse photos
random.seed(42)
test_set = []
for bucket in [small, medium, large]:
    n = 3 if bucket else 0
    if len(bucket) >= n:
        test_set.extend(random.sample(bucket, n))
    elif bucket:
        test_set.extend(bucket)

# Add 260740.jpg explicitly
for p in candidates:
    if '260740' in p:
        if p not in test_set:
            test_set.append(p)

# Get dimensions
results = []
for p in test_set:
    try:
        img = Image.open(p)
        w, h = img.size
        results.append({
            "path": p,
            "size_kb": round(os.path.getsize(p) / 1024),
            "width": w,
            "height": h
        })
        img.close()
    except:
        pass

out_path = r"D:\CADD\workspace\qclaw\projects\face-photo-sort\tests\benchmark_set.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nTest set ({len(results)} photos):")
for i, p in enumerate(results):
    print(f"  {i+1}. {os.path.basename(p['path'])} | {p['size_kb']}KB | {p['width']}x{p['height']}")

from pathlib import Path
