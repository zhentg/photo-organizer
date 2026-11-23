"""
下载 InsightFace / ArcFace ONNX 模型到 models/arcface/ 目录。

支持的模型（按推荐顺序）：
  1. buffalo_l  — 百变大咖秀，L_ResNet100E-IR+ArcFace@MS1M，512维特征，精度最高
  2. w600k_r50  — 轻量版，ResNet50+ArcFace@WebFace42M，512维特征
  3. r100_arcface_ms1m — 标准版，百万人脸训练，512维特征

模型来自 InsightFace Model Zoo（https://github.com/deepinsight/insightface/tree/master/model_zoo）
通过 deepinsight/insightface 官方 CDN 下载。
"""
import os
import sys
import urllib.request

# 备用镜像（国内可访问）
MIRRORS = [
    # deepinsight 官方 CDN
    "https://insightface.cn-sh2.ufileos.com/models/",
    # HuggingFace mirror（部分模型）
    # "https://hf-mirror.com/",
]

MODELS = {
    "buffalo_l": {
        "file": "w600k_mbf.onnx",          # MobileFaceNet+ArcFace，最流行
        "size_mb": 7.2,
        "desc": "百变大咖·轻量版  w600k_mbf.onnx (MobileFaceNet+ArcFace, 512d, ~7MB)",
    },
    "arcface_w600k_r50": {
        "file": "w600k_r50.onnx",           # ResNet50+ArcFace，中等
        "size_mb": 165,
        "desc": "ResNet50+ArcFace  w600k_r50.onnx (512d, ~165MB)",
    },
    # r100 在 Windows 上经常下载损坏，跳过
}

DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "arcface")


def get_url(filename: str) -> str:
    """遍历镜像返回第一个可达的 URL"""
    for base in MIRRORS:
        url = base + filename
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=8)
            return url
        except Exception:
            continue
    # fallback: 直接用官方地址
    return f"https://insightface.cn-sh2.ufileos.com/models/{filename}"


def download_file(url: str, dest: str, chunk_size: int = 8192):
    """下载文件并显示进度条"""
    import sys

    class DownloadProgress:
        def __init__(self, total: int):
            self.total = total
            self.done = 0

        def update(self, chunk: int):
            self.done += chunk
            pct = min(100, self.done * 100 // max(self.total, 1))
            bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
            sys.stdout.write(f"\r  [{bar}] {pct}% ({self.done//1024}KB/{self.total//1024}KB)")
            sys.stdout.flush()

    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        print(f"  → {url}")
        print(f"  目标大小: {total / 1024 / 1024:.1f} MB")

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"

        downloaded = 0
        progress = DownloadProgress(total) if total else None

        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress.update(len(chunk))

        os.replace(tmp, dest)
        print(f"\n  ✅ 下载完成: {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    print("=" * 60)
    print("  ArcFace ONNX 模型下载工具")
    print("=" * 60)
    print(f"\n模型存放目录: {os.path.abspath(DEST_DIR)}\n")

    for key, info in MODELS.items():
        filename = info["file"]
        dest = os.path.join(DEST_DIR, filename)

        if os.path.exists(dest):
            size = os.path.getsize(dest) / 1024 / 1024
            print(f"  [跳过] {filename} 已存在 ({size:.1f} MB)")
            continue

        print(f"\n  模型: {info['desc']}")
        url = get_url(filename)
        print(f"  开始下载...")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"\n  ❌ 下载失败: {e}")
            print("  提示: 可以手动下载后放入 models/arcface/ 目录")
            print(f"        下载地址: https://github.com/deepinsight/insightface/tree/master/model_zoo")

    print("\n  下载完成！\n")
    print("  使用方法（运行一次 App 后自动加载）:")
    print("    cd D:\\CADD\\workspace\\qclaw\\projects\\face-photo-sort")
    print("    ..\\..\\..\\tools\\miniconda3\\python.exe -m streamlit run app.py")
    print()
    for key, info in MODELS.items():
        f = info["file"]
        p = os.path.join(DEST_DIR, f)
        status = "✅" if os.path.exists(p) else "❌"
        print(f"    {status} {f}")


if __name__ == "__main__":
    main()
