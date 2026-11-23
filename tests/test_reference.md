# Face Detection Model Test Reference

## Purpose
Record known-good test images for validating face detection models.
When adding a new model, test it against these images and compare with known-correct results.

## Test Images

### 0. 10004c0.jpg（多人小脸/侧脸场景）
- **Path:** `data/10004c0.jpg`
- **Source:** 数码相机拍摄（国会大厦前水池边）
- **Scene:** 4人合影，前排1男1女（较大），后排2位女士（较小、有遮挡/侧脸）
- **正确答案: 4 张人脸**
- **Baseline (2026-04-20):**
  - OpenCV DNN (conf=0.5): ❌ 只检出 1 张（前排男孩 conf=0.91，后排3人 conf 仅 0.12~0.17 被阈值过滤）
  - OpenCV DNN (conf=0.12) + 眼睛验证: ✅ 正确检出 4 张（兜底方案，耗时 ~1.1s）
  - YOLOv8n-Face (conf=0.5): ✅ 正确检出 4 张（**新默认首选**，耗时 ~0.35s，推荐）
  - YuNet: 待测
  - RetinaFace mnet025: 待测
- **问题根因:** ResNet-SSD 对非正脸/小尺寸/有遮挡的人脸置信度远低于 YOLOv8n-Face
- **默认检测顺序 (2026-04-20 起):** YOLOv8n-Face → OpenCV DNN(conf=0.12 兜底) → YuNet → Haar

---

### 1. DSCN9691.JPG
- **Path:** `D:\photoes\Nikon相机照片\比特恢复的文件\02 本地磁盘 (E ) 删除文件  (5222 个文件)\Dir00310742\DSCN9691.JPG`
- **Source:** Nikon 相机拍摄
- **Scene:** 人像照片
- **正确答案: 1 张人脸**
- **Baseline (2026-04-20 更新):**
  - OpenCV DNN: ✅ 正确
  - YOLOv8n-Face: ✅ 正确
  - YuNet: ❌ 错误
  - RetinaFace mnet025: ❌ 错误（FPN anchor 解码待校准）

## How to Use
1. 将新模型添加到 `pages/9_🔬_模型对比.py` 的检测器列表
2. 上传本文件中的测试图片
3. 对比新模型结果与上述 baseline
4. 检测框应准确框住所有人脸，无漏检/误检
5. 更新此文件中的 baseline 记录
