# 📸 人脸照片分类工具

基于人脸识别的照片自动分类工具，使用 Streamlit 构建 Web 界面。

## 功能特性

- 📁 **批量导入** - 支持文件夹路径导入和拖拽上传
- 🔍 **多检测器融合** - YuNet（深度学习）+ YOLOv8n-Face + Haar Cascade 自动切换
- 👁️ **眼睛验证** - 对检测结果二次过滤，降低误检率
- 👤 **手动标注** - 用户指定每个人脸对应的姓名
- 📂 **分类浏览** - 按人物查看分类后的照片
- 💾 **导入导出** - 导出/导入人脸库和照片索引
- 📤 **照片导出** - 按人物自动整理照片到不同文件夹
- ⚙️ **参数可调** - 检测灵敏度、匹配阈值等全部可通过界面配置

## 技术方案

- **Web 框架**: Streamlit
- **人脸检测**: YuNet (ONNX) / YOLOv8n-Face / OpenCV Haar Cascade（自动降级）
- **特征编码**: 颜色直方图
- **数据存储**: SQLite（持久化）

## 检测器说明

| 检测器 | 特点 | 优先级 |
|--------|------|--------|
| YuNet | 深度学习，误检率极低 | 主检测器 |
| YOLOv8n-Face | YOLO 架构，对极端角度更鲁棒 | 备选 |
| Haar Cascade | OpenCV 经典算法，无需额外模型 | 最终回退 |

> 模型文件 `face_yolov8n.pt`（6.2MB）已自动下载并放置在 `models/yolov8n-face/` 目录

## 安装依赖

```bash
cd D:\CADD\workspace\qclaw\projects\face-photo-sort
pip install -r requirements.txt
```

YOLOv8n-Face 模型（可选，推荐下载以提升检测效果）：

```bash
# 下载 yolov8n-face.onnx 放入 models/yolov8n-face/ 目录
# 下载地址: https://huggingface.co/iqbalancinginho/YOLOV8n-face
```

## 运行

```bash
streamlit run app.py
```

或使用完整路径：

```bash
"D:\CADD\workspace\qclaw\tools\miniconda3\python.exe" -m streamlit run "D:\CADD\workspace\qclaw\projects\face-photo-sort\app.py"
```

## 可调参数

### 检测参数
- **检测灵敏度**（1-10）：控制检测到人脸的难易程度，值越高召回率越高
- **NMS 阈值**：合并重叠检测框的阈值，越低越严格
- **最小人脸尺寸**：低于此像素的人脸将被忽略
- **眼睛验证**：启用后对检测结果做眼睛验证，进一步过滤误检
- **严格模式**：必须检测到眼睛才算有效人脸，误检最少

### 匹配参数
- **匹配阈值**（0.30-0.75）：距离小于此值视为同一人，越小越严格

## 目录结构

```
face-photo-sort/
├── app.py                 # Streamlit 主程序
├── core/
│   ├── face_detector.py   # 人脸检测、匹配、索引核心逻辑
│   └── __init__.py
├── models/                # 模型与数据
│   ├── face_detection_yunet_2023mar.onnx  # YuNet 模型
│   ├── yolov8n-face/                       # YOLO 模型（需手动下载）
│   │   └── yolov8n-face.onnx
│   ├── face_db.db          # 人脸库（SQLite）
│   └── photo_index.db      # 照片索引（SQLite）
├── data/                   # 测试照片
├── output/                 # 导出照片目录
├── requirements.txt        # Python 依赖
└── README.md
```

## 数据文件

### face_db.db（SQLite）
存储已标注人物及其人脸编码特征。

### photo_index.db（SQLite）
存储照片与人脸的对应关系，支持 MD5 去重。

## 注意事项

- YOLOv8n-Face 模型下载后放入 `models/yolov8n-face/` 目录，否则使用 Haar 回退
- 建议照片中人脸清晰、正面、光照均匀
- 定期通过「导出管理」备份人脸库和索引
