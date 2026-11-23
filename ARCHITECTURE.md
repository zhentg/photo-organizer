# Face Photo Sort — 项目架构文档

> 本文档记录人脸照片分类工具的完整设计决策、技术方案和实现细节，
> 可作为后续类似 AI/Web 应用项目的规划模板。

**当前版本：** v4（2026-04-19）
**核心定位：** 离线运行 + 参数可调 + 完全可控的本地人脸照片分类工具。

---

## 一、项目概述

**目标：** 对本地照片按人脸进行自动分类管理，无需云端上传，保护隐私。

**用户画像：** 个人用户，拥有大量照片，希望按人物整理归类。

**核心价值：** 离线运行 + 参数可调 + 完全可控。

---

## 二、技术选型

### 为什么选 Streamlit

| 方案 | 优点 | 缺点 |
|------|------|------|
| Flask/FastAPI + 前端 | 完全可控、性能好 | 开发周期长 |
| Electron | 桌面原生体验 | 打包复杂、Bundle 大 |
| **Streamlit（最终）** | **最快交付、纯 Python、可迭代** | 限制可接受** |

**结论：** 先用 Streamlit 快速交付，后续如需桌面化可考虑 Electron 包装。

### 为什么选 SQLite 而非 JSON

| 方案 | 适用规模 | 优势 | 劣势 |
|------|----------|------|------|
| JSON 文件 | < 1000 条 | 简单、无依赖 | 读写锁问题、数据损坏风险 |
| SQLite | 任意规模 | ACID、增量读写、查询灵活 | 需要迁移逻辑 |
| **SQLite（最终）** | **万级记录完全 OK** | **稳定、可扩展** | **学习成本低** |

**教训：** 初期用 JSON 快速验证方向是对的，但超过 500 条记录后必须迁移到 SQLite。

---

## 三、核心架构图

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                            │
│                      (Dashboard)                         │
└──────────────┬──────────────────────────────────────────┘
               │ st.sidebar: render_sidebar()
               │ 返回 detector / matcher / photo_manager
               ▼
┌──────────────────────────────────────────────────────────┐
│                      pages/                              │
│  1-导入  2-标注  3-浏览  4-索引  5-导出  6-重复  7-缺失  8-假脸  9-模型对比 │
└──────────────────────────────┬───────────────────────────┘
                               │
        ┌──────────────────────┴───────────────────────────┐
        │                  core/shared.py                   │
        │   st.cache_resource 缓存 + 参数映射 + Sidebar UI  │
        └──────────────────────┬───────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        FaceDetector      FaceMatcher      PhotoManager
        (人脸检测)         (人脸库)          (照片索引)
              │                │                │
        ┌─────┴─────┐     face_db.db     photo_index.db
        │           │       (SQLite)        (SQLite)
   OpenCV DNN    编码器:
   ResNet-SSD    ArcFace 512d
   YuNet ONNX    (SFace 128d 备选)
   YOLOv8n-Face
   Haar Cascade
```

---

## 四、目录结构

```
face-photo-sort/
├── app.py                     # 首页 Dashboard（统计概览）
├── core/
│   ├── __init__.py           # 模块导出（FaceDetector / FaceMatcher / PhotoManager）
│   ├── face_detector.py      # 【核心 v4】检测器 + 人脸库 + 照片管理器
│   ├── face_index.py         # 编码向量索引（find_match 加速：numpy 矩阵 + 写入穿透）
│   ├── image_utils.py        # 图片解码工具（extract/draw_faces：draft 降采样 + lru_cache）
│   ├── name_search.py        # 姓名搜索（拼音首字母/全拼/子串匹配，依赖 pypinyin）
│   ├── name_combobox.py      # 单框姓名 combobox（输入即筛选 + radio 建议列表 + 直接新增）
│   ├── big_face_detector.py  # 大图专用检测管线（多尺度策略）
│   └── shared.py              # 【核心】Streamlit 共享组件（缓存/sidebar/参数映射）
├── pages/                     # Streamlit 多页面（目录自动识别，按文件名排序）
│   ├── 1_📁_导入照片.py       # Tab 1: 导入 + 批量检测 + MD5 去重
│   ├── 2_👤_人脸标注.py       # Tab 2: 未标注照片列表 + 批量确认
│   ├── 3_📂_分类浏览.py       # Tab 3: 按人物查看照片
│   ├── 4_📋_照片索引.py       # Tab 4: 照片总表 + 筛选 + 删除
│   ├── 5_💾_导出管理.py       # Tab 5: 导入/导出/重编码
│   ├── 6_🔍_重复管理.py       # Tab 6: MD5 重复照片 + 编码去重
│   ├── 7_⚠️_缺失文件.py       # Tab 7: 清理已删除文件的索引记录
│   ├── 8_🚫_假脸管理.py       # Tab 8: 管理已知假人脸
│   └── 9_🔬_模型对比.py       # Tab 9: 4 个检测器并行对比 + 可视化
├── models/                    # 模型文件
│   ├── opencv_dnn_face/      # OpenCV DNN ResNet-SSD（YOLO 失败后的兜底检测器）
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   ├── face_detection_yunet_2023mar.onnx  # YuNet ONNX（备选检测器）
│   ├── yolov8n-face/         # YOLOv8n-Face（首选检测器）
│   ├── retinaface/           # RetinaFace ONNX（用于对比页面）
│   ├── arcface/              # ArcFace ONNX（512维，优先编码器）
│   │   ├── w600k_mbf.onnx    # ← 主用模型
│   │   └── ARCFACE_README.md
│   └── face_recognition_sface_2021dec.onnx  # SFace 128维（备选编码器）
├── databases/                 # SQLite 数据库（活跃目录）
│   ├── face_db.db            # 人脸库 SQLite（persons/encodings/false_faces）
│   └── photo_index.db        # 照片索引进 SQLite（photos 表）
├── data/                      # 测试照片（!SCN*.JPG）
├── output/                    # 导出照片的输出目录
├── upload/                    # 用户上传的临时文件
├── requirements.txt
├── README.md
└── ARCHITECTURE.md           # 本文档
```

**设计原则：**
- `core/` 是纯业务逻辑，无 Streamlit 依赖，可独立测试
- `pages/` 是 Streamlit UI 页面，只负责渲染和交互
- `shared.py` 承上启下：导入 core，暴露给 pages
- 检测器（定位人脸）与编码器（提取特征向量）完全解耦

---

## 五、检测器架构（v4）

### 优先级链路（2026-05-17 起）

```
YOLOv8n-Face    ← 首选（多张小脸/侧脸最鲁棒，速度最快）
        ↓ 失败 / 不可用
OpenCV DNN ResNet-SSD  ← 兜底（opencv_dnn_conf，默认 0.12）
        ↓ 失败 / 不可用
YuNet ONNX  ← 备选（误检率极低，召回稍弱，有 landmarks）
        ↓ 失败 / 不可用
OpenCV DNN 低阈值二重兜底（fallback_dnn_conf=0.12，召回小脸）
        ↓ 失败 / 不可用
Haar Cascade  ← 最后保底（OpenCV 内置，无需额外模型）
```

> 注：v4 初期（2026-04）曾是 OpenCV DNN 首选，05-17 起改为 YOLO 首选，`prefer_opencv_dnn=False`。

### 眼睛验证（双重保险）

每个检测器跑完后，对每个检测框额外跑眼睛 Cascade：
- **严格模式**：必须检测到眼睛才算有效人脸
- **普通模式**：大脸（> 最小尺寸的 4 倍）无需眼睛；小脸必须眼睛验证
- 若过滤掉超过 70%，自动回退到原始结果（防止误过滤）

### 大图专用管线（big_face_detector.py）

针对超高分辨率照片（如 4000×3000px）优化：
- 多尺度检测：`scale = target_h / orig_h`，目标把人脸缩放到 200-700px
- scale 计算改用 `target_h / orig_h`（非 `orig_h / target_h`），修复漏检问题

---

## 六、编码器架构

### 优先级链路

```
ArcFace ONNX 512维  ← 优先（models/arcface/ 下任一 .onnx 均可）
        ↓ 不可用
SFace ONNX 128维    ← 备选（models/face_recognition_sface_2021dec.onnx）
        ↓ 不可用
灰度直方图 32维     ← 最后保底（颜色直方图，计算快但区分力有限）
```

### ArcFace 对齐策略

- **有 YuNet landmarks 时**：使用 5 个关键点做相似变换对齐（`estimateAffinePartial2D`），再 resize 到 112×112
- **无 landmarks 时**：从 bbox 裁剪直接 resize 到 112×112
- **预处理**：归一化到 [-1, 1]（减 mean / 除 std）

### SFace 对齐策略

- 使用 OpenCV 内置 `FaceRecognizerSF.alignCrop()` 对齐裁剪
- 若 YuNet landmarks 可用，自动 alignCrop 到 112×112

---

## 七、数据库设计

### face_db.db — 人脸库

```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    encoding TEXT NOT NULL,          -- JSON 序列化 List[float]（512d / 128d / 32d）
    photo_path TEXT NOT NULL,
    location TEXT,                    -- (top, right, bottom, left)
    bbox TEXT,                        -- (x, y, w, h)
    added_at TEXT NOT NULL           -- ISO 时间戳
);

CREATE TABLE false_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    encoding TEXT NOT NULL,
    photo_path TEXT NOT NULL,
    location TEXT,
    bbox TEXT,
    marked_at TEXT NOT NULL
);
CREATE INDEX idx_false_faces_photo ON false_faces(photo_path);
```

### photo_index.db — 照片索引

```sql
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    md5 TEXT,                         -- 用于 MD5 去重
    faces TEXT,                       -- JSON List[Dict]（所有检测到的人脸）
    persons TEXT DEFAULT '[]',        -- JSON List[str]（已标注人物）
    indexed_at TEXT NOT NULL
);
CREATE INDEX idx_photos_md5 ON photos(md5);
```

**关键设计决策：**
- 所有复杂类型（encoding、location、bbox）存为 JSON 字符串，保持灵活性
- `false_faces` 表在导入时自动过滤已知假人脸，无需修改其他表
- WAL 模式 + `busy_timeout=5000` → 并发读写安全
- 支持从 JSON 迁移到 SQLite（自动检测，跳过已迁移文件）

---

## 八、核心模块 API

### FaceDetector（core/face_detector.py）

```python
FaceDetector(
    yunet_threshold=0.5,       # 置信度阈值（越小越严格）
    yunet_nms_threshold=0.3,   # NMS 合并阈值
    yunet_min_face=60,         # 最小人脸尺寸 px
    eye_validation=True,        # 眼睛验证
    strict_mode=False,          # 严格模式：必须检测到眼睛
    prefer_opencv_dnn=True,    # 优先使用 OpenCV DNN
    opencv_dnn_conf=0.5,        # OpenCV DNN 置信度阈值
)

detector.detect_faces(image_path) -> List[Dict]
# Dict: { location, encoding, image_path, bbox }
# encoding 自动选择最高可用编码器（ArcFace 512d > SFace 128d > 直方图）
```

**检测流程（v4）：**
1. OpenCV DNN ResNet-SSD 检测 → 有结果则停止
2. 否则 YuNet 检测 → 有结果则停止（同时记录 landmarks）
3. 否则 YOLO 检测 → 有结果则停止
4. 否则 Haar Cascade → 最后保底
5. 眼睛验证（双重保险）
6. 提取人脸特征向量（ArcFace > SFace > 直方图）

### FaceMatcher（core/face_detector.py）

```python
FaceMatcher(face_db_path, match_threshold=0.60)

matcher.find_match(encoding)               # -> (name, distance)
matcher.add_person(name, encoding, ...)     # 添加人员
matcher.add_false_face(encoding, ...)       # 纳入黑名单
matcher.remove_encoding(id)                 # 删除编码
matcher.remove_photo_encodings(path)        # 删除照片所有编码
matcher.update_person_name(old, new)       # 重命名人员
matcher.delete_person(name)                 # 删除人员
matcher.re_encode_all(detector)             # 用新编码模型重编码全部人脸
matcher.re_encode_false_faces(detector)     # 重编码假人脸
matcher.get_duplicate_encodings(threshold)  # 检测低维编码重复
matcher.export_db() / import_db(data)       # 导入导出
matcher.reset_db()                           # 清空
```

**距离计算：** `face_distance = 1 - cosine_similarity`（余弦距离，0=完全相同）

**性能说明（2026-10-18）：** `find_match` 已改走 `core/face_index.py` 的 numpy 矩阵索引（3500ms → 1.6ms/次，语义不变）；`encodings.person_id` 建有索引，按人查询勿再全表扫。新增写数据库的代码路径必须按 HANDOVER 坑9 维护索引/快照缓存。

### PhotoManager（core/face_detector.py）

```python
PhotoManager(index_path, output_dir)

photo_manager.add_photo(path, faces, persons, md5)      # 写入索引
photo_manager.check_md5_exists(md5)                     # -> path 或 None（去重）
photo_manager.get_photos_by_person(name)                # -> List[Dict]
photo_manager.get_missing_photos()                      # 文件不存在但有记录
photo_manager.get_duplicate_photos()                    # MD5 重复组
photo_manager.re_encode_all_photos(detector)            # 重编码照片所有人脸
photo_manager.remove_photo(path)                        # 删除索引
photo_manager.update_photo_persons(path, persons)       # 更新人物列表
photo_manager.remove_face_from_photo(path, bbox)        # 删除单个人脸
photo_manager.clear_photo_faces(path)                  # 清空所有检测结果
photo_manager.reset_db()                                # 清空
```

---

## 九、Streamlit 架构

### shared.py — 共享机制

```python
# 缓存装饰器（参数变化时 Streamlit 自动刷新）
@st.cache_resource
def get_detector(yunet_threshold, yunet_nms, yunet_min_face, eye_validation, strict_mode):
    return FaceDetector(...)

@st.cache_resource
def get_matcher(match_threshold):
    return FaceMatcher(...)

@st.cache_resource
def get_photo_manager():
    return PhotoManager(...)

# 侧边栏（每个页面调用一次）
render_sidebar() -> (detector, matcher, photo_manager, params)

# 页面配置
page_config(title, icon)
```

**设计决策：**
- 参数变化时自动刷新缓存（Streamlit `@st.cache_resource` 自动处理）
- 侧边栏返回组件实例，避免每个页面独立初始化
- `params` dict 包含所有当前参数，供页面内使用

### 参数映射设计

```
用户输入（1-10 整数滑块）→ 实际阈值
灵敏度 1  →  0.70  (极严格)
灵敏度 5  →  0.50  (默认)
灵敏度 10 →  0.30  (极宽松)
```

用户感知简单（整数滑块），内部映射到合理阈值。

### Sidebar 布局（v4）

分两大区域：
1. **🔍 检测器（定位人脸）**：灵敏度/NMS/最小人脸/眼睛验证/严格模式
2. **🎭 编码器（提取特征向量）**：ArcFace 512d / SFace 128d 状态

---

## 十、9 Tab 功能矩阵

| Tab | 页面 | 核心逻辑 | 关键交互 |
|-----|------|---------|---------|
| 1 | 📁 导入照片 | `detect_faces` + `check_md5_exists` + 批量检测 | 进度条、预览、假脸自动过滤 |
| 2 | 👤 人脸标注 | `find_match` + `add_person` + `add_false_face` | Selectbox 下拉 + 批量确认按钮 |
| 3 | 📂 分类浏览 | `get_person_photos` + 重新识别 | 重新识别重置逻辑 |
| 4 | 📋 照片索引 | `get_all_photos` + `st.data_editor` | 多选删除、筛选 |
| 5 | 💾 导出管理 | `export_db` + `import_db` + 编码升级 | 下载 + 重扫描 |
| 6 | 🔍 重复管理 | `get_duplicate_photos` + `get_duplicate_encodings` | 阈值滑块 |
| 7 | ⚠️ 缺失文件 | `get_missing_photos` | 批量删除 |
| 8 | 🚫 假脸管理 | `get_false_faces` + `remove_false_face` | 网格布局 |
| 9 | 🔬 模型对比 | 4 检测器并行运行 + 结果可视化 | 分标签页对比、汇总表 |

---

## 十一、UI/UX 设计亮点

### 批量确认机制（Tab 2）

人脸标注是高频操作，设计了**批量确认**流程：

```
检测人脸 → 预填姓名（自动匹配）
        → 用户修正（改姓名/新增/排除）
        → 全部处理完 → 点击"批量确认"一次写入
```

这比逐张确认体验好得多。

### 检测器状态感知（Sidebar）

Sidebar 实时显示：
- 各检测器可用性（✅ 首选 / ✅ 备选 / ❌ 缺少模型）
- 各编码器维度（ArcFace 512d / SFace 128d）
- 已标注人物列表（带照片数量）
- 当前数据库记录数

### 编码升级（Sidebar）

一键重编码功能，将旧编码（低维直方图）升级为新编码（ArcFace 512d / SFace 128d），覆盖人脸库 + 假人脸库 + 照片索引三个数据库。

---

## 十二、模型文件来源

| 模型 | 路径 | 来源 |
|------|------|------|
| OpenCV DNN ResNet-SSD | `models/opencv_dnn_face/` | OpenCV 官方 GitHub（caffe 格式） |
| YuNet | `models/face_detection_yunet_2023mar.onnx` | OpenCV Zoo（233KB） |
| YOLOv8n-Face | `models/yolov8n-face/face_yolov8n.pt` | Bingsu/adetailer（HuggingFace） |
| RetinaFace | `models/retinaface/retinaface_mnet025_v2.onnx` | InsightFace Model Zoo |
| ArcFace | `models/arcface/w600k_mbf.onnx` | InsightFace（buffalo_l 套件） |
| SFace | `models/face_recognition_sface_2021dec.onnx` | OpenCV Zoo（~2.7MB） |

**YOLO 下载方案（国内镜像）：**
```python
# 1. 查询文件列表
GET https://huggingface.co/api/models/{user}/{repo}/tree/main
# 2. 直接下载（不走 git lfs）
GET https://huggingface.co/{user}/{repo}/resolve/main/{filename}
# 3. 目标: Bingsu/adetailer / face_yolov8n.pt
```

---

## 十三、JSON → SQLite 迁移机制

自动检测 JSON 文件是否存在，若存在且 SQLite 不存在，则迁移：

```python
# 迁移时重命名原文件，避免重复迁移
os.rename("face_db.json", "face_db.json.migrated")
# 损坏则 rename 为 .broken 并打印日志
```

迁移覆盖：persons / encodings / photos / false_faces 四张表。

---

## 十四、创建类似项目的模板清单

### 阶段一：技术验证（1-2天）
- [ ] 选型：Streamlit + SQLite + 单核心模块
- [ ] 数据模型：画出 ER 图
- [ ] 核心功能 Demo：只验证最关键的 1 个功能（此处是"检测到人脸"）
- [ ] 评估编码方案：颜色直方图够用吗？还是需要 ArcFace？

### 阶段二：基础功能（2-3天）
- [ ] 目录结构：`core/` + `pages/` + `models/`
- [ ] 数据库初始化 + 迁移逻辑
- [ ] Streamlit 多页面框架（shared.py）
- [ ] Tab 1 + Tab 2 完整实现
- [ ] 侧边栏参数控制

### 阶段三：周边功能（1-2天）
- [ ] Tab 3-9（索引、导出、去重、假脸管理、模型对比）
- [ ] 数据导出/导入
- [ ] README 文档

### 阶段四：打磨（1天）
- [ ] 参数默认值调优
- [ ] 错误处理（文件不存在、图片损坏）
- [ ] 进度条用户体验
- [ ] 模型下载自动化

### 常见陷阱

| 陷阱 | 解决方案 |
|------|---------|
| 检测器初始化慢 | 延迟加载（YOLO 按需加载） |
| 图片方向 EXIF 问题 | 用 Pillow `.convert('RGB')` 统一处理 |
| 中文路径问题 | 全程用 `pathlib.Path` 或 `os.path` |
| Streamlit 缓存导致参数不变 | `@st.cache_resource` + 参数变化时自动刷新 |
| 大文件夹扫描卡界面 | 分批处理 + 进度条 |
| JSON 数据损坏 | WAL 模式 + 自动备份 `.broken` |
| OpenCV DNN 不兼容 ONNX | 改用 ONNX Runtime 替代 OpenCV DNN |

---

## 十五、未来扩展方向

1. **更强编码：** 接入 ArcFace ONNX，显著提升识别准确率（✅ 已实现）
2. **CLI 模式：** 不启动 Web，用命令行批量处理
3. **增量扫描：** 监听文件夹变化，自动处理新照片
4. **多设备同步：** 人脸库加密后同步到其他设备
5. **相似照片分组：** 不按人脸，按照片内容相似度
6. **移动端适配：** Streamlit 移动端布局优化

---

*文档生成：2026-04-18 | 最后更新：2026-04-19 | 项目版本：face-photo-sort v4*
