# 移交说明 — face-photo-sort 项目

> 接手时**先读这份文档**，再读 `ARCHITECTURE.md` 和 `README.md`。
> 这份文档聚焦"接手人最容易踩坑的地方"，ARCHITECTURE 讲"项目怎么搭起来的"。

**项目路径：** `D:\CADD\workspace\qclaw\projects\face-photo-sort\`
**Python 环境：** `D:\CADD\workspace\qclaw\tools\miniconda3\python.exe`（3.13.12，**项目内依赖装在 conda 主环境**）
**启动命令：**
```powershell
Set-Location "D:\CADD\workspace\qclaw\projects\face-photo-sort"
& "D:\CADD\workspace\qclaw\tools\miniconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
```

---

## 一、当前状态（2026-10-18）

| 维度 | 状态 |
|------|------|
| 文档 | ✅ README.md / ARCHITECTURE.md / requirements.txt / run_streamlit.bat |
| 核心代码 | ✅ `core/`：`face_detector.py`（v4）、`big_face_detector.py`、`opencv_dnn_detector.py`、`retinaface_detector.py`、`shared.py`、`face_index.py`（向量索引，2026-10-18 新增）、`image_utils.py`（图片 draft 降采样 + 缓存，2026-10-18 新增）、`name_search.py`（姓名拼音筛选，2026-10-18 新增）、`name_combobox.py`（单框姓名 combobox，2026-10-19 新增） |
| Streamlit 页面 | ✅ `pages/`：1导入 / 2标注 / 3浏览 / 4索引 / 5导出 / 6重复 / **7缺失（2026-10-18 补回）** / 8假脸 / 9模型对比 |
| 模型（首选） | ✅ `opencv_dnn_face/`（OpenCV DNN ResNet-SSD，~15MB，现为 YOLO 失败后的兜底） |
| 模型（备选） | ✅ `face_detection_yunet_2023mar.onnx`（YuNet）+ `yolov8n-face/`（YOLOv8n，.pt 优先，**现为首选检测器**） |
| 编码器 | ✅ `arcface/w600k_mbf.onnx`（优先）+ `face_recognition_sface_2021dec.onnx`（备选） |
| 人脸库 | ✅ `databases/face_db.db`（41MB，43人/3213条512维编码/77假脸） |
| 照片索引 | ✅ `databases/photo_index.db`（55MB，8867张照片元数据 + 人脸位置） |
| 测试图 | ✅ `data/`（4张：10004c0.jpg、DSCN9691.JPG、!SCN9444.JPG、!SCN9454.JPG） |
| **巨型模型（可选）** | ✅ **2026-10-18 已清理**：删除 `.deepface/`（580MB vgg_face）+ `arcface/` 冗余变体（1k3d68/det_10g/w600k_r50，~334MB）。`models/` 从 976MB 降到 103MB |
| databases/ 目录 | ✅ **活跃数据库目录**——两个 .db 都在这里（**不在 `models/`**，旧文档写错过）；cruft（.empty/.db.db）已于 10-18 清理 |
| output/、upload/ | ✅ 运行时产物目录，按需清理 |

**未打包说明：** 历史上 `models/.deepface/` 和 `models/arcface/` 冗余变体加起来 ~890MB **没有打入 zip**，且已于 2026-10-18 直接删除。如确需 `w600k_r50.onnx` 等变体，按 `utils/download_arcface.py` 里的来源重新下载即可，不影响运行。

---

## 二、接手人需要做的 4 件事

### 1. 装依赖（5 分钟）

```powershell
& "D:\CADD\workspace\qclaw\tools\miniconda3\python.exe" -m pip install -r "D:\CADD\workspace\qclaw\projects\face-photo-sort\requirements.txt"
```

依赖列表在 `requirements.txt`：
- `streamlit >= 1.28.0`
- `opencv-python-headless >= 4.8.0`（**注意：headless 版，不带 GUI 组件**）
- `Pillow >= 10.0.0`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `ultralytics >= 8.0.0`（YOLOv8 依赖）
- `onnxruntime >= 1.16.0`

**已知坑（2026-04-30）：** `sentence-transformers`、`scikit-learn` 等库在 Python 3.13 装不上。**本项目没用这些**，可以放心用 3.13。

### 2. 准备模型文件（10 分钟）

如果接手的是**新机器**，模型文件不在 `models/` 下，按 `ARCHITECTURE.md` 第五章顺序下载：

| 优先级 | 模型 | 大小 | 路径 | 来源 |
|--------|------|------|------|------|
| 1 (首选) | OpenCV DNN ResNet-SSD | ~15MB | `models/opencv_dnn_face/` | opencv_extra + opencv_3rdparty github |
| 2 (备选) | YuNet ONNX | ~350KB | `models/face_detection_yunet_2023mar.onnx` | OpenCV Zoo |
| 3 (备选) | YOLOv8n-Face .pt | ~6MB | `models/yolov8n-face/face_yolov8n.pt` | HuggingFace `Bingsu/adetailer`（**注意：不是 `iqbalamienho/YOLOV8n-face`，那个 401**） |
| 4 (兜底) | Haar Cascade | 内置 | cv2.data | OpenCV 自带，无需下载 |
| 编码器1 (优先) | ArcFace w600k_mbf | ~92MB | `models/arcface/w600k_mbf.onnx` | insightface 模型集 |
| 编码器2 (备选) | SFace | ~37MB | `models/face_recognition_sface_2021dec.onnx` | OpenCV Zoo |

**注意：** 缺模型时代码不会崩，只是会降级到下一个检测器。`render_sidebar()` 会显示当前哪些模型可用。

### 3. 启动并验证（2 分钟）

```powershell
Set-Location "D:\CADD\workspace\qclaw\projects\face-photo-sort"
& "D:\CADD\workspace\qclaw\tools\miniconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
```

打开 http://localhost:8501 ，左侧 sidebar 应该看到：
- "YOLOv8n: ✅ 首选"
- "OpenCV DNN: ✅ 兜底"
- "YuNet: ✅ 备选"
- "Haar: ✅ 始终可用"
- "ArcFace: ✅ 512维（推荐）"
- "SFace: ✅ 128维（备用）"

> ⚠️ 旧版本文档写的"OpenCV DNN: ✅ 首选"是 4 月的状态，5 月已改为 YOLO 首选（见第四节 2026-05-17 条目）。

**如果 sidebar 报"缺少模型文件"**，按第 2 步补齐。

### 4. 接手 db 文件（可选）

- `databases/face_db.db` — 已标注的人物库（不要随便删，删了所有标注就没了）
- `databases/photo_index.db` — 已索引的照片清单（删了下次要重扫所有照片）
- `databases/*.db-shm` / `*.db-wal` — SQLite 活动文件，app 运行时自动管理，**不要手动删**

接手时**直接复用原 db 即可**，SQLite 文件可跨机器拷。
**安全规则：代码里一律用 `shared.py` 的 `.json` 路径（内部自动转 `.db`），不要直接传 `.db` 路径给 `FaceMatcher`/`PhotoManager`——原因见坑8。**

---

## 三、代码最容易踩坑的地方

### 坑1：Unicode 路径 + OpenCV

`cv2.imread()` 在路径含中文/emoji 时会**静默失败**返回 None。本项目所有 `imread` 都用 workaround：
```python
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
```
**新加读取图片的代码务必用这个模式**，不要直接 `cv2.imread(path)`。

### 坑2：Streamlit `pages/` 排序

页面文件名带数字前缀（`1_📁_导入照片.py`）按数字排；如果新加页面，要**用数字前缀 + emoji + 中文名**。
注意文件名编码：emoji 在 PowerShell 里看着正常，但复制粘贴可能乱码，建议用文件管理器改。

### 坑3：`@st.cache_resource` 参数必须 hashable

`get_detector()` 缓存了 FaceDetector 实例。**改 detector 参数时要 streamlit 缓存生效**，需要把 `match_threshold` 这种 float 也当 hash key。**不要传 dict/list 进 cached 函数**。
当前 `get_detector()` 已经传入所有基础参数（threshold/nms/min_face/...），不要随便改签名。

### 坑4：检测器优先级别乱改（2026-05-17 起已变）

`FaceDetector.detect_faces()` 当前链路是 **YOLOv8n → OpenCV DNN → YuNet → DNN低阈值兜底 → Haar**：
- **YOLOv8n 首选**（多张小脸/侧脸最鲁棒，速度最快；`yolo_conf` 默认 0.5）
- **OpenCV DNN 兜底**（YOLO 无结果时用，`opencv_dnn_conf` 默认 0.12）
- **YuNet** 进一步兜底（误检率极低）
- **DNN 低阈值二重兜底**（`fallback_dnn_conf=0.12`，召回更多小脸）
- **Haar** 最后保底（会过拟合背景）

`prefer_opencv_dnn=False` 在 `shared.py` 已写死；4 月的"OpenCV DNN 首选"方案已废弃。
改顺序前先在 `data/` 的 4 张测试图 + `tests/benchmark_*.py` 上跑一遍确认。

### 坑5：miniconda 环境切换

项目用的是**用户级 miniconda**（`D:\CADD\workspace\qclaw\tools\miniconda3\`，不是系统的 Python）。
- 安装包用 `& "D:\CADD\workspace\qclaw\tools\miniconda3\python.exe" -m pip install <pkg>`
- 不要用 `pip install`（会用系统 Python）

### 坑6：Windows 编码

PowerShell 写文件容易引号嵌套出错。如果需要写复杂的多行 Python，**用 `write` 工具写 .py 文件，再 exec 跑**，别在 `python -c "..."` 里塞多行代码。

### 坑7：照片路径中的空格

测试照片路径 `D:\photoes\Nikon相机照片\比特恢复的文件\00 找到的文件   (9173 个文件)\...` 里有 3 个连续空格。
- `os.walk` 没问题
- `Pathlib` 没问题
- **PowerShell 命令行直接传会丢空格**，建议写脚本处理

### 坑8：不要把 `.db` 路径直接传给 FaceMatcher / PhotoManager（会毁数据！）

`FaceMatcher(face_db_path)` / `PhotoManager(index_path)` 内部会用传入路径调 `_maybe_migrate_from_json()`：
- 传 `.json` 路径（`shared.py` 的做法）：正常，仅当 json 存在才迁移
- **传 `.db` 路径：迁移逻辑把 .db 当 JSON 解析 → 失败 → 把真库改名成 `.broken` → 再建一个空库**（2026-10-18 实测踩中，`photo_index.db` 55MB 被改名，靠改名恢复）

**2026-10-18 已修复**：两个 `__init__` 增加了 `endswith('.json')` guard，只有旧版 json 路径才走迁移。
**你仍然要遵守**：新代码一律用 `shared.py` 的 `FACE_DB_PATH` / `PHOTO_INDEX_PATH`（.json 形式），别图省事直接传 .db。

### 坑9：新增写操作必须同步维护两套缓存（2026-10-18 性能优化引入）

标注页提速依赖两套缓存，新增任何写数据库的代码路径时必须配套维护，否则用户会看到过期数据：

1. **编码索引**（`core/face_index.py`，进程内 numpy 矩阵）：
   - INSERT 类（加人/加假脸）→ 调 `note_encoding_added()` / `note_false_encoding_added()`（写入穿透，不重建）
   - 原地 UPDATE / DELETE 类（改名/删除/重编码/重置/导入）→ 调 `invalidate_face_index()`（下次查询重建）
2. **Streamlit 快照**（`core/shared.py`，`st.cache_data`）：任何页面写操作后调 `invalidate_snapshot_caches()`（人物列表 + 未标注照片列表）

漏做的后果：匹配结果/人物下拉/未标注列表不刷新，直到进程重启。

### 坑10：改 `core/` 里类的方法后，必须重启服务（热重载不够！）

`shared.py` 的 `get_matcher()` / `get_photo_manager()` / `get_detector()` 都是 `@st.cache_resource`——**实例在首次导入时创建并跨 rerun 缓存**。Streamlit 热重载会重新 import 模块生成新类，但缓存资源返回的仍是**旧类的旧实例**，新加的方法不在它身上。

2026-10-18 实际踩到：给 `PhotoManager` 加 `get_photo_count()` 后热重载，页面报
`AttributeError: 'PhotoManager' object has no attribute 'get_photo_count'`。

**规则：改了 `core/face_detector.py` 里任何类的公开方法（新增/改签名），重启 Streamlit 进程；** 只改页面文件或函数体内部逻辑才可以靠热重载。重启命令见第二节启动命令。

### 坑11：碰照片像素的代码必须先问 EXIF（2026-10-19 EXIF 摆正改造）

相机横持拍的照片（本库 46 张，EXIF `Orientation=8`）在文件里是**躺着**的。任何"打开照片 → 检测/编码/裁剪/画框"的代码路径：

- 检测/编码一律走 `image_utils.open_upright()` 拿摆正图，bbox 用 `map_box_to_upright/original` 两个方向映射（库里存的坐标永远是**原始像素空间**，与旧数据兼容）；
- 显示裁剪走 `extract_face_image()`（内部已按 orientation 旋转输出）；
- 直接 `Image.open(path).convert('RGB')` 开图做 face 相关处理 = 制造侧躺编码/侧躺缩略图，是 2026-10-19 之前一整代 bug 的根因。

`draw_faces_on_image`（总览画框）是唯一例外：它画在原始像素空间的照片上，框与躺着的照片对齐，**不要**旋转（旋转了框就偏）。

### 坑12：`FaceDetector` 拿不到 `photo_index_path` = 手动旋转被无声忽略（2026-10-20）

用户手动转过的角度存在 `photos.manual_rotate`，`detect_faces()` 靠 `self.photo_index_path` 现读现用。**任何会写回人脸的脚本/页面都必须传这个路径**（`FaceDetector(photo_index_path=...)`），否则：重检测按"算法自己的方向"来，人工纠正被抹掉，而库里 `manual_rotate` 还是旧值——预览转、检测不转，最典型的静默不一致。

已传的：`core/shared.py: get_detector()`、`tools/batch_repair.py`、`tools/repair_faces.py`、`tools/rewrite_sideways.py`。新写批量脚本时照抄。

---

## 四、最近的改动记录

### 2026-11-11 08:50：**全 9 页体检**：慢页提速 + 扫描/匹配准确度修复

用户："每个页面再检查一遍，看看还有没有可以优化速度的，或者提高扫描准确度的？"

**实测（Streamlit AppTest，fresh / 交互 rerun / F5 新建会话，单位 ms）**

| 页面 | 修前 rerun | 修后 rerun | 修前 F5 | 修后 F5 |
|----|----|----|----|----|
| 3 分类浏览 | 2761 | **519** | 2591 | **439** |
| 4 照片索引 | 5534 | **619** | 8812 | **551** |
| 6 重复管理 | 2175 | **104** | 2896 | **102** |
| 7 缺失文件 | 706 | **94** | 636 | **102** |
| 8 假脸管理 | 10323 | **209** | 9785 | **220** |
| 1/2/5/9 | 已快 | 不变 | 已快 | 不变 |

冷进程首开：页 4 6443→4049ms，页 8 10461→1132ms。9 页 AppTest 全 `exceptions=0`。

**速度：把每轮 rerun 的全表扫描收进 `st.cache_data` 快照**
- `core/shared.py` 新增一族快照：`get_all_photos_cached` / `get_duplicate_photos_cached` / `get_missing_photos_cached` / `get_false_faces_cached` / `get_photos_by_person_cached` / `get_stranger_count_cached` / `get_rotate_log_cached` / `get_index_export_json`；`invalidate_snapshot_caches()` 统一 clear 全部（原来只清 2 个），并顺手同步失效清理的 DB 签名，避免写库后下一轮白扫 8867 个路径。
  实测单次成本：`get_all_photos` 2030ms、`get_duplicate_photos` 2146ms、`get_photos_by_person` 1579ms、`get_missing_photos` 438ms、`json.dumps(indent=2)` 2769ms/84MB。进缓存后只剩一次 pickle 拷贝（26MB≈320ms，其余<10ms）。
- 页 4：`get_all_photos()` 走快照；「文件状态=全部」时不再对 8867 行做 `os.path.exists`（页首清理已保证剩下的都在）；90MB 的 JSON 备份改成点「准备 JSON 导出」才生成（旧实现每轮都 dumps 一遍）；旋转记录走快照。
- 页 8：215 张假脸原来每轮全部解码一遍（cProfile 9.9s / 页面 10.5s，且图片 LRU 只有 16 条必然抖动）→ 改成每页 24 张分页；列表走快照；`core/image_utils.py` `_CACHE_SIZE` 16→32（页 3 一次渲染 20 张）。
- 页 3：陌生人计数 + 按人物照片列表走快照；页 6：重复照片扫描走快照；页 7：`get_missing_photos()` 改为「先只取 path 判存在，再仅对缺失行解析 JSON」（缺失为 0 时省掉 17734 次 json.loads）。
- 页 9：4 个模型加载函数加 `@st.cache_resource`（YOLO .pt + 2 个 ONNX session + Caffe net 原来每轮 rerun 重载）。

**准确度 / 正确性**
1. **假脸黑名单阈值解耦**（最严重）：`find_match` 和 `_is_false_face` 原来直接用 `match_threshold`（默认 0.60）——那是"是不是同一个人"的宽松阈值，用它判"是不是同一张误检"会把同人的正常照片全过滤掉。新增 `FALSE_FACE_THRESHOLD = 0.30`；`_is_false_face` 改走编码索引里已归一化的假脸矩阵（numpy 点积），批量导入时从「215 次 json.loads/脸」降到 1 次点积。
2. **`persons LIKE '%"name"%'` → `json_each` 精确匹配**：LIKE 对 ASCII 不区分大小写且 `%`/`_` 是通配符。实测 `LIKE '%"anne"%'` 会误命中 **2250 张 "Anne" 的照片**（导出时把别人的照片拷进自己目录）；`json_each` 精确查 "anne" = 0 张。加 `json_valid()` 守卫防脏数据。陌生人计数同步改为按照片去重（一张照片两个「陌生人」不再算 2 张）。
3. **重复编码检测原本恒为空**：`get_duplicate_encodings()` 只保留 `len(enc) < 128` 的"32维直方图"，而全库 3664 条都是 512 维 ArcFace → 功能等于没开。改为分块 numpy 矩阵 + 归一化余弦距离（与全应用 0.6 阈值同语义），实测 0.40 阈值下 1.4s 检出 500 对（上限）。
4. **导入页藏脸**：页 1 只要有一张脸自动匹配上就写 `persons`，而 `get_unlabeled_photos()` 只捞 persons 为空的照片 → 同照片其余没匹配的脸永久进不了标注队列。实测全库 **271 张照片 / 408 张脸**被藏住。改为「全部脸都匹配上才写 persons，否则写空」（已匹配的脸已在 face_db，标注页会预填，确认时按 bbox 去重不重复入库）。历史数据修复用 `tools/repair_hidden_faces.py`（默认 dry-run，`--apply` 自动备份后清空这些照片的 persons）。
5. **`add_photo` UPDATE 分支忽略 persons**（旧标注在重新扫描/导入后还挂着；显式 `[]` 也清不掉）→ 改 `persons=COALESCE(?,persons)`：省略=保留、`[]`=清空、名单=覆盖。`import_index()` 补回 `md5` / `manual_rotate`（原来导入后 MD5 去重失效、人工摆正被抹掉）。页 4 的 JSON 导出补上 `{"photos": ...}` 外壳，与 `import_index()` 对齐（原来是裸 dict，导入会得到 0 张）。
6. **坑9 缓存失效补齐**：页 1 导入后、页 5 三个写路径（导入人脸库/导入索引/重新扫描）、页 6/7 删除后，全部补 `invalidate_snapshot_caches()`；删除照片时同步 `matcher.remove_photo_encodings(path)`，否则 face_db 里留着已删照片的编码继续污染 `find_match`。
7. 页 5 重新扫描改 `os.walk`（原来单文件夹 `os.listdir` 漏子目录）+ 过滤已知假人脸，且不再传 `persons=[]`（避免新增的 COALESCE 语义抹掉人名）。
8. 页 9 加 `ImageOps.exif_transpose()`（原来直接 `convert('RGB')`，横持拍照被当躺图分析，对比结论与生产管线对不上）+ RetinaFace anchor 越界保护；页 8 撤销说明改为事实（撤销只移黑名单，脸不会自动回队列）；页 3 人物「张」数按 `photo_path` 去重（`get_all_persons()["photos"]` 是每编码一条）。

**改动文件**：`core/shared.py`、`core/face_detector.py`、`core/face_index.py`、`core/image_utils.py`、pages 1/3/4/5/6/7/8/9、新增 `tests/test_scan_accuracy.py`、`tools/repair_hidden_faces.py`。
**验证**：4 个测试套件全过（test_deskew / test_manual_rotate / test_missing_cleanup / test_scan_accuracy）；9 页 AppTest 0 异常；两库 `integrity_check ok`，数据未变（8867 照片 / 3664 编码 / 215 假脸）；重启 8501 后 health + 10 个页面全 200。改了 `core/`，**必须重启进程**（坑10）。

> **仍待决定 / 未做**：① 271 张历史藏脸的数据修复（`--apply` 会让这 271 张重新进标注队列，属数据变更，等用户点头）；② `re_encode_all` / `re_encode_false_faces` 仍不套用 `roll` + `manual_rotate`（重编码横躺照片会拿到侧躺裁剪），`force_rotated` 只找 EXIF≠1，漏了手动转过的照片；③ 页 4 的 `st.data_editor` 一次推 8867 行到浏览器（首开仍有 ~4s），要彻底解决得做 SQL 分页；④ 页 8 的 false_faces 表不存 roll，旋转照片的假脸缩略图仍是侧躺的；⑤ 页 9 的「最佳模型」按检出脸最多推荐，会奖励误检；⑥ 冷启动 4.6s 里的 face_index 建索引 1.7s，可加磁盘缓存但写路径多、风险高。

### 2026-11-11 07:45：**标注页刷新又变慢** → 失效记录清理改成进程级（用户报障）

**现象**：人脸标注页每次 F5 刷新都要等 ~3s（进程刚启动第一次打开 ~7s），页面内交互反而只要 ~0.4s。

**实测定位**（Streamlit AppTest，同一台机器）：

| 场景 | 修前 | 修后 |
|----|----|----|
| 冷进程首次打开 | 7278ms | **4628ms** |
| F5 / 新开会话（缓存热） | **2889ms** | **405ms** |
| 页内交互 rerun | ~410ms | ~410ms |

差分实测：把 `clean_missing_photos()` 换成 no-op，新会话从 2938ms → 403ms，即 **~2.5s 全部花在它身上**；cProfile 显示 17,734 次 `json.loads` + 8,873 次 `os.path.exists`。

**根因**：页面用 `st.session_state["_missing_cleaned"]` 做「每会话清理一次」，但 **F5 会新建 session**，于是每次刷新都重跑 `clean_missing_photos()` → `pm.get_all_photos()` 把 8867 行的 `faces`/`persons` JSON 全解析一遍（~2s）+ 8867 次文件路径检查。这个坑 2026-10-18 那轮性能优化没覆盖到（当时只改了 sidebar / 快照 / 图片解码）。

**改法**：
1. `PhotoManager.get_all_photo_paths()`（`core/face_detector.py`）：只 `SELECT path FROM photos`，不碰 JSON；顺带加 `db_path` 属性。
2. `clean_missing_photos()`（`core/shared.py`）：改用上面的路径查询；去重从 `st.session_state` 提到**进程级**——按 `(DB mtime_ns, size)` 签名 + 60s TTL 跳过重复扫描，DB 被外部脚本改过会自动重查；新增 `force=True` 给「全部清理」按钮强制执行；删了记录后补调 `invalidate_snapshot_caches()`（旧实现漏了，会留过期快照）。
3. 页面 2 / 3 去掉 `_missing_cleaned` 会话守卫（去重已在 shared 里）；页面 7 的「💥 全部清理」改 `force=True`；页面 4 保持「打开时自动清理」语义不变。

**改动文件**：`core/face_detector.py`、`core/shared.py`、`pages/2_👤_人脸标注.py`、`pages/3_📂_分类浏览.py`、`pages/7_⚠️_缺失文件.py`；新增回归测试 `tests/test_missing_cleanup.py`（4 项：路径查询 / 真实清理 / 进程级去重 / force 强制）。

**验证**：新测试 4/4 通过；`test_deskew.py`、`test_manual_rotate.py` 全过；8 个页面 AppTest 均 `exceptions=0`；已重启 8501（health 200、首页及页面 2/3/4/7 均 200）。改了 `core/`，**必须重启进程**才生效（坑10）。

> 遗留：冷启动仍有 4.6s，其中 `face_index` 首次建索引约 1.7s、模型导入约 1.1s、页面 4 无条件 `get_all_photos()` 全表解析 ~2s（那页本就要列全库，属预期）。如需进一步压冷启动，可给编码矩阵加磁盘缓存（按 DB 签名失效），但要同步维护所有写路径，风险高于收益，暂不做。

### 2026-10-21 00:45：**照片旋转只允许 90° 的整数倍**（用户报障 101f480.jpg）

**现象**：101f480.jpg（1600×1200，EXIF=1）在页面里被转了 −15.4°，预览成了带黑角的菱形；用户要求"旋转必须是 90 的整数倍"。

**根因**：`detect_faces` 里有一级"姿态摆正（deskew）"——量出双眼连线角，把**整张照片**反向旋转随便多少度再检测，并把 `roll = 手动 + 这个任意角` 存进库。预览/裁剪照着 `roll` 转，于是出现非 90° 的画面。全库有 17 张照片、29 张脸带这种非 90 的 roll（−23°~+49.8°）。附带问题：整图转完重新检测时可能掉到 DNN 兜底，捡到假框（101f480 的框落在天空上）。

**改法**：把"照片方向"和"人脸对齐"彻底拆开——
1. **整图旋转只做方向**：EXIF / 手动（±90、180）/ 四向探索（90、270）。存库的 `roll` 永远是 90 的整数倍，预览不会再出现黑角。
2. **脸上的倾斜改在编码时按脸消化**（`_encode_face_aligned` / `_upright_face_crop`）：带 35% 边距裁出这张脸 → YuNet 量双眼角 → 转正 → 再量，最多 4 轮 → `alignCrop` 5 点对齐 → ArcFace。**只动这张脸的 112×112 输入，照片本身不动、roll 也不记录**。
3. **正脸保持原编码路径**：只有 `|双眼角| ≥ 12°`（`deskew_threshold`）的脸才走对齐；不歪的脸仍走 `bbox 直接缩放` / YuNet `alignCrop`，8k 张老照片的编码口径不变（实测同人距离 0.160 → 0.183/0.168/0.165，仍 < 0.6）。
4. `encode_region()` 重编码也走同一条路径，`--force-rotated` 重编码结果与检测一致。

**实测**（同一个人 5C3A4598 旋转后自比）：歪 15° 对齐 0.183 / 不对齐 0.321；25° 0.168 / 0.551；45° 0.165 / 0.918。即"不转照片也能认回本人"。

**数据修正**：17 张非 90 roll 的照片全部重检测写回（14 张未标注走 `batch_repair`，3 张已标注走 `rewrite_sideways` + `sync_labeled_encodings --from-file`，0 个名字丢失）。顺带修正：1abb00 3→6 张真人脸（原来把侧躺当歪 15°）、d32c40 5→2（原来 5 个里 3 个是招牌/木头）、e7e900 1→4、f1a600 1→2；101f480 从 0 张（框在天空）变成 1 张（男孩，正确落框）。库：8867 张照片 / 5074 张脸，roll 非 90 的数量 **0**。

**遗留**：101f480 / 2ece40 这类"人物整体歪 ~45°"的照片，歪的不是 90°，按新规则照片保持原样；人脸靠对齐仍能认出（101f480 的小孩因过小未被 YOLO 检出）。需要正着看只能人工处理。

### 2026-10-20 23:45：手动旋转按钮（↺/↻ 90°，可连点）+ 库里留"手动转过的标记"

**需求**：照片旁边加旋转按钮，点一次转 90° 并按转后的照片重新搜索人脸，可多次点到摆正；库里要有手动旋转标记，方便集中排查错误原因。

**交互**
- 标注页（页面 2）：每张照片缩略图下方一行 `↺ 左转90° / ↻ 右转90° / ⟲ 重置`，下面一行显示 `🧭 手动 右转 90°（manual_rotate=-90）`。
- 分类浏览页（页面 3）：每张已标注照片的缩略图下方同样的三个图标按钮（`compact`）。
- 照片索引页（页面 4）：表格勾选后批量「↺ 选中左转90°/↻ 选中右转90°」（含"零人脸"的照片，那些进不了任何队列）；新增「手动旋转」列 + 「按手动旋转筛选」+ 「🧭 手动旋转记录」表（可导 CSV）。
- 转过之后**自动四向探索关闭**（`allow_cardinal=False` 由 `manual_rotate != 0` 自动决定）：方向由用户拍板，不会出现"点 4 次被算法转回原地"。

**数据模型**（`photo_index.db`，`PhotoManager.__init__` 里幂等迁移）
- `photos.manual_rotate INTEGER NOT NULL DEFAULT 0` —— **逆时针为正**、(-180,180]，与 `roll` 同向同单位；
- `photos.manual_rotate_at TEXT` —— 最后一次点按钮的时间；
- `manual_rotate_log(id, path, from_deg, to_deg, faces_before, faces_after, names_before, names_after, status, note, created_at)` —— 每次点击一行，`status ∈ ok / no_face / names_lost / too_many_faces / encoding_partial / error`。**排查直接查这张表**。
- 脸上的 `roll` 含义不变（"还要逆时针转多少度才正"），但现在 = `manual_rotate + 自动判断角`，所以 `extract_face_image` / `upright_photo_view` 不用改。

**检测顺序**（`detect_faces`）：手动旋转 → EXIF → deskew/四向。三者用仿射矩阵链复合（`_compose_affine`），最后一次性映射回**原始像素坐标**存库；`map_box_to_original` 现在用 EXIF 尺寸（不是手动旋转后的尺寸），否则横持照片会宽高互换。

**已标注照片的安全网**（`core/rotation.py`）
- 人名按"旧框→新框中心最近 + 一对一贪心"搬到新框，容差 0.75×对角线；搬不动的名字记 `names_lost`，页面弹警告让用户补标；
- 编码同步：删掉该照片在 `face_db.encodings` 的行再按新框/新向量写回（975a80 那类"索引改了、识别库没改"的复发点，这里一次做掉）；
- 检出 > `max_faces`（默认 20）视为误检簇，**不落库**，只记日志 `too_many_faces`。

**回归**：`tests/test_manual_rotate.py`（6 组）——归一化、手动角叠在自动角之上（同向结果必须完全一致、反向距离 0.00→0.41）、bbox 映射面积比 1.01、手动后不再自动探索、已标注往返（名字保住+编码重写+重置还原）、日志与待复查列表。`tests/test_deskew.py` 仍全过。

**已知遗留**：手动转错方向不会自动报错（YOLO 对旋转不敏感，歪着也能检出）——靠界面预览判断；转过但一张脸都没检出的照片会进标注页顶部「🧭 手动旋转待复查」板块，不会消失。

### 2026-10-20 23:15：101bcc0.jpg → 闸门漏掉"YOLO 全漏 + YuNet 给竖框"这一类，并砍掉 180° 探索

**用户报障**：101bcc0.jpg 要旋转（1600×1200、EXIF=1、整张画面躺倒、两个人站着）。

**为什么之前没转**：侧躺时 YOLO 0 张脸、DNN 0 张脸，YuNet 在草地上给了一个 **62×68 的竖框**（分 0.82）。
旧闸门是 `backend not in ("yolo","yunet")`——YuNet 命中了所以不算可疑；另一条看"框是横的"
（`w >= 1.05h`）——这个框是竖的，也不触发。于是整张照片被判成 1 张脸、没有 roll。

**代码改动（`core/face_detector.py`）**：
1. 闸门改成 `probe is None or backend != "yolo" or (低分 and 横框)`——**首选检测器 YOLO 缺席本身就是方向不可信的证据**，不必再看框形。
2. `_try_cardinal` **不再试 180°**。这是本轮最重要的发现：倒着的脸 YuNet 照样给 0.93~0.95，
   而 `_probe_face` 会把"左右眼标签互换"当成 180° 归一化，`|roll|<45` 因此拦不住 180°。
   实测 86d400（七人正立合影）、10f7440、fe87c0 三张正立照片被转成倒立。
   倒立照片极罕见，转错一张废一张 → 只留 90/270，180 交给人工。
3. `tests/test_deskew.py` 新增 `test_yunet_junk_box`（101bcc0 必须出 2 张脸、roll=+90）。

**全库扫描（新工具 `tools/find_yolo_miss_sideways.py`）**：索引里"有脸但没 roll"的 **2993 张**先只跑 YOLO 快筛
（约 4 分钟）→ YOLO 漏检 **197 张** → 完整检测确认：115 张结果变了，其中**带 roll 27 张**
（17 未标注 / 10 已标注），另 88 张只是框数变化（多是 YuNet 在没人的照片上瞎报，没写）。

**写回**：
- 未标注 17 张 → `batch_repair.py --max-faces 8`：全部重写（d4bb80 有 8 个框，含 4 个背景游客误检，留给你 X）；
- 已标注 2f1c40（许剑华）数量 1→1，`rewrite_sideways.py` 搬名成功；d55a40 旧框对不上新框，跳过；
- `sync_labeled_encodings.py --all-rolled --apply` 同步了 2f1c40 的识别编码；
- 全库现在 **517 张照片 / 705 张脸**带 roll（此前 498/671）。

**仍待人工（已标注、新旧脸数不一致，不敢自动搬名）**：2f0a40(1→3)、563040(1→2)、563640(1→2)、
72f440(1→2)、b69640(1→14)、d55a40(对不上框)、10f7440（本来就正立，只是歪 12.7°，没动）。

### 2026-10-20 22:15：975a80.jpg 复核 → 发现"已标注照片的识别编码没跟着重写"（重要缺口）

**用户报障**：975a80.jpg 需要旋转（2592×1944、EXIF=1、画面躺倒、已标「高振雯」）。

**复核结论**：检测早已正确——`detect_faces` 返回 1 张脸 `bbox=[359,875,258,191]`、`roll=-90`，照片索引（`photo_index.db`）里也是摆正后的数据，`upright_photo_view` + `draw_faces_on_image` 预览正、框贴在脸上。真正的 bug 在**识别库**。

**根因**：`tools/rewrite_sideways.py` 只写照片索引（`pm.clear_photo_faces` / `add_photo` / `update_photo_persons`），**没有碰 `face_db.db` 的 `encodings`**。于是 42 张被重写的已标注照片：
- 索引里：新 bbox + roll + 摆正后重新算的 encoding；
- 识别库里：仍是"躺着"时的旧 bbox 和旧向量 → `find_match` 继续走歪向量，人脸距离虚高。

975a80 实测：旧 `bbox=[372,813,274,276]`、旧编码对高振雯其余 32 条编码最近距离 0.68（>0.60 阈值，匹配不上）；同步后 `bbox=[359,875,258,191]`、最近 0.042。

**修复**：新工具 `tools/sync_labeled_encodings.py`（默认 dry-run，`--apply` 才写）：
- `--path-like <片段>` 修单张；`--all-rolled` 扫全库所有 roll≠0 的已标注照片；
- 护栏：索引 faces 数 == persons 数且全非空；编码行数与人脸数一致；新旧框中心距离 < 0.6×旧框对角线；**且该旧框必须离"分给它的新框"最近**（防串脸），任一不满足整张跳过；
- 按人名删旧行 → 用索引里的新 location/bbox/encoding 重写；顺带补上"有脸有人名但编码行缺失"的（953240.jpg）。

**结果**：40 张已标注带 roll 照片里，8 张本就同步、**32 张重写**（含 975a80），编码总数 3653 → 3654，无跳过、无重复行。抽样 18 张逐条对比旧/新编码对同一人图库的距离：**全部变小**（如 2f4c00 许诺 0.935→0.745、597b40 顾雨煦 0.721→0.497、f32940 陈欣怡 0.601→0.344），无一退化。

**仍需留意**：`tools/check_rolled_encodings.py` 列出 18 个"自己人最近距离 >0.6 或别人更近"的已标注脸（如 7faec0 沈旭 0.676、184880 高振霆 0.730）——这是标注样本少/长相相近的**既有**问题，不是本次引入，需要时用标注页人工复核。

**改动文件**：`tools/sync_labeled_encodings.py`、`tools/check_rolled_encodings.py`（新增）；`databases/face_db.db`（写库前备份在 `data/_backup/face_db.20261020_221223.db`）。改完已重启 8501 服务（face_index 是进程内缓存，不重启不生效）。

### 2026-10-20（晚）：批量清理"躺着"的照片 + 预览摆正（用户报障 7faec0.jpg）

**背景**：7faec0.jpg 同样是 EXIF 干净但画面躺倒（2048×1536 横构图里两个人侧躺）。它早在 4 月就被 YOLO 在侧躺像素上检出、被标成「陌生人」——两个框一个裁到鼻梁、一个裁到耳朵，是典型的"躺着也能检出、但检出的是废框"。

**根因**：四向探索的闸门只认"框是横的（w ≥ 1.05h）"或"主检测器没命中"，而 7faec0 的旧框恰好方方正正（147×158、73×93），YuNet 裁剪分 0.72/0.75 又刚好过阈值 → 两道闸门都放过了它。

**代码改动（`core/face_detector.py`）**：
1. 闸门新增 **probe is None 也算可疑**——侧躺的脸上 YuNet 常常一个关键点都给不出（b18c40.jpg：YOLO 检出、YuNet 无点、框方正，之前漏判）；
2. `_try_cardinal` 判据换成 **裁剪单脸分**（不再用整图分）：整图缩 640 时小脸直接漏检，实测 7d2940 正确方向整图 0.0、错误方向反而 0.868，整图分完全不可用；
3. 探**最大的前两张脸**取最高分：组照里最大那张可能是侧脸/被挡（75f800 正确方向最大脸 probe=None）；
4. 阈值 `max(0.86, 原方向分+0.08)`，同时要求残余角 |roll| < 45°（倒 180° 的脸 YuNet 也给 0.868，但关键点是歪的）。

**显示摆正（`core/image_utils.py` + 页面）**：新增 `upright_photo_view(path, max_width, roll)` → (PIL 图, bbox 映射器)。`draw_faces_on_image` 改用它（标注页"检测到的人脸"预览不再躺着），分类浏览页网格也改用它。**缩略图正、整图躺着**的割裂感消除。

**批量清理**（新工具 `tools/sideways_scan.py`、`tools/batch_repair.py`、`tools/rewrite_sideways.py`）：
- 扫描 8810 张（`比特恢复` 文件夹）：判据 = 整图 YuNet 四方向取最高分，转正后 ≥ max(0.86, 原方向+0.08) → 候选 **631 张**（7%），其中 549 未标注、82 已标注；
- 未标注 549：并行重检测写回 **532 张**（另 8 张无脸、2 张无变化、7 张检出 >6 框跳过待人工）；
- 补跑"仍无 roll"的 131 张 → 再修 42 张；
- 已标注 82：`rewrite_sideways.py` 按"框中心最近"把旧人名搬到新框（数量必须一致，否则跳过）→ **42 张重写、0 张丢名字**，33 张因新旧脸数不一致跳过、3 张对不上框、1 张超 6 框。
- 结果：全库 **498 张照片 / 671 张脸**带 roll 标记（-90: 368，+90: 273）。

**用户同名注意**：7faec0 后来被用户标注为「沈旭 / 李朝霞」，而该照片正是这两人的编码来源之一——`find_match` 返回距离 0.0 是自匹配，不是 bug。

**验证**：`tests/test_deskew.py` 新增 `test_upright_view`（侧躺照片预览必须是竖构图）；deskew / EXIF / 陌生人 / 姓名下拉四套回归全过；抽查 20 张批量结果人脸缩略图全部正立；服务已重启，三页 200。

### 2026-10-20：四向探索 — 无 EXIF 的侧躺/倒立照片（用户报障 7d2940.jpg）

**现象**：用户说"7d2940.jpg 应该旋转 90° 来提取人脸"。该照片 EXIF `Orientation=1`（没有旋转标记），但画面整个躺倒（老照片翻拍）。库里那条框 (1490,1039,139,145) 是躺在画面里的侧脸，编码谁都匹配不上。

**为什么之前的 deskew 顶不住**：
1. YOLO 对旋转极耐，躺着的脸照样检出 → "检出 0 张脸"这种信号根本不会出现；
2. `estimate_roll` 在侧脸上仍能拿到 5 个关键点，而且会**"假装水平"**——7d2940 在 0° 上量出 -5.7°，低于 12° 阈值，于是什么也没做。
   四个方向的 YuNet 置信度才是真信号：**0°=0.618 / 90°=0.851 / 180°=0.750 / 270°=0.929**，270°（=顺时针 90°，正是用户说的方向）最高。

**A. 探针升级：`_probe_face()` 返回 (roll, YuNet置信度)**
- 正常正脸实测 0.91~0.95（抽 12 张，min 0.912）；侧躺掉到 0.6 左右 → 置信度比 roll 更可靠。
- `_measure_roll()` 保留（只要角度），内部转调 `_probe_face()`。

**B. 四向探索 `_try_cardinal(bgr, base_score)`**
- 把照片分别转 90/180/270，各跑一遍 `_detect_chain` + `_probe_face`；
- 候选必须：关键点能立住（|roll|<45°）且置信度 ≥ max(0.88, base+0.08)；
- 选置信度最高的方向，`roll` 存"要转多少度才正"（270→-90），bbox 经仿射映射回原图坐标。

**C. 触发闸门（关键，踩过坑）**
```python
sideways_box = 盒宽 >= 盒高 * 1.05        # 正立人脸框永远竖着
suspicious = backend not in ("yolo","yunet") or (置信度<0.85 and sideways_box)
```
- 只用置信度会误伤：67de00.jpg 正常竖拍，被 180° 方向的高分候选骗到，整张转翻。
- 加了"横框"闸门后：67de00 / 64ebc0 不再被动；抽查 25 张正常照片只触发 1 次，且那 1 张（563040）确实是侧躺。
- `backend` 也进了判据：`_detect_chain()` 现在返回 `(faces, infos, backend)`。563040 明明是侧躺，YOLO 一张都没检出，掉到 OpenCV DNN 兜底后报出 **46 个屏幕里的小脸**——这种"兜底命中"本身就是方向不对的证据。

**D. 写回（已执行）**
- 7d2940 → Anne(0.302)，roll=-90；979e80 → Anne；9b0c40 → 3 张（Anne/华培新/许剑华），全部 roll=-90。
- 同批查出的另外 4 张**没写**：8e4fc0 / ae9280 检出的是花纹/漩涡（误检），efac80 第 3 张是花屏，d7a800 只部分修正——留在队列里等人工 X。
- 全库按"脸框横着"筛出 120 张可疑，其中 114 张**已标注**（改框会让人名贴错脸），只处理了 6 张未标注。

**测试**：`tests/test_deskew.py` 新增 `test_cardinal_sideways`（坚持 roll=-90 + 能认出人）与 `test_upright_not_flipped`（67de00 不许被翻转）。

**坑**：`_detect_chain()` 返回值从 2 元组变 3 元组，所有调用点（detect_faces / _try_cardinal / 各处重检测）都必须解包。

### 2026-10-19（深夜 2）：姿态摆正 deskew + 46 张横持照片写回执行完毕

**用户需求**："跑 46 张横持照片的写回，加上自动纠偏"。

**A. 自动纠偏（deskew）——`core/face_detector.py` + `core/image_utils.py`**

流程图：`detect_faces` → `_detect_chain()`（原检测逻辑原样搬进来）→ `estimate_roll()`（YuNet 5 点量双眼连线角）→ `|roll| ≥ 12°` 才 `rotate_expand(bgr, roll)` 整图反向旋转 → 再跑一次 `_detect_chain` → 在**转正图**上编码 → bbox 经 `map_box_through_affine` + `map_box_to_original` 映射回原始像素空间。

- 新增 `image_utils.rotate_expand(arr, deg)`（扩画布旋转，返回 M）与 `map_box_through_affine(bbox, M)`（4 角点过仿射取 AABB）。
- 新增 `FaceDetector.estimate_roll(arr, bbox)`：带 35% 边距裁脸 → **长边缩放到 640** → YuNet 取 5 点 → 两眼 atan2。量不到返回 0（宁可不转）。
- 新增 `FaceDetector.encode_region(bgr, bbox, roll)`：重编码路径专用，roll≠0 时先转正再检测取紧框编码。
- 参数：`FaceDetector(deskew=True, deskew_threshold=12.0)`；阈值依据是 tilt_experiment2 的实测（15° 起开始认错人）。
- 存库的脸多一个可选字段 `"roll"`（只在真转过的脸上出现）；`extract_face_image(..., roll=)` 会把缩略图转正；标注页已传 `face.get("roll", 0.0)`。
- **符号约定（坑）**：`roll` 正 = 画面里脸顺时针歪；纠正就是 `rotate_expand(bgr, roll)`（cv2 正角=逆时针）。写反过一次，表现为越转越歪、距离 0.92 全不匹配。
- `re_encode_all_photos` 改用 `encode_region`（带 roll），重编码不会把已摆正的脸再"掰"回歪的。

**实测效果**（5C3A4598，把照片旋转 +25° 后检测）：

| | 距离 | 匹配 |
|---|---|---|
| 关 deskew | 0.551 | Lili（勉强） |
| 开 deskew | **0.055** | Lili |

**性能**：单张 +58ms（0.66s→0.72s，约 +9%）。两个优化必须保留：
- `cv2.FaceDetectorYN.create()` 每次要重载 onnx（~300ms）→ 实例缓存（`_yunet_det`，阈值变了才重建）；
- 量 roll 的裁剪**不要**直接喂原尺寸（1400px 的裁剪要 350ms）→ 统一长边 640。

**B. 46 张横持照片写回（已实际执行，先备份了数据库）**

- 备份：`data/_backup/photo_index_235456.db`、`face_db_235456.db`。
- `tools/repair_faces.py --from-file data/_rotated_paths.txt` → **42 张重检测写回**（0 张无脸、0 张护栏拦截），4 张已标注跳过。
- 4 张已标注的走强制重编码：`re_encode_all(force_rotated)` 重编 4 条 encodings、`re_encode_all_photos(force_rotated)` 重编 64 张索引脸、`re_encode_false_faces` 重编 7 条。
- 校验：4654 → Lili(0.26)/Anne(0.42)（改造前 inf/inf）；4656 → 干净 2 张（原先 11 个凤冠误检）；4652/4653 组照 4 张全部对上本人；640dc0 四口之家 4 张真人（无误检，已目检 contact2.png）。
- 库统计：零人脸照片 5863 → 5825，1 张脸 2118 → 2146。
- 服务已重启（坑10），三个页面 200。

**新增测试**：`tests/test_deskew.py`（符号约定 / 歪 25° 能认回本人且明显优于关掉 / 正常照片不带 roll / roll 裁剪不出错）。`test_exif_upright`、`test_stranger`、`test_name_combobox` 回归全过。

**顺手修的坑**：`tools/repair_faces.py` 新增 `--from-file`——46 条路径里有含空格的 Nikon 目录，用 shell 展开会被拆成多个参数，全部"路径不存在"。

### 2026-10-19：倾斜角实测（回答"要不要先旋转再识别"）

**新增两个只读实验脚本**（不改库）：
- `tools/tilt_experiment.py`：整图旋转 0/10/20/30/45/60/75/90°，只测 YOLO 置信度。
  结论：**8 张清晰正脸照片在 0~90° 全部检出、conf 只从 0.84 掉到 0.77**——YOLO 对旋转极耐。
  注：`_detect_yolo()` 返回的是 4 元组 (x,y,w,h)，**丢掉了 conf**；要量置信度必须直接调 `det._yolo_model`（见脚本内 `yolo_conf()`）。
- `tools/tilt_experiment2.py`：整图旋转后走完整 `detect_faces` + `FaceMatcher.face_distance` / `find_match`。
  结论（6 张单真人已命名照片，阈值 0.60）：

  | 倾斜 | 平均距离 | 认对本人 | 检出 |
  |---|---|---|---|
  | 15° | 0.316 | 5/6 | 6/6 |
  | 30° | 0.583 | 3/6 | 6/6 |
  | 45° | 0.793 | 1/6 | 6/6 |
  | 60° | 0.913 | 0/6 | 6/6 |
  | 90° | 0.887 | 0/6 | 6/6 |

  **要点：检测几乎不受角度影响，识别从 15° 就开始错人**（15° 已有 1 张认错，30° 一半错，≥60° 全废）。
  原因：YOLO 有旋转增广（训练时见过歪脸），ArcFace 对齐依赖人脸姿态，roll 一大就把脸"掰"成另一张脸。

- `tools/deskew_scan.py [样本数] [--all]`：用 YuNet 双眼连线估计每张照片的人脸 roll 角，出分布。
  抽样 250 张（库里已有人脸的照片，随机种子固定）→ 检出 116 张：
  均值 4.4° / 中位 2.1° / P90 9.5°；≤5° 占 77.6%，≤10° 占 91.4%，>15° 只占 3.5%，>45° 2 张。
  倾斜最大的样本集中在 `Nikon相机照片\比特恢复的文件`（翻拍/老照片）。

**结论（对产品的影响）**：真正需要"先转正"的分两类——
1. **EXIF 横持**（46 张，orientation=8）：不转基本等于瞎，已由 EXIF 改造解决，**写回仍待用户确认**（见下条）。
2. **画面真倾斜**（约 3.5% 超过 15°）：检测不需要帮忙，但**认人会错**。若要救，用 YuNet 5 点算 roll → 反向旋转 → 再检测编码，而不是无脑试 4 个 90°（90° 在 EXIF 场景有用，对"拍歪"没用）。

### 2026-10-19（深夜，最重要）：EXIF 摆正统一改造 — 横持拍摄照片检测/编码/显示全部改正

**背景（用户报障 5C3A4654 "face recog is wrong"）**：排查发现该照片（及全库 46 张）EXIF `Orientation=8`——相机横持拍的照片在文件里是"躺着"的，而 `detect_faces` 直接 `Image.open` 不解 EXIF，导致整条管线在躺着的像素上跑：
- YOLO 对侧躺 90° 的脸置信度暴跌（4656 实测最高 0.187 < 0.5 阈值）→ 触发 DNN 兜底 → 11 个误检，真人脸全丢；
- ArcFace 从侧躺裁剪出的编码谁都匹配不上（4654 两张脸 `find_match` 双双 `dist=inf`）；
- 标注页脸缩略图也是躺着的。

**改动（`core/image_utils.py` + `core/face_detector.py` + `core/shared.py`）**：
1. `image_utils` 新增 EXIF 基础设施：`get_orientation()` / `open_upright()`（按 orientation transpose 摆正）/ `map_box_to_upright()` / `map_box_to_original()`。坐标公式经 PIL 真旋转逐 orientation 实测校验 + 1600 组随机框往返测试（`tests/test_exif_upright.py`）。
2. `detect_faces`：**在摆正图上检测和编码**，bbox 映射回原始像素坐标系存库（与旧数据、`extract_face_image` 落点完全兼容；正照片 orientation=1 走原路径零变化）。
3. `extract_face_image`：裁剪结果按 orientation 旋转——**显示出来的脸永远是正的**（标注页/假脸管理页自动生效）。
4. 三处重编码（`re_encode_all` / `re_encode_false_faces` / `re_encode_all_photos`）改为摆正图编码 + bbox 映射，新增 `force_rotated=True` 参数：横持照片的旧编码即使维度达标也强制重编码（旧编码是侧躺裁剪的，不重编码匹配永远错）。
5. 侧边栏「🔄 编码升级」区新增**「🔄 重编码横持照片」按钮**（一键 force 三库）。

**全库扫描结论**：8867 张中 46 张横持（43 张在 `2024春节古装照片`、3 张在 Nikon 恢复文件夹，全部 orientation=8）。

**修复效果实测**（未写库）：
- 4654：2 张脸 → Lili(0.255) / Anne(0.423)，摆正前是 `inf/inf`；
- 4656：11 个误检 → 干净 2 张（新娘 Lili 0.332），护栏不再拦截；
- 4627：找回高振雯(0.386)。
- 46 张 dry-run：42 张可修、4 张已标注跳过（走 force 重编码）、0 张触发护栏。

**待执行**（用户确认后）：`tools/repair_faces.py` 跑 46 张横持照片写回 + 侧边栏「重编码横持照片」+ 重启服务。

### 2026-10-19（深夜）：新增 `tools/repair_faces.py` — 4 月旧索引人脸错误统一修复工具

**背景（用户连续报障 5C3A4614/4627/4630 等）**：4 月建库用的 OpenCV DNN 老管线在部分照片上 (a) 把袍子/手/灯笼/凤冠误检成人脸入库，(b) 漏检真人脸。用户在标注页把误检框逐个 X 掉后走"全排除"分支（`clear_photo_faces`）——整张照片清空掉出标注队列，**漏检的真人脸永久丢失**。当前 YOLO 管线对这些照片基本正确。

**改动：新增 `tools/repair_faces.py`**（纯脚本，不动页面）：
- 用法：`python tools/repair_faces.py <照片或文件夹> [--dry-run] [--max-faces N] [--include-labeled]`
- 对每张 **persons 为空（未标注）** 的照片：`clear_photo_faces → detect_faces（当前管线）→ add_photo`（与标注页「重新匹配」同一路径；`add_photo` 对已存在行只更新 faces/md5/indexed_at，不动 persons）。
- **已标注照片默认跳过**——重检测会打乱"第几张脸对应哪个名字"的位置映射，人名会贴错脸。
- **`--max-faces` 护栏（默认 6）**：单张检出超限视为误检簇直接跳过并列入"待人工复核"。实例：5C3A4656（横持相机+凤冠）YOLO 最高置信度仅 0.187 全部低于 0.5 → 触发 DNN 兜底 → 11 个误检（凤冠首饰+石栏），真人脸（新郎 YuNet 可检、新娘仅 0.187 弱检）反被漏掉——这类照片宁可不动也不能把垃圾框写进队列。
- 路径经 Git Bash 传入会被转成正斜杠，`collect_paths` 用 `normpath` 规范化匹配库键。
- **执行后必须重启 streamlit 服务**：未标注列表是进程内快照缓存（`st.cache_data`），外部写库不自动刷新（坑10）。
- dry-run 实测（2024春节古装照片 57 张）：42 张可修（0→1~3 个真人框）、14 张已标注跳过、1 张（4656）触发护栏。**尚未实际执行，待用户确认。**

### 2026-10-19（深夜）：一键「陌生人」标记 + 修复队列移位导致的会话键串脸

**需求（用户原话）**：有的照片上所有人都是陌生人，加一个按钮一键把所有 face 标为陌生人（或默认把匹配失败的标为陌生人）。

**改动：**
1. **新增 `core/stranger.py`**：`STRANGER_NAME = "陌生人"` + `mark_photo_as_strangers(photo_manager, matcher, photo_path, faces, excluded)`。**关键设计：「陌生人」是照片级桶标签，不是人脸库里的"人"——只写 `photo_index.persons`（如 `["陌生人","陌生人"]`），不写 encodings。** 两个原因：(a) `find_match` 是全局最近邻，陌生脸汇成巨型簇会把已知人物错配成陌生人（预填和全库匹配都被污染）；(b) 陌生人日后可能要命名，而 `update_person_name` 会连同所有陌生人一起改名，无法安全拆分。已排除（X）的人脸仍走假脸流程，分支语义与「批量确认」逐行对齐。
2. **标注页**（`pages/2_👤_人脸标注.py`）：每张照片卡片加「👤 本张全是陌生人」按钮（在姓名区下方），点击后整张照片移出标注队列；`core/shared.py` 转发新符号。
3. **修复一个潜伏的串脸 bug（本次按钮会高频触发，顺手连「批量确认」的老路一起修）**：照片被确认/移出队列后，队列重排、后面照片的 `idx` 全部前移，而会话键还是旧下标——用户填给照片 B 的姓名会跑到照片 C 的框里，排除标记也会串。新增 `_remap_keys_after_removal(photos, removed_idxs)`：取出全部队列相关键（`sb_/cb_/rd_/ti_/q_/px_`，正则匹配，含历史残留）后按新队列位置写回；批量确认和陌生人按钮都改用它。
4. **分类浏览页**（`pages/3_📂_分类浏览.py`）：`PhotoManager.get_stranger_photo_count()` 统计后，把「陌生人」并入人物列表（快照字典先 `dict()` 拷贝，不污染进程级缓存）——这批照片现在可浏览、可用「批量重新识别」重置（误标后悔药）。注意：陌生人不在 face_db persons 表的新标记里，但**历史数据里有**（用户以前在 combobox 手输过"陌生人"，person id=27、215 条 encodings，见下）。

**已验证**：`tests/test_stranger.py`（`python tests/test_stranger.py`）——分支语义（全排除/部分排除/无排除/绝不写 encodings）/ `get_stranger_photo_count` 与直连 SQL 一致（当前 147 张）/ AppTest 点按钮：只写 persons、`add_person` 零调用、会话键按新队列重映射（`sb_1_0→sb_0_0`）。分类浏览页 AppTest 冒烟：无异常、陌生人 metric 147 张、选中后详情 147 张走通。`test_name_combobox.py` 回归全过。**改了 `core/`（新模块 + face_detector 方法），按坑10 必须重启服务**（已重启）。

**遗留/待定**：face_db 里历史「陌生人」挂着 215 条 encodings（旧的手输流程写的），仍在全局匹配池里——若担心已知人物被预填/全库匹配误标成陌生人，可清掉这 215 条（保留 persons 行，名字仍可下拉选）；本次新按钮不再追加编码。用户提的"默认把匹配失败的标为陌生人"未采纳： Silent mislabel（没来及命名的人会被悄悄标成陌生人），显式按钮更稳。

### 2026-10-19（晚）：批量栏移至页面底部 + 修复预填擦除用户输入的 bug

**需求（用户原话）**：批量确认和重新匹配的按钮放到页面下方，改完姓名顺手点，不用回页首。

**改动：**
1. `pages/2_👤_人脸标注.py`：「收集待确认 + 批量确认/重新匹配栏」整块（原第 155-287 行）从照片列表上方移到**照片列表之后**（页面最底部）。收集逻辑一并后移还有个好处：它在 combobox 渲染完之后才读 `sb_`，本轮刚点选的姓名当轮就计入统计（以前要等下一轮 rerun）。
2. **修复一个会废掉 combobox 的 bug**：预填块原来对"自动匹配没中"的人脸，每次 rerun 都会重新 `find_match` 并写 `cb_ = ""`——用户在姓名框里打的拼音首字母，下一次 rerun（回车/失焦/点页面任何地方）就被擦掉，建议列表永远展开不了。现在只要 `sb_` 键存在（哪怕是 `""`）就直接 `continue`，只对首次渲染/批量确认后的人脸跑自动匹配。这也顺手省掉了每轮对未匹配人脸重复 `find_match`。
3. **修复「唯一匹配即自动确认」吞掉新增入口（用户报障：姓名框输入 lzg 没有弹出新增框）**：`core/name_combobox.py` 的 `auto_select_name()` 原来在"只有一个匹配"时也自动确认——用户输 "lzg" 恰好唯一命中某人（如刘志高）时，回车/输完最后一个字母就直接确认并收起建议列表，「➕ 新增」根本来不及点。拼音首字母会撞车（lzg ≠ 刘志高），不能替用户做决定。**现在只有"原样等于某个已有姓名"才自动确认**；有匹配时 caption 也追加提示「要新增的人不在里面？点最下方➕ 新增」。

**已验证**：新增 `tests/test_name_combobox.py`（`python tests/test_name_combobox.py` 直接跑：单测 + AppTest 集成）——唯一匹配不自动确认 / 原样等于已有姓名才确认 / 有匹配时列表同时带匹配姓名和「➕ 新增」/ 无匹配输入（lzg）也能弹出新增 / 点新增后 `sb_` 当轮生效且底部批量统计更新，全部通过。批量栏移动 + 预填修复的验证：AppTest 全过——输入 "a" 后建议列表展开（`Anne/Nina/麦安晓/➕ 新增`）、点选后 `sb_0_0='Anne'`、底部统计当轮从「1 张已处理」变「2 张已处理」、批量栏渲染序号在最后一张照片卡片之后、页面无异常。**注意：本次改了 `core/name_combobox.py`，按坑10 必须重启服务**（已重启，浏览器刷新即可）；纯页面级改动才可热重载。

### 2026-10-19（凌晨）：标注页姓名框合并为单框 combobox（重要，取代当晚早些时候的双控件方案）

**需求（用户原话）**：不要在姓名框上方再加输入框，直接利用现有姓名框——激活后输入拼音首字母即过滤出匹配姓名列表，无匹配则可新增。

**改动：**
1. 新增 `core/name_combobox.py`：`render_name_combobox(prefix, person_names)` 把人脸行的「筛选框 + selectbox + 新姓名框」三控件合并为**一个输入框**。状态：`sb_{p}`=已确认姓名（可为新名字）、`cb_{p}`=框内文本、`rd_{p}`=radio 点选值。**列表展开条件是 `cb != sb`**（打字中即展开，确认即收起），不需要额外展开标志位。
   - 建议列表用 `st.radio` 渲染（最多 8 行 +「…还有 N 人」），选项含匹配姓名、「➕ 新增「x」」（把输入原样作为新姓名）、「✕ 清除选择」
   - **提交逻辑放在渲染开头统一消费 `rd_{p}`**（不依赖 on_change 回调）——原因见下方"AppTest 局限"
   - 回车/失焦（`on_change`）时：输入原样等于已有姓名 → 自动确认（`auto_select_name()`，纯函数可单测）。~~"只有一个匹配"也自动确认~~ **已移除**（2026-10-19 晚，用户报障）：首字母撞车时会替用户确认并收起列表，吞掉「➕ 新增」入口
2. `core/name_search.py` 保留不变（匹配逻辑复用）；`core/shared.py` 转导出 `render_name_combobox`。
3. `pages/2_👤_人脸标注.py`：预填块同时写 `sb_`/`cb_`；批量确认收集、状态提示、清理键（`sb_/cb_/ti_/q_/px_`）全部适配；删掉 `ti_`（新姓名框）和 `q_`（旧筛选框）整套逻辑。

**已验证**：AppTest 8 场景全过——已选姓名显示在框里且列表收起 / 首字母过滤 / radio 点选确认且框内更新 / 无匹配走新增行 / 清除行清空 / 批量确认统计正确（含预填单脸照片）/ `auto_select_name` 与 `resolve_pick` 单测；标注页 HTTP 200。

**AppTest 局限（重要，写测试时会再遇到）**：AppTest 的 `set_value()` **不触发任何 on_change 回调**（text_input 和 radio 都实测确认），但真实浏览器正常。所以凡是"提交"类逻辑不要只写在 on_change 里——放在渲染时读取 widget 状态统一处理，浏览器和 AppTest 两条路径行为一致。另外 AppTest 的 `at.session_state` 不支持 `.keys()`/`to_dict()`，只能用 `k in at.session_state` 探测；`at.radio` 会数到 sidebar 的 radio，判断具体控件要用 key。

### 2026-10-18（深夜）：标注页姓名拼音首字母筛选（双控件方案，已被上面的 combobox 取代）

**需求**：人名下拉框支持输入字母按拼音首字母匹配，无匹配时提示新增。

**改动：**
1. 新增 `core/name_search.py`：`filter_persons(names, query)`——拼音首字母包含 query（"zs"→张三）/ 全拼前缀（"lisi"→李四）/ 中文子串（"李"→李四）/ 英文子串（"an"→Anne），大小写不敏感；`name_keys()` 带进程内缓存。依赖 `pypinyin`（已加入 requirements.txt，**新环境需 pip install**）。
2. `pages/2_👤_人脸标注.py`：每个人脸行的下拉框上方加筛选输入框（`q_{idx}_{i}`）。行为细节：
   - 输入后按回车/失焦即过滤下方下拉框（Streamlit 核心组件无原生逐键筛选，这是零依赖方案）
   - 当前已选姓名若被过滤掉会**钉在选项首位**，防止 selectbox 值被重置
   - 无匹配时显示提示"未匹配「xxx」，可直接选择新增"
   - 选中姓名后 `on_change` 自动清空筛选框，下拉框恢复完整列表
   - "➕ 新增..." 选项始终在列表中，原新增流程（`ti_{idx}_{i}` 输入新姓名）不变
3. `core/shared.py` 转导出 `filter_persons`；页面 5 的死 import（`extract_face_image`）已清。

**已验证**：51 个真实姓名逐一用自身首字母/全拼均可搜到；AppTest 四场景（无匹配提示 / 钉住已选 / 首字母过滤 / 选中后清空并恢复完整列表）全部通过；标注页 HTTP 200。

### 2026-10-18（晚）：标注页性能优化（重要）

**问题**：人脸标注页每次交互（选个姓名）都全量重跑页面脚本，实测 **~13-15s/次**；多人照片更慢（每张脸都重新全图解码 + 每张脸一次全库匹配）。

**优化（按拆文件约定，新逻辑放 `core/face_index.py`，主文件只做委托）：**

| 项 | 前 | 后 |
|----|----|----|
| `find_match`（3213条编码全扫 + 逐条 json.loads + Python 循环） | 3500ms/次 | **1.6ms/次**（numpy 矩阵 + 预归一化，冷启动建索引 ~2.7s/进程） |
| `get_all_persons`（49 人各开一个连接 + 全表扫 encodings） | 2047ms | **180ms**（单条 JOIN 查询） |
| sidebar 照片计数（`get_all_photos` 全表解析 8867 行 faces JSON） | 4400ms | **50ms**（新增 `PhotoManager.get_photo_count()`，COUNT(*)） |
| 页面单次交互（AppTest 实测，3 张照片 7 人脸） | ~13-15s | **1.4s** |

**关键改动：**
1. 新增 `core/face_index.py`：`FaceEncodingIndex`（编码 numpy 矩阵 + 写入穿透 + 假脸过滤）。`FaceMatcher.find_match` 改为委托，返回语义完全不变（已逐条比对验证）。
2. 新增 `idx_encodings_person` 索引（`_ensure_indexes`，幂等）：按人查询不再全表扫 22MB 编码文本。
3. `shared.py` 新增快照缓存：`get_persons_snapshot()` / `get_unlabeled_photos_cached()` + `invalidate_snapshot_caches()`（版本号作缓存键，进程级共享，写操作后统一失效）。
4. sidebar / app.py / pages 2,3,4,7 全部改走快照和计数接口。
5. **缓存维护纪律见坑9**——新增写路径必须配套调 note_*/invalidate_*。

**已验证**：AppTest 无头跑标注页无异常；find_match 与旧算法逐条一致；写入穿透/假脸穿透/改名失效均通过；`integrity_check` ok。

### 2026-10-18（续）：图片解码优化（第 2 项，已完成）

**背景：** 优化第 1 项落地后，交互期剩余大头是图片解码——源照片 6000×4000，而标注页只需要 80px 脸缩略图和 300px 总览图，旧实现每取一张脸就全图解码一次（~350-500ms/次），N 人照片解码 N+1 次。

**改动：**
1. 新增 `core/image_utils.py`：`_load_scaled(path, target_w, target_h)`（lru_cache 16 条）用 PIL `draft()` 让 libjpeg 解码时直接按 1/2^k 档位缩放（**必须同时传宽和高**——档位对两个方向同时生效），返回已解码图 + scale。实测 6000×4000 MPO：全解码 ~350ms → 1500×1000 ~135ms（2.6 倍；熵解码仍读全部系数，收益是 2-4 倍而非像素比）。
2. `extract_face_image` / `draw_faces_on_image` 增加可选参数 `max_width=None`：**None 时行为与旧实现完全一致**（已用 `np.array_equal` 验证）。两种语义别混：
   - `extract_face_image(max_width=W)`：W 是**脸的目标宽度**，按脸占比反推照片解码尺寸，并对齐到 draft 档位（同照片相近大小的脸共享缓存条目）；脸比显示尺寸还小才全解码
   - `draw_faces_on_image(max_width=W)`：W 是**总览图的解码宽度**，高度同比
3. 调用点传参：页面 2 `max_width=600`（总览）/ `200`（脸）、页面 8 `200`；页面 5 的死 import（`extract_face_image` 只 import 未用）已清。

**效果（实测）：**
- 2048×1536 的 11 人照片：冷 115ms / 热 27ms（1 次总览解码 + 1 次脸解码共享）；脸输出与全分辨率裁剪**尺寸相同、像素差 0.00**
- 6000×4000 MPO 按脸尺寸：800px 脸 2.4 倍提速且输出正好 200×250（像素差 0.54）；1600/3000px 大脸 2.4-2.9 倍；100/200px 小脸走全解码直通（与旧实现同速，多解也没有更多细节）
- 已验证：新旧裁剪内容一致、画框坐标换算正确、默认参数输出字节级一致、两页面 AppTest 无异常

**⚠️ 本条目曾有一次错误实现（当天发现当天修，教训记录）：** 初版把 `max_width` 当成**整张照片**的解码宽度，导致 6000px 照片解成 160px 宽、脸只剩 15-30px，用户反馈"脸图很不清晰"。且当时用"旧图缩小后 vs 新图"做一致性测试，两个都是糊的所以没暴露。**教训：降采样类优化必须用"全分辨率原图"做基准比对，且要核对输出尺寸是否符合显示意图。** 另有两个附带修复：draft 请求不能传正方形（高度条件会把档位顶大，6000×4000 请求 1500×1501 会白解 3000×2000）；`load()` 后 fp 即为 None，`.copy()` 是白复制（72MB 原图复制一次 ~100ms）；超过 600 万像素的解码结果不进 LRU（防 72MB 缓存条目）。

### 2026-10-18：接手日清理 + 数据事故修复（重要）

**数据事故（已恢复）：** 测试时直接传 `.db` 路径给 `PhotoManager`，触发迁移逻辑把 `photo_index.db`（55MB/8867张）当 JSON 解析失败并改名 `.broken`。内容未损，已改名恢复 + `integrity_check` 验证通过。

**修复：** `core/face_detector.py` 两个 `__init__` 增加 `.json` guard（详见坑8）。

**清理：** 删除 `.deepface/`（580MB）+ `arcface/` 冗余变体（1k3d68/det_10g/w600k_r50，~334MB）+ `databases/` cruft（`.empty`×2、`.db.db`）。models/ 976MB → 103MB。

**补回页面 7：** `pages/7_⚠️_缺失文件.py`（缺失文件查看 + 勾选清理 + 全部清理，复用 `get_missing_photos()` / `clean_missing_photos()`）。

**文档修正：** db 位置 `models/` → `databases/`；检测器优先级更新为 YOLO 首选；补录 2026-05-17 未文档改动。

### 2026-05-17：YOLO 首选化 + 性能调优（补录，当时未写文档）

**改动：**
- 检测器优先级从 "OpenCV DNN 首选" 改为 **YOLOv8n 首选**（`prefer_opencv_dnn=False` 写死；多张小脸/侧脸更鲁棒且最快）
- 新增 `fallback_dnn_conf=0.12`：YOLO/DNN/YuNet 都失败时，用低阈值再跑一次 OpenCV DNN 召回小脸
- `opencv_dnn_conf` 默认值从 0.5 降为 0.12；sidebar 对应 slider 范围 0.05–0.5、默认 0.1
- 导出 YOLO ONNX 版（`face_yolov8n.onnx` / `yolov8n-face.onnx`，两文件内容相同，代码实际仍优先加载 `.pt`）
- `_to_db_path()` 的 `.db` 分支修复为原样返回（此前会产生 `.db.db` 双扩展名，`databases/photo_index.db.db` 就是那时的残留，已删）

**benchmark（`tests/`）：** 7 张测试图对比，pre-resize 640 方案最快（~100ms/张，大图从 ~190ms 降下来）；ONNX Runtime 方案（~188ms）慢于 ultralytics `.pt`，未采用。

### 2026-04-20：OpenCV DNN 修复（重要）

**问题：** 10004c0.jpg（4人群照）原本只检测到 1 张脸。

**根因：**
1. `opencv_dnn_conf=0.5` 太高，过滤掉了小脸（conf 只有 0.10-0.13）
2. `min_face_height=60` 过滤掉了实际只有 48px 的人脸
3. 自定义 `_nms(iou=0.4)` 按面积排序保留最大框，抑制了相邻人脸

**修复：** 改用 `cv2.dnn.NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold)`：
- 新增 `opencv_dnn_conf` 参数（默认 **0.1**，可在 sidebar 调）
- 新增 `opencv_dnn_min_face` 参数（默认 **30px**，独立于 YuNet）
- 用 cv2 内置 NMS 而不是自定义 `_nms`

**改动文件：**
- `core/face_detector.py` — `_detect_opencv_dnn()` 重写
- `core/shared.py` — `get_detector()` 新增 2 个参数 + sidebar 新增 2 个 slider

**测试图：** `data/10004c0.jpg` 修复后检测到 2 张脸（conf 0.91 + 0.115）。
注意：cv2.dnn.NMSBoxes 在 conf>=0.1 阈值下会返回约 20 个框，需要再加 min_face 过滤；目前实现是**先 NMS 再过滤**，不是先过滤再 NMS，可能还有改进空间。

### 2026-04-19：架构升级 v4

- `face_detector.py` 整合了检测器 + 匹配器 + 照片管理器（减少跨文件改动）
- 引入 OpenCV DNN ResNet-SSD 作为首选检测器
- 引入 ArcFace 512维作为优先编码器（SFace 128维备选）
- SQLite 替代 JSON（解决读写锁问题）

---

## 五、需要接手人决定的事

> 2026-10-18 接手日已决策的项标注 ✅；仍开放的标 ⬜。

1. **`.deepface/` 和 `arcface/` 巨型模型 (~890MB) 要不要保留？**
   - ✅ **已删**（2026-10-18）。运行时只用 `w600k_mbf.onnx`；如后续需要 r50 等变体，按 `utils/download_arcface.py` 重新下载。
   - 保留：能跑历史脚本（如果有）
   - 删除：节省磁盘

2. **`databases/` 里的 cruft（`.empty`×2、`.db.db`）要不要清理？**
   - ✅ **已清**（2026-10-18）。注意：`databases/` 本身不是空目录，两个正式 .db 都在这里。

3. **是否升级 Python 3.13 → 3.12？**
   - ✅ **保持 3.13**。本项目无 3.13 不兼容的依赖（实测 streamlit 1.56 / onnxruntime 1.24 / ultralytics 8.4 / opencv 4.13 全部正常）。

4. **是否继续在 v4 架构上迭代？**
   - ✅ **已定约定：拆文件**。`face_detector.py` 已 1490 行，不再继续往里塞：
     - 新检测器/新管线 → `core/` 下新文件（如 `core/detector_ensemble.py`，参考现有 `big_face_detector.py` / `retinaface_detector.py` 的独立文件模式）
     - 新人脸库/照片逻辑 → 同样拆新文件，经 `core/__init__.py` 导出
     - **不做大爆炸式重构**：现有 1490 行稳定运行中，重构风险大于收益；只约束增量
   - 页面逻辑一律放 `pages/`，不写进 core

5. **CV2 DNN NMS 还有优化空间**（见上"最近改动"最后一段），要不要做？
   - ⬜ 仍开放。检测器已换成 YOLO 首选，DNN 只做兜底，优先级下降；建议先做 A/B 测试（`data/10004c0.jpg` 上对比"先 NMS 再过滤" vs "先过滤再 NMS"）再决定。

---

## 六、应急联系方式 / 决策记录

接手过程中如果遇到**架构层面**（不是 bug）的决策冲突，按以下优先级：
1. `ARCHITECTURE.md` 的设计原则
2. `README.md` 的功能描述
3. 本 `HANDOVER.md` 第 3 节"最容易踩坑的地方"
4. 实在拿不准就保持现状 + 留 TODO

代码细节（变量命名、行数、写法风格）自由发挥，不强求一致。

---

**文档结束。** 读完这份 + ARCHITECTURE.md + README.md，应该能在 30 分钟内把项目跑起来。
