# pages/9_🔬_模型对比.py
"""
人脸识别模型对比工具
比较 OpenCV DNN / YuNet / YOLOv8n-Face / RetinaFace 四个检测器的效果
"""
import streamlit as st
import os
import sys
import time
import numpy as np
from PIL import Image, ImageOps
import cv2
import io

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.shared import page_config

# 页面配置
page_config("模型对比 - 人脸照片分类工具", "🔬")

st.title("🔬 人脸识别模型对比")
st.markdown("""
比较不同人脸检测器的效果。上传照片后，系统将同时运行 **4 个检测器**，
展示每个人脸的定位框、检测数量、置信度和处理耗时，帮助你选择最佳模型。
""")
st.markdown("---")

# ─────────────────────────────────────────────
# 检测器封装
# ─────────────────────────────────────────────

# 模型加载很贵（YOLO .pt + 2 个 ONNX session + Caffe net）；
# 原来每轮 rerun 都重新加载一次。cache_resource 跨 rerun 复用同一实例。
@st.cache_resource
def _load_yunet():
    """加载 YuNet"""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'face_detection_yunet_2023mar.onnx'
    )
    if not os.path.exists(model_path):
        return None, None, None
    img = Image.new('RGB', (640, 480))
    h, w = 480, 640
    detector = cv2.FaceDetectorYN.create(model_path, '', (w, h), score_threshold=0.3, nms_threshold=0.3, top_k=50)
    return detector, h, w

@st.cache_resource
def _load_yolo():
    """延迟加载 YOLOv8n-Face"""
    try:
        from ultralytics import YOLO
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'yolov8n-face', 'face_yolov8n.pt'
        )
        if not os.path.exists(model_path):
            return None
        return YOLO(model_path)
    except Exception:
        return None

@st.cache_resource
def _load_opencv_dnn():
    """加载 OpenCV DNN ResNet-SSD（优先 Caffe，失败则用 ONNX）"""
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'opencv_dnn_face'
    )
    prototxt = os.path.join(model_dir, 'deploy.prototxt')
    weights = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    if os.path.exists(prototxt) and os.path.exists(weights):
        try:
            return cv2.dnn.readNetFromCaffe(prototxt, weights), 'caffe'
        except Exception:
            pass
    onnx_path = os.path.join(model_dir, 'opencv_face_detector.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            return sess, 'onnx'
        except Exception:
            pass
    return None, None

@st.cache_resource
def _load_retinaface():
    """加载 RetinaFace ONNX（使用 ONNX Runtime）"""
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'retinaface'
    )
    candidates = [
        os.path.join(model_dir, 'retinaface_mnet025_v2.onnx'),
        os.path.join(model_dir, 'retinaface_mobilenet025_v2.onnx'),
        os.path.join(model_dir, 'retinaface_r50_v2.onnx'),
    ]
    import onnxruntime as ort
    for p in candidates:
        if os.path.exists(p):
            try:
                sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                return sess
            except Exception as e:
                st.warning(f"RetinaFace 模型加载失败: {os.path.basename(p)} — {e}")
                continue
    return None

# ─────────────────────────────────────────────
# 各检测器检测函数
# ─────────────────────────────────────────────

def detect_yunet(bgr_img, detector):
    """YuNet 检测"""
    h, w = bgr_img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr_img)
    results = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[-1]) if len(f) >= 7 else 0.5
            results.append({'bbox': (x, y, bw, bh), 'score': score})
    return results

def detect_yolo(bgr_img, model, conf=0.3):
    """YOLOv8n-Face 检测"""
    if model is None:
        return []
    results = model(bgr_img, verbose=False, conf=conf)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_score = float(box.conf[0])
            boxes.append({
                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                'score': conf_score
            })
    return boxes

def detect_opencv_dnn(bgr_img, net_info, conf=0.3):
    """OpenCV DNN ResNet-SSD 检测（支持 Caffe 和 ONNX）"""
    if net_info is None:
        return []
    net, mode = net_info
    h, w = bgr_img.shape[:2]
    results = []

    if mode == 'onnx':
        import onnxruntime as ort
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300),
                                      mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        input_name = net.get_inputs()[0].name
        outputs = net.run(None, {input_name: blob})
        detections = outputs[0]
        for det in detections[0, 0]:
            conf_score = float(det[2])
            if conf_score < conf:
                continue
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                continue
            results.append({'bbox': (x1, y1, bw, bh), 'score': conf_score})
    else:
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300),
                                      mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        for det in detections[0, 0]:
            conf_score = float(det[2])
            if conf_score < conf:
                continue
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                continue
            results.append({'bbox': (x1, y1, bw, bh), 'score': conf_score})

    # NMS
    if results:
        boxes_xyxy = [(r['bbox'][0], r['bbox'][1], r['bbox'][0]+r['bbox'][2], r['bbox'][1]+r['bbox'][3])
                      for r in results]
        scores = [r['score'] for r in results]
        keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores, conf, 0.4)
        if len(keep) > 0:
            keep_flat = [int(x) for x in keep.flatten()]
            results = [results[i] for i in keep_flat if i < len(results)]
    return results

def detect_retinaface(bgr_img, sess, conf=0.3):
    """
    RetinaFace ONNX 检测（ONNX Runtime）
    支持 InsightFace RetinaFace MobileNet0.25 的多尺度 FPN 输出格式。
    输出 9 个张量（stride 32/16/8 各 3 个：cls_prob, bbox_pred, landmark_pred）
    """
    if sess is None:
        return []
    h, w = bgr_img.shape[:2]

    # 预处理
    input_size = 640
    img_resized = cv2.resize(bgr_img, (input_size, input_size))
    # BGR -> RGB, HWC -> CHW, 归一化到 [-1, 1]
    blob = cv2.dnn.blobFromImage(img_resized, 1.0 / 128, (input_size, input_size),
                                  (127.5, 127.5, 127.5), swapRB=True)
    # blobFromImage 已经是 NCHW float32

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: blob})

    # InsightFace RetinaFace mnet025 输出 9 个张量，分 3 组 (stride 32/16/8):
    #   cls:   (1, 4, fh, fw)   → 2 anchors × 2 classes (bg/fg)
    #   bbox:  (1, 8, fh, fw)   → 2 anchors × 4 (dx,dy,dw,dh)
    #   lm:    (1, 20, fh, fw)  → 2 anchors × 10 (5 landmarks × xy)
    strides = [32, 16, 8]
    anchor_cfg = {
        32: [16, 16],
        16: [32, 32, 16, 16],
        8:  [32, 32, 16, 16, 8, 8, 4, 4],
    }

    all_dets = []
    sx, sy = w / input_size, h / input_size

    for stride_idx, stride in enumerate(strides):
        cls_out  = outputs[stride_idx * 3 + 0]
        bbox_out = outputs[stride_idx * 3 + 1]

        _, C_cls, fh, fw = cls_out.shape
        n_anchors = C_cls // 2

        # reshape → (fh*fw*n_anchors, 2) 和 (fh*fw*n_anchors, 4)
        cls_flat  = cls_out[0].reshape(n_anchors, 2, fh * fw).transpose(0, 2, 1).reshape(-1, 2)
        bbox_flat = bbox_out[0].reshape(n_anchors, 4, fh * fw).transpose(0, 2, 1).reshape(-1, 4)

        anc_sizes = anchor_cfg[stride]  # [w0, h0, w1, h1, ...]
        total = fh * fw * n_anchors

        for idx in range(total):
            score = float(cls_flat[idx, 1])
            if score < conf:
                continue

            anc_i = idx // (fh * fw)
            # stride=32 只配了 1 组 anchor，模型若输出更多通道会越界
            if anc_i * 2 + 1 >= len(anc_sizes):
                continue
            pos_i = idx % (fh * fw)
            iy, ix = pos_i // fw, pos_i % fw

            cx = (ix + 0.5) * stride
            cy = (iy + 0.5) * stride
            aw = anc_sizes[anc_i * 2]
            ah = anc_sizes[anc_i * 2 + 1]

            dx, dy, dw, dh = bbox_flat[idx]
            # clip dw/dh to avoid exp overflow
            dw = min(dw * 0.1, 4.0)
            dh = min(dh * 0.1, 4.0)

            pred_cx = cx + aw * dx * 0.1
            pred_cy = cy + ah * dy * 0.1
            pred_w = aw * np.exp(dw)
            pred_h = ah * np.exp(dh)

            x1 = (pred_cx - pred_w / 2) * sx
            y1 = (pred_cy - pred_h / 2) * sy
            x2 = (pred_cx + pred_w / 2) * sx
            y2 = (pred_cy + pred_h / 2) * sy

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 10 or bh < 10:
                continue

            all_dets.append({
                'bbox': (int(x1), int(y1), int(bw), int(bh)),
                'score': score,
            })

    # NMS
    if all_dets:
        boxes_xyxy = [(r['bbox'][0], r['bbox'][1],
                       r['bbox'][0] + r['bbox'][2],
                       r['bbox'][1] + r['bbox'][3]) for r in all_dets]
        scores = [r['score'] for r in all_dets]
        keep = cv2.dnn.NMSBoxes(boxes_xyxy, scores, conf, 0.4)
        if len(keep) > 0:
            keep_flat = [int(x) for x in keep.flatten()]
            all_dets = [all_dets[i] for i in keep_flat if i < len(all_dets)]
    return all_dets

# ─────────────────────────────────────────────
# 绘制检测框
# ─────────────────────────────────────────────

COLORS = {
    'OpenCV DNN': (0, 212, 255),     # 青色
    'YuNet': (255, 0, 128),           # 品红
    'YOLOv8n-Face': (160, 255, 0),    # 亮绿
    'RetinaFace': (255, 170, 0),      # 琥珀
}

def draw_boxes(img_array, detections, color_bgr, label=None):
    """在图像上绘制人脸框"""
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        score = det.get('score', 0)
        
        # 框
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color_bgr, 2)
        
        # 填充
        overlay = img_array.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color_bgr + (50,), -1)
        cv2.addWeighted(overlay, 0.2, img_array, 0.8, 0, img_array)
        
        # 标签
        tag = f"{label} #{i+1}" if label else f"#{i+1}"
        tag += f" {score:.2f}" if score else ""
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)
        cv2.rectangle(img_array, (x, y-th-6), (x+tw+6, y), color_bgr, -1)
        cv2.putText(img_array, tag, (x+3, y-3), font, font_scale, (0,0,0), thickness)

# ─────────────────────────────────────────────
# 上传图片
# ─────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "📤 上传照片（支持多张）",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 上传一张或多张照片开始对比分析")
    st.markdown("""
    ---
    ### 📋 检测器说明

    | 检测器 | 特点 | 适用场景 |
    |--------|------|----------|
    | **OpenCV DNN** | ResNet-SSD，零依赖，CPU 友好 | 大脸特写、证件照 |
    | **YuNet** | 深度学习，误检率低，带关键点 | 通用场景，多人合照 |
    | **YOLOv8n-Face** | YOLO 架构，速度快 | 批量处理，实时检测 |
    | **RetinaFace** | FPN 多尺度，最鲁棒 | 极端角度、小人脸、复杂场景 |

    ---
    💡 **提示**: 不同照片可能适合不同的检测器，建议多试几张找到最适合你的组合。
    """)
    st.stop()

# ─────────────────────────────────────────────
# 加载所有检测器
# ─────────────────────────────────────────────

st.markdown("---")
st.info("🔄 正在加载检测器模型（首次运行可能需要几秒）...")

yunet_detector, yunet_h, yunet_w = _load_yunet()
yolo_model = _load_yolo()
opencv_dnn_net = _load_opencv_dnn()     # (net, mode) 或 None
retinaface_net = _load_retinaface()     # ort.InferenceSession 或 None

# 检测器状态
st.markdown("**检测器状态:**")
cols = st.columns(4)
detector_status = [
    ("OpenCV DNN", opencv_dnn_net is not None),
    ("YuNet", yunet_detector is not None),
    ("YOLOv8n-Face", yolo_model is not None),
    ("RetinaFace", retinaface_net is not None),
]

for col, (name, available) in zip(cols, detector_status):
    with col:
        st.metric(name, "✅ 可用" if available else "❌ 不可用")

st.markdown("---")

# ─────────────────────────────────────────────
# 处理每张图片
# ─────────────────────────────────────────────

# 汇总统计
all_summary = {name: {"total_faces": 0, "total_time": 0, "count": 0} 
               for name, _ in detector_status}

for file in uploaded_files:
    st.subheader(f"📷 {file.name}")
    
    # 读取图片
    bytes_data = file.read()
    # 生产管线用 image_utils.open_upright() 处理 EXIF Orientation；这里原来
    # 直接 convert('RGB')，横持拍的照片会被当成「躺着的图」分析，
    # 对比结论与真实检测管线对不上。
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(bytes_data))).convert('RGB')
    img_array = np.array(pil_img)
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.image(pil_img, use_container_width=True)
        st.caption(f"原始图片 | {pil_img.size[0]}×{pil_img.size[1]} px")
    
    # ── 运行所有检测器 ──
    results_all = {}
    
    with col_right:
        tabs = st.tabs([name for name, _ in detector_status])
        
        for tab, (name, _) in zip(tabs, detector_status):
            with tab:
                start_time = time.time()
                
                if name == "OpenCV DNN":
                    results = detect_opencv_dnn(bgr_img, opencv_dnn_net, conf=0.3)
                elif name == "YuNet":
                    results = detect_yunet(bgr_img, yunet_detector)
                elif name == "YOLOv8n-Face":
                    results = detect_yolo(bgr_img, yolo_model, conf=0.3)
                elif name == "RetinaFace":
                    results = detect_retinaface(bgr_img, retinaface_net, conf=0.3)
                
                elapsed = (time.time() - start_time) * 1000  # ms
                
                # 统计
                all_summary[name]["total_faces"] += len(results)
                all_summary[name]["total_time"] += elapsed
                all_summary[name]["count"] += 1
                results_all[name] = results
                
                # 绘制结果图
                result_img = img_array.copy()
                draw_boxes(result_img, results, COLORS[name], name)
                
                st.image(result_img, use_container_width=True)
                
                # 指标
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("检测到人脸", len(results))
                with m_col2:
                    st.metric("耗时", f"{elapsed:.1f} ms")
                
                if results:
                    st.caption(f"置信度: {[f'{r['score']:.3f}' for r in results]}")
    
    st.markdown("---")

# ─────────────────────────────────────────────
# 对比汇总
# ─────────────────────────────────────────────

st.subheader("📊 检测器对比汇总")

summary_data = []
for name, _ in detector_status:
    s = all_summary[name]
    avg_faces = s['total_faces'] / max(s['count'], 1)
    avg_time = s['total_time'] / max(s['count'], 1)
    summary_data.append({
        "检测器": name,
        "总人脸数": s['total_faces'],
        "平均人脸/图": f"{avg_faces:.1f}",
        "平均耗时": f"{avg_time:.1f} ms",
        "状态": "✅" if s['count'] > 0 else "❌"
    })

st.dataframe(summary_data, use_container_width=True)

# ─────────────────────────────────────────────
# 推荐
# ─────────────────────────────────────────────

st.markdown("---")
st.subheader("💡 建议")

best_by_faces = max(detector_status, key=lambda x: all_summary[x[0]]['total_faces'])
best_by_speed = min([(n, all_summary[n]['total_time'] / max(all_summary[n]['count'], 1)) 
                     for n, _ in detector_status if all_summary[n]['count'] > 0],
                    key=lambda x: x[1])

st.success(f"""
- **最多人脸检测**: {best_by_faces[0]}（共 {all_summary[best_by_faces[0]]['total_faces']} 个）
- **最快检测**: {best_by_speed[0]}（平均 {best_by_speed[1]:.1f} ms/图）
""")

st.info("""
📌 **选择建议**:
- 大脸特写、证件照 → 优先试试 **OpenCV DNN**
- 日常照片、多人合照 → **YuNet** 或 **RetinaFace**
- 批量处理、追求速度 → **YOLOv8n-Face**
- 复杂场景（遮挡、小脸） → **RetinaFace**
""")
