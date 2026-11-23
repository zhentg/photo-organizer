# app.py - 首页
import streamlit as st
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="人脸照片分类工具",
    page_icon="📸",
    layout="wide"
)

# 首页内容
st.title("📸 人脸照片分类工具")
st.markdown("---")

st.markdown("""
### 🎯 功能介绍

这是一个基于深度学习的人脸照片分类管理工具，支持：

- **📁 导入照片** - 从文件夹导入照片，自动检测人脸
- **👤 人脸标注** - 为检测到的人脸指定姓名，支持批量操作
- **📂 分类浏览** - 按人物浏览照片，支持重新识别
- **📋 照片索引** - 查看所有已索引的照片，支持搜索和筛选
- **💾 导出管理** - 导出/导入人脸库和照片索引
- **🔍 重复管理** - 检测和清理重复的照片和人脸编码
- **⚠️ 缺失文件** - 清理文件已删除但索引中仍存在的记录

### 📌 使用流程

1. 点击左侧 **📁 导入照片** 导入照片文件夹
2. 系统自动检测人脸并尝试匹配已知人物
3. 点击 **👤 人脸标注** 为未知人脸指定姓名
4. 在 **📂 分类浏览** 按人物查看和整理照片
5. 使用 **💾 导出管理** 导出整理好的照片

---

💡 **提示**: 使用左侧导航栏切换不同功能页面
""")

# 显示统计信息
try:
    from core.shared import get_photo_manager, get_persons_snapshot

    photo_manager = get_photo_manager()
    all_persons = get_persons_snapshot()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已索引照片", photo_manager.get_photo_count())
    with col2:
        st.metric("已标注人物", len(all_persons))
    with col3:
        encoding_count = sum(len(p.get("photos", [])) for p in all_persons.values())
        st.metric("人脸编码", encoding_count)

except Exception as e:
    st.info("系统初始化中...")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    💡 点击左侧导航栏开始使用
</div>
""", unsafe_allow_html=True)
