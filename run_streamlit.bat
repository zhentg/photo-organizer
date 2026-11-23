@echo off
chcp 65001 >nul
cd /d D:\CADD\workspace\qclaw\projects\face-photo-sort
D:\CADD\workspace\qclaw\tools\miniconda3\python.exe -m streamlit run app.py --server.port 8501 --server.baseUrlPath photoClassification --browser.gatherUsageStats false
