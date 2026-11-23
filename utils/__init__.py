"""Utils module"""
import os

def ensure_dir(path: str):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
