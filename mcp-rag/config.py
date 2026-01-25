"""
MCP RAG Demo 服务配置模块
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# Chroma 配置
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "rag_demo"

# 嵌入模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 检索配置
DEFAULT_TOP_K = 3

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"

# 确保必要目录存在
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
