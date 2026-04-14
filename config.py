"""
评论写手系统配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "wangzhe_comments"  # 王者荣耀评论集合

# 嵌入模型配置
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
EMBEDDING_DIM = 768  # text2vec-base-chinese 输出维度

# LLM 配置
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# RAG 配置
TOP_K = 20  # 默认检索数量
MAX_COMMENTS = 100  # 最大生成数量
MIN_COMMENT_LEN = 10  # 最小评论长度（过滤垃圾数据）
MAX_COMMENT_LEN = 1000  # 最大评论长度

# 数据路径
DATA_FILE = "D:/文案查重复.xlsx"
