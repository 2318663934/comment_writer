"""
评论写手系统配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "wangzhe_comments"  # 王者荣耀评论集合（默认）
COLLECTION_ANIME = "anime_comments"   # 二次元风格评论集合

# 评论风格配置
STYLE_CONFIG = {
    "王者荣耀": {
        "collection": "wangzhe_comments",
        "description": "王者荣耀玩家评论风格",
    },
    "二次元": {
        "collection": "anime_comments",
        "description": "二次元/B站动漫游戏评论风格",
    },
}
DEFAULT_STYLE = "王者荣耀"

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

# 知识库检索模式
# "rag"      — 纯 RAG: 用 E:/产品信息知识库 Milvus (原行为)
# "wiki"     — 纯 Wiki: 用 E:/deepsearch/wiki/ (LLM-Wiki 6 产品)
# "hybrid"   — 混合: Wiki 优先, 失败降级到 RAG (推荐)
KNOWLEDGE_MODE = os.getenv("KNOWLEDGE_MODE", "hybrid")

# 智谱多模态配置（用于图片/视频信息提取）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4.6v-flash")
ZHIPU_ENABLED = os.getenv("ZHIPU_ENABLED", "false").lower() == "true"

# Whisper 语音转录配置
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # tiny/base/small/medium/large

# 数据路径
DATA_FILE = "D:/文案查重复.xlsx"
