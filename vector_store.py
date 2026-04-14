"""
Milvus向量数据库操作模块
"""
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)
from sentence_transformers import SentenceTransformer
import numpy as np

from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM
)


class VectorStore:
    """Milvus向量数据库操作类"""

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT):
        self.host = host
        self.port = port
        self.collection_name = COLLECTION_NAME
        self.embedding_model = None
        self._connect()

    def _connect(self):
        """建立数据库连接"""
        alias = "default"
        connections.connect(
            alias=alias,
            host=self.host,
            port=self.port
        )
        print(f"已连接到Milvus at {self.host}:{self.port}")

    def _get_embedding_model(self):
        """获取或加载嵌入模型"""
        if self.embedding_model is None:
            print(f"加载嵌入模型: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self.embedding_model

    def create_collection(self, force: bool = False):
        """
        创建评论集合

        Args:
            force: 如果集合已存在，是否强制重建
        """
        if utility.has_collection(self.collection_name):
            if force:
                utility.drop_collection(self.collection_name)
                print(f"已删除现有集合: {self.collection_name}")
            else:
                print(f"集合已存在: {self.collection_name}")
                return

        # 定义集合schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="comment", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="engagement", dtype=DataType.FLOAT),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields, description="评论向量集合")

        # 创建集合
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"创建集合成功: {self.collection_name}")

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("索引创建完成")
        collection.flush()

    def embed_comments(self, comments: List[str]) -> np.ndarray:
        """
        将评论列表转换为向量

        Args:
            comments: 评论列表

        Returns:
            numpy array of embeddings
        """
        model = self._get_embedding_model()
        embeddings = model.encode(comments, show_progress_bar=True)
        return embeddings

    def insert_comments(self, comments: List[Tuple[str, float]]):
        """
        批量插入评论到向量数据库

        Args:
            comments: List of (comment_text, engagement) tuples
        """
        collection = Collection(self.collection_name)

        # 分离评论和互动量
        comment_texts = [c[0] for c in comments]
        engagements = [c[1] for c in comments]

        # 生成向量
        embeddings = self.embed_comments(comment_texts)

        # 准备插入数据
        entities = [
            comment_texts,  # comment field
            engagements,    # engagement field
            embeddings.tolist()  # embedding field
        ]

        # 插入数据
        collection.insert(entities)
        collection.flush()
        print(f"成功插入 {len(comments)} 条评论")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        搜索最相似的评论

        Args:
            query: 搜索文本
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        collection = Collection(self.collection_name)
        collection.load()

        # 生成查询向量
        query_embedding = self.embed_comments([query])[0].tolist()

        # 搜索
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["comment", "engagement"]
        )

        # 整理结果
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "id": hit.id,
                    "comment": hit.entity.get("comment"),
                    "engagement": hit.entity.get("engagement"),
                    "distance": hit.distance
                })

        return search_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not utility.has_collection(self.collection_name):
            return {"exists": False}

        collection = Collection(self.collection_name)
        stats = collection.num_entities
        return {
            "exists": True,
            "name": self.collection_name,
            "entities": stats
        }

    def close(self):
        """关闭连接"""
        connections.disconnect("default")


def init_vector_store(force_recreate: bool = False) -> VectorStore:
    """
    初始化向量数据库

    Args:
        force_recreate: 是否强制重建集合

    Returns:
        VectorStore实例
    """
    store = VectorStore()
    store.create_collection(force=force_recreate)
    return store


if __name__ == "__main__":
    # 测试连接
    store = VectorStore()
    stats = store.get_collection_stats()
    print(f"集合状态: {stats}")
    store.close()
