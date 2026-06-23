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
    """Milvus向量数据库操作类（支持多集合）"""

    def __init__(self, host: str = MILVUS_HOST, port: int = MILVUS_PORT,
                 collection_name: str = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name or COLLECTION_NAME
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
        # max_length 按字节计，中文每个字符约3字节，3000字节≈1000个中文字
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="comment", dtype=DataType.VARCHAR, max_length=3000),
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
        from config import MAX_COMMENT_LEN
        collection = Collection(self.collection_name)

        # 分离评论和互动量，同时截断过长评论
        comment_texts = []
        engagements = []
        for c, e in comments:
            if len(c) > MAX_COMMENT_LEN:
                c = c[:MAX_COMMENT_LEN]
            comment_texts.append(c)
            engagements.append(e)

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

    def search_mmr(
        self,
        query: str,
        top_k: int = 20,
        mmr_lambda: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        使用MMR（最大边际相关）算法搜索多样化的评论

        Args:
            query: 搜索文本
            top_k: 返回数量
            mmr_lambda: MMR参数，0-1之间，越高越注重相关性，越低越注重多样性

        Returns:
            多样化搜索结果列表
        """
        import numpy as np

        collection = Collection(self.collection_name)
        collection.load()

        # 生成查询向量
        query_embedding = self.embed_comments([query])[0]

        # 先检索更多候选（确保有足够选择空间）
        candidate_count = min(top_k * 10, 200)
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=candidate_count,
            output_fields=["comment", "engagement", "embedding"]
        )

        # 提取候选结果和向量（过滤掉embedding为None的结果）
        candidates = []
        candidate_embeddings = []
        for hits in results:
            for hit in hits:
                emb = hit.entity.get("embedding")
                if emb is None:
                    continue
                candidates.append({
                    "id": hit.id,
                    "comment": hit.entity.get("comment"),
                    "engagement": hit.entity.get("engagement"),
                    "distance": hit.distance
                })
                candidate_embeddings.append(emb)

        if not candidates:
            return []

        candidate_embeddings = np.array(candidate_embeddings)

        # 计算查询与所有候选的相关性（转换为相似度，距离越小越相似）
        # L2距离转相似度: sim = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + np.array([c["distance"] for c in candidates]))

        # MMR选择
        selected = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining_indices:
                break

            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                # 相关性分数
                relevance = similarities[idx]

                # 多样性分数：与已选中结果的最大相似度
                if selected:
                    selected_embeddings = np.array([candidate_embeddings[i] for i in selected])
                    diff = selected_embeddings - candidate_embeddings[idx]
                    diversities = 1.0 / (1.0 + np.linalg.norm(diff, axis=1))
                    diversity = float(np.max(diversities))
                else:
                    diversity = 0.0

                # MMR分数
                mmr_score = mmr_lambda * float(relevance) - (1 - mmr_lambda) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected]

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

    def switch_collection(self, collection_name: str):
        """切换到指定的集合"""
        self.collection_name = collection_name

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取当前集合统计信息"""
        if not utility.has_collection(self.collection_name):
            return {"exists": False, "name": self.collection_name}

        collection = Collection(self.collection_name)
        stats = collection.num_entities
        return {
            "exists": True,
            "name": self.collection_name,
            "entities": stats
        }

    @staticmethod
    def get_available_collections() -> list:
        """列出所有可用的评论集合"""
        from config import STYLE_CONFIG
        # 确保连接存在
        from pymilvus import connections as _conns
        try:
            if "default" not in [alias for alias, _ in _conns.list_connections()]:
                _conns.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception:
            try:
                _conns.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            except Exception:
                return []
        available = []
        for style_name, cfg in STYLE_CONFIG.items():
            col_name = cfg["collection"]
            try:
                if utility.has_collection(col_name):
                    collection = Collection(col_name)
                    available.append({
                        "style": style_name,
                        "collection": col_name,
                        "entities": collection.num_entities,
                    })
            except Exception:
                continue
        return available

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
