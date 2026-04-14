"""
RAG检索模块 - 基于话题检索相关评论
"""
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from config import TOP_K


class RAGRetriever:
    """RAG检索器"""

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        topic: str,
        num_comments: int = 10,
        direction: str = "中性向"
    ) -> List[Dict[str, Any]]:
        """
        检索与话题相关的评论

        Args:
            topic: 用户输入的话题
            num_comments: 需要生成的评论数量
            direction: 评论方向 (正性向/中性向/中正性向)

        Returns:
            检索到的评论列表
        """
        # 根据生成数量调整检索数量
        # 检索多一些以便LLM学习
        retrieval_count = max(TOP_K, num_comments * 2)

        # 构建检索query
        search_query = self._build_search_query(topic, direction)

        # 执行向量检索
        results = self.vector_store.search(search_query, top_k=retrieval_count)

        # 过滤和处理结果
        filtered = self._filter_results(results, direction)

        # 按相关性排序并去重
        deduped = self._deduplicate(filtered)

        return deduped[:retrieval_count]

    def _build_search_query(self, topic: str, direction: str) -> str:
        """
        构建检索查询

        Args:
            topic: 话题
            direction: 评论方向

        Returns:
            组合后的搜索query
        """
        # 方向关键词
        direction_keywords = {
            "正性向": "好评 赞扬 支持 喜欢 期待",
            "中性向": "客观 分析 评价 看法 观点",
            "中正性向": "理性 中立 客观 分析"
        }

        keywords = direction_keywords.get(direction, "")
        return f"{topic} {keywords}"

    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        direction: str
    ) -> List[Dict[str, Any]]:
        """
        过滤检索结果

        保留相关性较高且符合方向的结果
        """
        # 简单过滤：保留距离较小（相似度较高）的结果
        # L2距离越小表示越相似
        filtered = [r for r in results if r["distance"] < 2.0]

        # 如果过滤后结果太少，放宽条件
        if len(filtered) < 5:
            filtered = results[:10]

        return filtered

    def _deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对检索结果去重

        基于评论内容相似度去重
        """
        seen = set()
        deduped = []

        for r in results:
            comment = r["comment"]
            # 简单去重：取前30个字符作为key
            key = comment[:30].lower().strip()

            if key not in seen and len(comment) > 10:
                seen.add(key)
                deduped.append(r)

        return deduped

    def retrieve_for_few_shot(
        self,
        topic: str,
        num_examples: int = 5
    ) -> List[str]:
        """
        检索Few-shot示例评论

        用于LLM理解评论风格
        """
        results = self.retrieve(topic, num_comments=num_examples)
        return [r["comment"] for r in results[:num_examples]]


if __name__ == "__main__":
    # 测试检索
    retriever = RAGRetriever()

    # 检查集合状态
    stats = retriever.vector_store.get_collection_stats()
    print(f"集合状态: {stats}")

    if stats.get("exists") and stats.get("entities", 0) > 0:
        # 测试检索
        results = retriever.retrieve("孙策新皮肤", num_comments=5)
        print(f"\n检索到 {len(results)} 条相关评论:")
        for i, r in enumerate(results[:5]):
            print(f"{i+1}. {r['comment'][:60]}... (距离:{r['distance']:.3f})")
    else:
        print("集合为空，请先运行 build_database.py 构建数据库")
