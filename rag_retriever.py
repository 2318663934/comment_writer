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
        return self.retrieve_for_directions(topic, num_comments, [direction])

    def retrieve_for_directions(
        self,
        topic: str,
        num_comments: int,
        directions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        检索与话题相关的评论（支持多方向一次性检索）

        Args:
            topic: 用户输入的话题
            num_comments: 需要生成的评论数量
            directions: 评论方向列表

        Returns:
            检索到的评论列表（去重后）
        """
        # 根据生成数量调整检索数量，确保每方向有足够参考
        retrieval_count = max(TOP_K, num_comments * len(directions) * 3)

        # 构建检索query，合并所有方向的关键词
        search_query = self._build_search_query(topic, directions)

        # 执行向量检索
        results = self.vector_store.search(search_query, top_k=retrieval_count)

        # 过滤：使用最宽松条件，保留所有相关结果
        filtered = [r for r in results if r["distance"] < 3.0]

        # 按相关性排序并去重
        deduped = self._deduplicate(filtered)

        return deduped[:retrieval_count]

    def _build_search_query(self, topic: str, directions: List[str]) -> str:
        """
        构建检索查询

        Args:
            topic: 话题
            directions: 评论方向列表

        Returns:
            组合后的搜索query
        """
        # 合并所有方向的关键词
        all_keywords = []
        direction_keywords = {
            "正性向": "好评 赞扬 支持 喜欢 期待 正面",
            "中性向": "客观 分析 评价 看法 观点 中性",
            "中正性向": "理性 中立 偏正面 整体"
        }
        for d in directions:
            keywords = direction_keywords.get(d, "")
            all_keywords.extend(keywords.split())

        # 去重
        unique_keywords = list(set(all_keywords))
        keywords_str = " ".join(unique_keywords)

        return f"{topic} {keywords_str}"

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

        基于评论内容相似度去重，使用更精确的指纹算法
        """
        from collections import defaultdict
        import re

        def normalize(text: str) -> str:
            """文本归一化：去除标点、转为小写、去除多余空格"""
            text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)  # 只保留中文和字母数字
            return text.lower().strip()

        def get_fingerprint(comment: str, length: int = 50) -> str:
            """提取评论指纹：取多个位置的字符组合"""
            cleaned = normalize(comment)
            if len(cleaned) <= length:
                return cleaned
            # 取前中后三段组合，增强区分度
            return cleaned[:20] + cleaned[len(cleaned)//2-10:len(cleaned)//2+10] + cleaned[-20:]

        seen_fingerprints = set()
        deduped = []

        for r in results:
            comment = r["comment"]
            if len(comment) <= 10:
                continue

            fp = get_fingerprint(comment)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
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
