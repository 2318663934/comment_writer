"""
RAG检索模块 - 基于话题检索相关评论
"""
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from config import TOP_K
import time


class RAGRetriever:
    """RAG检索器"""

    # 类级别检索记忆，追踪最近使用过的Chunk ID和访问时间
    _retrieval_memory: Dict[str, List[tuple]] = {}  # {session_id: [(chunk_id, timestamp), ...]}
    _memory_max_size = 500  # 每个session最多记录500个chunk
    _memory_ttl = 600  # 10分钟后自动过期
    _recent_generation_count: Dict[str, int] = {}  # 记录连续生成次数
    _last_generation_time: Dict[str, float] = {}  # 上次生成时间

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        self._session_id = str(id(self))
        self._last_access = time.time()

    def _get_recent_chunk_ids(self) -> tuple:
        """
        获取最近使用过的Chunk ID集合
        Returns: (chunk_ids_set, recent_count) - ID集合和连续生成次数
        """
        self._last_access = time.time()
        current_time = time.time()

        # 清理过期记忆
        if self._session_id in self._retrieval_memory:
            self._retrieval_memory[self._session_id] = [
                (cid, ts) for cid, ts in self._retrieval_memory[self._session_id]
                if current_time - ts < self._memory_ttl
            ]

        if self._session_id not in self._retrieval_memory or not self._retrieval_memory[self._session_id]:
            return set(), 0

        # 获取连续生成次数
        recent_count = self._recent_generation_count.get(self._session_id, 0)

        # 返回所有历史chunk ID（不只是最近的）
        chunk_ids = [cid for cid, _ in self._retrieval_memory[self._session_id]]
        return set(chunk_ids), recent_count

    def _add_to_memory(self, chunk_ids: List[int]):
        """将Chunk ID添加到记忆"""
        current_time = time.time()

        if self._session_id not in self._retrieval_memory:
            self._retrieval_memory[self._session_id] = []

        # 添加新ID到开头（最新的在前面）
        memory_list = self._retrieval_memory[self._session_id]
        for cid in chunk_ids:
            # 移除旧的记录（如果存在）
            memory_list = [(c, t) for c, t in memory_list if c != cid]
            # 添加新记录到开头
            memory_list.insert(0, (cid, current_time))

        # 限制记忆大小
        if len(memory_list) > self._memory_max_size:
            memory_list = memory_list[:self._memory_max_size]

        self._retrieval_memory[self._session_id] = memory_list

        # 增加连续生成计数并记录时间
        self._recent_generation_count[self._session_id] = self._recent_generation_count.get(self._session_id, 0) + 1
        self._last_generation_time[self._session_id] = current_time

    def reset_generation_count(self):
        """重置连续生成计数（当用户长时间未生成时调用）"""
        self._recent_generation_count[self._session_id] = 0

    def _filter_recent_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从检索结果中过滤掉历史使用过的Chunk，使用动态排除比例

        Args:
            results: 检索结果

        Returns:
            过滤后的结果
        """
        current_time = time.time()

        # 检查是否超时（2分钟未生成则重置计数）
        if self._session_id in self._last_generation_time:
            if current_time - self._last_generation_time[self._session_id] > 120:
                self._recent_generation_count[self._session_id] = 0

        recent_ids, recent_count = self._get_recent_chunk_ids()
        if not recent_ids:
            return results

        # 动态排除比例：连续生成次数越多，排除越多
        # 第1次: 40%, 第2次: 60%, 第3次+: 80%
        if recent_count <= 1:
            exclude_ratio = 0.4
        elif recent_count == 2:
            exclude_ratio = 0.6
        else:
            exclude_ratio = 0.8

        # 按距离排序（距离小的相关性高）
        sorted_results = sorted(results, key=lambda x: x.get("distance", float('inf')))

        # 先把所有结果分为"未使用"和"已使用"两组
        unused = [r for r in sorted_results if r["id"] not in recent_ids]
        used = [r for r in sorted_results if r["id"] in recent_ids]

        # 优先从unused中选择，保证相关性
        max_exclude = int(len(sorted_results) * exclude_ratio)

        # 如果unused足够，直接返回unused
        if len(unused) >= len(sorted_results) * (1 - exclude_ratio):
            return unused[:int(len(sorted_results) * (1 - exclude_ratio))] + used[:int(len(sorted_results) * exclude_ratio)]

        # 否则按比例混合
        selected = []
        used_count = 0

        for r in unused:
            selected.append(r)

        for r in used:
            if used_count < max_exclude:
                selected.append(r)
                used_count += 1

        # 打乱结果顺序增加多样性
        import random
        random.shuffle(selected)

        return selected

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
        directions: List[str],
        mmr_lambda: float = 0.7,
        event_info: str = "",
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        检索与话题相关的评论（支持多方向一次性检索，使用MMR增加多样性）

        Args:
            topic: 用户输入的话题
            num_comments: 需要生成的评论数量
            directions: 评论方向列表
            mmr_lambda: MMR参数，0-1之间，越高越注重相关性，越低越注重多样性
            event_info: 事件背景文本，用于提取查询变体
            seed: 随机种子，用于控制变体选择顺序和结果打乱

        Returns:
            检索到的评论列表（去重后）
        """
        import random
        random.seed(seed)

        # 根据生成数量调整检索数量，确保每方向有足够参考
        retrieval_count = max(TOP_K, num_comments * len(directions) * 3)

        # 生成多个查询变体（从事件背景中自适应提取）
        query_variants = self._generate_query_variants(topic, directions, event_info)

        # 多路检索：每个变体分别检索
        all_results = []
        for query in query_variants:
            results = self.vector_store.search_mmr(query, top_k=retrieval_count, mmr_lambda=mmr_lambda)
            all_results.extend(results)

        # 去重合并（基于MMR结果再做一次去重）
        deduped = self._deduplicate(all_results)

        # 过滤：使用宽松条件
        filtered = [r for r in deduped if r["distance"] < 3.0]

        # 过滤掉最近使用过的Chunk（检索记忆）
        filtered = self._filter_recent_chunks(filtered)

        # 记录本次使用的Chunk ID到记忆
        chunk_ids = [r["id"] for r in filtered[:retrieval_count]]
        self._add_to_memory(chunk_ids)

        return filtered[:retrieval_count]

    def _extract_keywords_from_event(self, event_info: str) -> List[str]:
        """
        从事件背景中提取关键词（适用于官方公告风格）

        Args:
            event_info: 事件背景文本

        Returns:
            提取出的关键词列表
        """
        import re
        import random

        if not event_info or not event_info.strip():
            return []

        keywords = []

        # 模式1: #话题# 或 【话题】 标签
        hashtags = re.findall(r'[#【]([^#】]+)[#】]', event_info)
        keywords.extend([h.strip() for h in hashtags if 1 <= len(h.strip()) <= 10])

        # 模式2: 常见官方公告关键词组合（2-6字）
        # 游戏相关后缀词
        suffixes = ['皮肤', '活动', '更新', '版本', '玩法', '角色', '英雄', '赛事',
                    '限定', '惊喜', '庆典', '节日', '礼包', '福利', '上线', '开启',
                    '来袭', '登场', '发布', '爆料', '精修', '优化', '调整', '修改']
        for suffix in suffixes:
            pattern = rf'([\u4e00-\u9fa5]{{1,5}}{re.escape(suffix)})'
            matches = re.findall(pattern, event_info)
            keywords.extend([m for m in matches if 3 <= len(m) <= 8])

        # 模式3: 特定前缀 + 名词组合
        prefixes = ['新', '首', '限定', '联动', '荣耀', '王者', '传奇', '史诗', '传说', '珍贵']
        for prefix in prefixes:
            pattern = rf'{prefix}[\u4e00-\u9fa5]{{1,4}}'
            matches = re.findall(pattern, event_info)
            keywords.extend([m for m in matches if 2 <= len(m) <= 6])

        # 模式4: 数字 + 时间/数量词（如5月1日、2024春节、30天等）
        time_patterns = re.findall(r'(\d+[号日周月年]|\d+天|\d+小时)', event_info)
        # 只保留较短的数字相关词
        for tp in time_patterns:
            if len(tp) <= 6:
                keywords.append(tp)

        # 模式5: 提取引号内的内容（通常是重点强调词）
        quoted = re.findall(r'"([^"]+)"', event_info)
        keywords.extend([q for q in quoted if 2 <= len(q) <= 8])

        # 模式6: 连续的中文短语（2-4字词组滑动窗口）
        # 提取所有2-4字的连续中文作为候选
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', event_info)
        # 过滤掉常见的无意义词
        stop_words = {'的是', '的', '是的', '是的', '是的', '就是', '就是', '就是',
                     '有的', '有的', '这个', '那个', '这些', '那些', '大家', '大家',
                     '对于', '关于', '由于', '因此', '所以', '但是', '而且', '并且'}
        for word in chinese_words:
            if word not in stop_words:
                keywords.append(word)

        # 去重并随机打乱
        keywords = list(set(keywords))
        random.shuffle(keywords)

        # 返回3-8个有意义的关键词
        return keywords[:8]

    def _generate_query_variants(self, topic: str, directions: List[str], event_info: str = "") -> List[str]:
        """
        生成多个查询变体，从事件背景中自适应提取关键词

        Args:
            topic: 原始话题
            directions: 评论方向列表
            event_info: 事件背景文本

        Returns:
            查询变体列表
        """
        import random

        # 合并方向关键词
        direction_keywords = {
            "正性向": "好评 赞扬 支持 喜欢",
            "中性向": "客观 分析 评价",
            "中正性向": "理性 中立 整体"
        }
        direction_kw = " ".join([direction_keywords.get(d, "") for d in directions])

        # 从事件背景中提取关键词
        event_keywords = self._extract_keywords_from_event(event_info)

        # 构建基础查询
        base_query = f"{topic} {direction_kw}"

        # 根据是否有事件关键词选择不同的变体模板
        if event_keywords:
            # 随机选择2-3个关键词构建变体（每次不同）
            selected_kws = random.sample(event_keywords, min(len(event_keywords), random.randint(2, 3)))

            templates = [
                # 原始+关键词组合
                lambda kws=selected_kws: f"{topic} {' '.join(kws)} {direction_kw}",
                # 拆分关键词单独检索
                lambda kws=selected_kws: f"{topic} {kws[0]} {direction_kw}",
                lambda kws=selected_kws: f"{topic} {kws[-1]} {direction_kw}" if len(kws) > 1 else None,
                # 关键词+时间/活动角度
                lambda kws=selected_kws: f"{topic} {kws[0]} 活动 {direction_kw}",
                lambda kws=selected_kws: f"{topic} {kws[0]} 限定 {direction_kw}" if len(kws) > 0 else None,
            ]
            # 过滤掉None
            templates = [t for t in templates if t is not None]
        else:
            # 无事件关键词时使用通用变体
            templates = [
                lambda: base_query,
                lambda: f"{topic} 玩家 游戏 体验 {direction_kw}",
                lambda: f"{topic} 这次 更新 {direction_kw}",
                lambda: f"{topic} 玩法 活动 内容 {direction_kw}",
                lambda: f"{topic} 感觉 觉得 {direction_kw}",
            ]

        # 生成变体
        variants = []
        for template in templates:
            variant = template()
            if variant and variant not in variants:
                variants.append(variant)

        # 随机打乱顺序，保证每次生成顺序不同
        random.shuffle(variants)

        # 返回3-5个变体
        num_variants = min(len(variants), max(3, min(5, len(directions) * 2 + 1)))
        return variants[:num_variants]

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
