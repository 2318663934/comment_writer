"""
数据加载模块 - 从Excel文件读取评论数据
"""
import numpy as np
import pandas as pd
import re
from typing import List, Tuple
from config import DATA_FILE, MIN_COMMENT_LEN, MAX_COMMENT_LEN, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer


def load_comments_from_excel(file_path: str = None) -> List[Tuple[str, float]]:
    """
    从Excel文件加载评论数据

    Returns:
        List of (comment, engagement) tuples
    """
    if file_path is None:
        file_path = DATA_FILE

    df = pd.read_excel(file_path)
    print(f"Excel文件加载完成: {df.shape[0]}行 x {df.shape[1]}列")

    comments = []
    seen = set()  # 用于去重

    # 遍历所有列，提取评论文本
    for col in df.columns:
        for idx, val in enumerate(df[col]):
            if not isinstance(val, str):
                continue

            # 清理评论文本
            cleaned = clean_comment(val)

            # 过滤短评论和重复评论
            if len(cleaned) < MIN_COMMENT_LEN:
                continue
            if len(cleaned) > MAX_COMMENT_LEN:
                continue
            if cleaned in seen:
                continue

            # 尝试从相邻列获取互动量
            engagement = 0.0
            if idx < len(df) - 1:
                next_val = df[col].iloc[idx + 1]
                if isinstance(next_val, (int, float)) and not pd.isna(next_val):
                    engagement = float(next_val)

            seen.add(cleaned)
            comments.append((cleaned, engagement))

    print(f"提取到 {len(comments)} 条有效评论")
    return comments


def comment_quality_score(text: str) -> float:
    """
    评估评论质量，分数越高越像真实玩家评论（而非官方/营销语料）。

    评估维度：
    - 惩罚官方/营销用语
    - 奖励口语化表达
    - 惩罚空洞敷衍内容
    """
    score = 0.5

    # ---- 惩罚项：官方/营销/PR 用语 ----
    official_patterns = [
        '推荐', '值得入手', '性价比', '不容错过', '强烈推荐',
        '总的来说', '综上所述', '首先', '其次', '最后',
        '这款', '该游戏', '玩家们', '大家快来', '年度最佳',
        '巅峰之作', '良心之作', '精心打造', '诚意满满',
        '必入', '入手推荐', '大家可以', '一定要试',
        '不要错过', '绝对值得', '太值得', '非常值得',
    ]
    official_count = sum(1 for p in official_patterns if p in text)
    if official_count > 0:
        score -= 0.12 * min(official_count, 5)

    # 官方长句式特征（长 + 多个官方用词 = 典型的PR软文）
    if len(text) > 80 and official_count >= 2:
        score -= 0.2

    # ---- 惩罚项：口语空洞敷衍 ----
    empty_patterns = ['还行吧', '一般般', '凑合', '就那样', '感觉一般']
    for p in empty_patterns:
        if p in text:
            score -= 0.15

    # ---- 惩罚项：过度使用感叹号（AI 特征） ----
    if text.count('！') + text.count('!') > 2:
        score -= 0.15

    # ---- 过滤垃圾 ----
    if len(text) < 10:
        score -= 0.3

    # ---- 奖励项：口语化表达（长短评论均可具备） ----
    casual_patterns = [
        '感觉', '觉得', '有点', '好像', '就是', '其实',
        '不过', '但是', '还是', '真的', '说实话',
        '之前', '上次', '这次', '总算', '终于',
        '还行', '还行吧', '不错', '可以', '挺',
    ]
    casual_count = sum(1 for p in casual_patterns if p in text)
    score += 0.03 * min(casual_count, 8)

    return max(0.0, min(1.0, score))


def clean_comment(text: str) -> str:
    """
    清理评论文本
    """
    if not isinstance(text, str):
        return ""

    # 移除表情符号文字描述 [大笑R][doge] 等
    text = re.sub(r'\[.*?\]', '', text)

    # 移除#话题标签
    text = re.sub(r'#\w+', '', text)

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 移除首尾空白
    text = text.strip()

    return text


def filter_by_quality(
    comments: List[Tuple[str, float]],
    min_score: float = 0.3
) -> List[Tuple[str, float]]:
    """
    按质量分数过滤评论，移除低质量/官方风格的评论

    Args:
        comments: List of (comment, engagement) tuples
        min_score: 最低质量分数阈值（0-1），默认0.3

    Returns:
        过滤后的评论列表
    """
    if not comments:
        return comments

    scored = [(c, e, comment_quality_score(c)) for c, e in comments]

    # 统计分布
    all_scores = [s for _, _, s in scored]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    filtered_in = sum(1 for _, _, s in scored if s >= min_score)

    print(f"质量过滤: 平均分 {avg_score:.3f}, 阈值 {min_score}, "
          f"保留 {filtered_in}/{len(comments)} 条")

    return [(c, e) for c, e, s in scored if s >= min_score]


def deduplicate_comments(comments: List[str]) -> List[str]:
    """
    对评论列表去重
    """
    seen = set()
    result = []
    for c in comments:
        normalized = c.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(c)
    return result


def semantic_deduplicate(
    comments: List[Tuple[str, float]],
    threshold: float = 0.82,
    batch_size: int = 500
) -> List[Tuple[str, float]]:
    """
    基于语义的去重：计算两两余弦相似度，阈值>threshold视为重复，
    保留互动量(engagement)最高者。

    Args:
        comments: List of (comment, engagement) tuples
        threshold: 相似度阈值，超过则视为重复（默认0.82）
        batch_size: 分批计算嵌入的批量大小

    Returns:
        去重后的 comments list
    """
    if len(comments) < 2:
        return comments

    print(f"开始语义去重，共 {len(comments)} 条评论，阈值 {threshold}...")

    # 加载模型
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 分批计算嵌入
    all_embeddings = []
    texts = [c[0] for c in comments]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings)

    # L2归一化，余弦相似度 = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 防止除零
    normalized = embeddings / norms

    n = len(comments)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # 并查集：遍历上三角，相似则合并
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(normalized[i], normalized[j]))
            if sim > threshold:
                union(i, j)

    # 按engagement降序排序后去重
    sorted_indices = sorted(range(n), key=lambda idx: comments[idx][1], reverse=True)

    kept = set()
    result = []
    for idx in sorted_indices:
        root = find(idx)
        if root not in kept:
            kept.add(root)
            result.append(comments[idx])

    print(f"语义去重完成: {len(comments)} -> {len(result)}")
    return result


if __name__ == "__main__":
    comments = load_comments_from_excel()
    print(f"\n前5条评论示例:")
    for i, (c, e) in enumerate(comments[:5]):
        print(f"{i+1}. [{e:.1f}] {c[:60]}...")
