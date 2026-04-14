"""
数据加载模块 - 从Excel文件读取评论数据
"""
import pandas as pd
import re
from typing import List, Tuple
from config import DATA_FILE, MIN_COMMENT_LEN, MAX_COMMENT_LEN


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


if __name__ == "__main__":
    comments = load_comments_from_excel()
    print(f"\n前5条评论示例:")
    for i, (c, e) in enumerate(comments[:5]):
        print(f"{i+1}. [{e:.1f}] {c[:60]}...")
