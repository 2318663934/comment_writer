"""
数据库构建脚本 - 从Excel文件构建Milvus向量数据库
"""
import sys
import time
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, "e:/评论写手")

from data_loader import load_comments_from_excel, clean_comment
from vector_store import VectorStore


def build_database(batch_size: int = 500, force_recreate: bool = False):
    """
    从Excel文件构建向量数据库

    Args:
        batch_size: 批量插入大小
        force_recreate: 是否强制重建数据库
    """
    print("=" * 60)
    print("评论写手 - 数据库构建")
    print("=" * 60)

    # 初始化向量存储
    print("\n[1/4] 初始化Milvus连接...")
    store = VectorStore()

    # 创建集合
    print("\n[2/4] 创建集合...")
    store.create_collection(force=force_recreate)

    # 加载评论数据
    print("\n[3/4] 加载Excel数据...")
    comments = load_comments_from_excel()

    if not comments:
        print("错误: 未找到有效评论数据")
        return

    # 批量插入
    print(f"\n[4/4] 插入向量数据 (共{len(comments)}条)...")

    total_batches = (len(comments) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="插入进度"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(comments))
        batch = comments[start_idx:end_idx]

        try:
            store.insert_comments(batch)
        except Exception as e:
            print(f"\n批次 {i+1} 插入失败: {e}")
            # 继续下一个批次

    # 验证结果
    print("\n验证数据库...")
    stats = store.get_collection_stats()
    print(f"集合状态: {stats}")

    store.close()

    print("\n" + "=" * 60)
    print("数据库构建完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建评论向量数据库")
    parser.add_argument("--batch-size", type=int, default=500, help="批量插入大小")
    parser.add_argument("--force", action="store_true", help="强制重建数据库")
    args = parser.parse_args()

    # 等待Milvus启动
    print("等待Milvus服务就绪...")
    time.sleep(2)

    build_database(batch_size=args.batch_size, force_recreate=args.force)
