# WikiBackend — 评论写手接入 LLM-Wiki 知识库

> 🎯 让评论写手用 deepsearch 项目的 **6 产品知识库** (4353 md) 替代 Milvus RAG。

## 📊 后端对比

| 维度 | RAG (Milvus) | WikiBackend (LLM-Wiki) |
|------|--------------|------------------------|
| 速度 | ⚡ ~1s | 🐢 10-30s/轮 (本地 Qwen) |
| 检索 | 模糊 (向量相似度) | 精确 (ReAct 实体浏览) |
| 数据 | 文本切片 (碎) | 完整实体 (frontmatter+body) |
| 可维护 | 重 build embedding | 改 md 即可 |
| 可审计 | 难 (向量无解释) | **强** (trace 列出访问的 md) |
| 离线 | ❌ 需 Milvus 服务 | ✅ 只需本地 Qwen |
| 数据量 | 取决于 collection | **6 产品 4353 md** |

## 🚀 3 种接入方式

### 方式 1: 替换 retriever (推荐)

```python
# 原 RAGRetriever:
from rag_retriever import RAGRetriever
backend = RAGRetriever(collection_name="王者评论")

# 新 WikiBackend:
from wiki_backend import WikiBackend
backend = WikiBackend(product="王者")
```

接口完全兼容:`retrieve(topic, num_comments, direction) -> List[Dict]`,
每个 dict 含 `id`/`comment`/`distance`,`comment_generator.py` 无需修改。

### 方式 2: 在 config.py 加开关

```python
# config.py 加一个开关:
KNOWLEDGE_BACKEND = "rag"  # "rag" 或 "wiki"

def make_backend(product: str):
    if KNOWLEDGE_BACKEND == "wiki":
        from wiki_backend import WikiBackend
        return WikiBackend(product=product)
    else:
        from rag_retriever import RAGRetriever
        return RAGRetriever(collection_name=...)
```

### 方式 3: 单 topic 临时用 Wiki

```python
from wiki_backend import get_wiki_backend

backend = get_wiki_backend("王者")
refs = backend.retrieve("李白的技能", num_comments=5)
# 用于补充 RAG 不足的某些话题
```

## 🎮 6 个产品

```python
PRODUCT_TO_WIKI_DIR = {
    "王者":           "wangzhe",       # 479 md, 128 英雄 + 皮肤 + 模式
    "王者荣耀":       "wangzhe",
    "王者世界":       "wangzhe-world",  # 75 md, 14 英雄 + 13 地图 + 4 皮肤
    "wangzhe-world":  "wangzhe-world",
    "洛克":           "luoke",          # 2894 md, 602 精灵 + 1775 道具
    "洛克王国":       "luoke",
    "金铲铲":         "jcczz",          # 671 md, 183 弈子 + 169 装备
    "金铲铲之战":     "jcczz",
    "无畏契约":       "valm",           # 153 md, 24 英雄 + 18 武器
    "DNF":            "dnf",            # 81 md, 60 职业 + 7 NPC + 30 副本
    "地下城与勇士":   "dnf",
}
```

## ⚙️ 性能调优

```python
backend = WikiBackend(
    product="王者",
    use_cache=True,       # 默认 True, 1 小时缓存
    timeout_sec=60,       # LLM 单次超时
    verbose=True,         # 打印 query 过程
)
```

缓存策略:`{product: {query: (ts, result)}}`,1 小时内同 query 直接复用。

## 🧪 对比测试

```bash
# RAG vs Wiki 对比 (10 题)
python compare_rag_vs_wiki.py --output compare_report.md
```

会输出每题:
- RAG 答案 / Wiki 答案
- 响应时间
- 来源追溯 (wiki 给 md 路径, RAG 给 chunk_id)
- 评分 (事实性 / 时效性 / 可读性)

## 📁 文件清单

| 文件 | 用途 |
|------|------|
| `wiki_backend.py` | **主文件**, drop-in 替代 RAGRetriever |
| `compare_rag_vs_wiki.py` | (待加) 对比测试脚本 |
| `WIKI_BACKEND_README.md` | (本文件) |

## ⚠️ 限制

1. **速度慢**: 每次 retrieve 10-30s (本地 Qwen ReAct), 评论生成如要求 1s 内完成不适用
2. **依赖本地 Qwen**: 需 llama.cpp `192.168.100.211:8080` 跑着
3. **缓存粒度粗**: 同一 product 同一 query 复用, 不会因 `direction` 变化
4. **数据规模**: 评论写手评论库可能更垂直 (玩家社群口吻), wiki 偏官方设定

## 💡 推荐用法

- **A/B 对比**: 同一话题跑 RAG 和 Wiki, 看哪个答案更贴玩家
- **混合**: RAG 拿玩家口吻素材 + Wiki 拿官方设定, 互补
- **慢批任务**: 离线生成大量评论时用 Wiki, 实时交互用 RAG

## 🔧 故障排查

```python
# 检查依赖
from wiki_backend import _WIKI_AVAILABLE, _IMPORT_ERR
print(_WIKI_AVAILABLE, _IMPORT_ERR)
# False 时: 确认 E:\deepsearch\ 存在 + scripts/query.py 可 import

# 检查本地 Qwen
import requests
r = requests.get("http://192.168.100.211:8080/health", timeout=5)
print(r.status_code)
# 非 200: Qwen 服务没起来
```
