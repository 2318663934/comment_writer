# Comment Writer — 混合检索评论生成系统

> 🎯 **LAN 部署**: 本机 Wiki + 服务器 RAG,通过 HTTP 互通

基于 Milvus 向量检索 + LLM-Wiki 知识库的混合检索评论生成系统。支持多产品 (王者荣耀/洛克王国/金铲铲/DNF/无畏契约) 评论自动生成,头脑风暴多角度,以及产品知识库的灵活切换。

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    Web 服务层                          │
│   Flask app.py: 5 大路由                              │
│   · /         主页                                    │
│   · /chat     多轮 chat 界面                          │
│   · /onboard  反馈协同入库                            │
│   · /review   99-待审 UI                              │
│   · /raw      wiki/ 浏览器                            │
└──────────┬──────────────────────┬────────────────────┘
           │                      │
┌──────────▼──────────┐  ┌───────▼────────────────────┐
│   CommentGenerator   │  │   WikiProductBackend       │
│   评论生成引擎        │  │   产品信息后端 (混合模式)    │
│   · 头脑风暴         │  │   · local / http / auto     │
│   · 风格迁移         │  │   · 调本机 LLM-Wiki API      │
│   · 多产品支持       │  │   · drop-in 替代 Milvus     │
└──────────┬──────────┘  └───────┬────────────────────┘
           │                      │
┌──────────▼──────────────────────▼────────────────────┐
│   混合检索层 (Hybrid Retrieval) ⭐                    │
│   ┌─────────────────────────┐  ┌──────────────────┐  │
│   │ WikiProductBackend       │  │  Milvus RAG       │  │
│   │ (调本机 LLM-Wiki 知识库)  │  │  (真实评论库)     │  │
│   │ · HTTP 调本机 :8088      │──▶  fallback / hybrid│  │
│   │ · 精确路径快查 0.5s      │  │                    │  │
│   │ · 6 产品 4353md          │  │  wangzhe_comments: │  │
│   │   (王/王世/洛克/jcczz/  │  │  25,738 条真实评论 │  │
│   │    dnf/valm)            │  │                    │  │
│   └─────────────────────────┘  └──────────────────┘  │
│   KNOWLEDGE_MODE: rag / wiki / hybrid (默认)          │
└──────────┬───────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│   数据持久层                                          │
│   · Milvus (Docker): 评论库 / 产品分析库              │
│   · Flask session: 多轮 chat 历史                    │
│   · strategy_cases/: 写作策略案例                     │
└──────────────────────────────────────────────────────┘

外部依赖 (本机):
┌──────────────────────────────────────────────────────┐
│  Wiki API Server (端口 8088)                         │
│  · 复用 e:/deepsearch/wiki/ 知识库 (4353 md)         │
│  · Bearer token 鉴权 + 60 req/min/IP 限速             │
│  · 配套: e:/行业稿件写作 共享                          │
└──────────────────────────────────────────────────────┘
```

## 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **app.py** | `app.py` | Flask 主服务, 5 大路由 + CSRF + Lock |
| **comment_generator.py** | `comment_generator.py` | 评论生成核心 (含 hybrid 检索) |
| **wiki_product_backend.py** | `wiki_product_backend.py` | ⭐ Wiki drop-in 后端 |
| **rag_retriever.py** | `rag_retriever.py` | Milvus 评论库检索器 |
| **vector_store.py** | `vector_store.py` | Milvus 向量存储 |
| **multimodal_extractor.py** | `multimodal_extractor.py` | 智谱多模态 (图片/视频) 提取 |
| **data_loader.py** | `data_loader.py` | 评论数据加载 |
| **config.py** | `config.py` | 系统配置 (含 KNOWLEDGE_MODE) |
| **.env** | `.env` | 环境变量 (含 API 密钥) |

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker (运行 Milvus)
- 本机: 访问 `e:/deepsearch` 知识库 (LAN 局域网)

### 2. 安装依赖

```bash
# 克隆仓库
git clone git@github.com:xiaobaiaigroup/comment_writer.git
cd comment_writer

pip install -r requirements.txt   # 含 flask / pymilvus / requests / 等
```

### 3. 启动 Milvus

```bash
cd milvus
docker-compose up -d
```

### 4. 配置 .env

```bash
# OpenAI 兼容 (实际可指向 DeepSeek)
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-pro

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 智谱多模态 (可选, 用于图片/视频信息提取)
ZHIPU_API_KEY=xxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4.6v-flash
```

### 5. 启动本机 Wiki API Server

```bash
# 在 e:/deepsearch 目录 (本机):
python wiki_api_server.py --host 0.0.0.0 --port 8088 --api-key deepsearch123
```

### 6. 启动评论写手服务

```bash
# 方式 1: 直接启动
python app.py

# 方式 2: 指定 KNOWLEDGE_MODE
KNOWLEDGE_MODE=hybrid python app.py   # 默认
KNOWLEDGE_MODE=rag python app.py      # 纯 RAG (原行为)
KNOWLEDGE_MODE=wiki python app.py     # 纯 Wiki (强制本机)
```

## 混合检索: LLM-Wiki + Milvus 评论库 (2026-06 新增)

评论写手在头脑风暴阶段需要产品上下文, 既保留原 Milvus 产品信息库作为兜底, 又支持切换到本机 LLM-Wiki 知识库。

### 3 种模式

| 模式 | 行为 | 适用 |
|------|------|------|
| `rag` | 纯 Milvus 评论库 (原行为) | 服务器无 Wiki API |
| `wiki` | 纯 LLM-Wiki (强制本机) | 数据迁移期 |
| `hybrid` (默认) | Wiki 优先, 失败/空降级 RAG | **推荐** |

### 配置 Wiki HTTP 模式

```bash
# 在 .env 加:
WIKI_API_URL=http://192.168.x.x:8088     # 本机 IP
WIKI_API_KEY=deepsearch123
WIKI__KNOWLEDGE_MODE=hybrid
WIKI__TIMEOUT_SEC=60
```

> 详细 LAN 部署: 见 [LAN_DEPLOY_README.md](LAN_DEPLOY_README.md)
>
> Wiki 后端对比: 见 [WIKI_BACKEND_README.md](WIKI_BACKEND_README.md)

### 部署模式

| 部署 | Wiki API URL | 备注 |
|------|-------------|------|
| **本机直跑** | (不设) | `_resolve_backend` 自动用 `local` |
| **服务器部署** | `http://192.168.x.x:8088` | 走 HTTP, 服务器无需装 Qwen/embedding |
| **混合 (本机+服务器)** | 都设 | 灵活切换 |

### 接口 (drop-in 替代 `ProductKnowledgeBase`)

```python
# 原 RAG 模式:
from crawler.product_retriever import ProductKnowledgeBase
kb = ProductKnowledgeBase()
results = kb.search("星光对决 莫扎特", product="lok_world", top_k=10)

# 新 Wiki 模式 (无侵入):
from wiki_product_backend import search as kb_search  # 同名函数
results = kb_search("星光对决 莫扎特", product="洛克王国世界", top_k=10)
# 两种接口完全一致: search(query, product, top_k) -> List[Dict]
# 返回字段: product / product_display / id / title / url / source / date
#         / content_text / content_length / distance
```

### 性能数据

| 查询 | 旧 (ReAct) | 新 (精确路径) |
|------|----------|------------|
| 星光对决 (事件) | 23.78s + fallback | **<0.01s** ⚡ |
| 王者荣耀 李白 | 23s | **<0.01s** ⚡ |
| 金铲铲薇恩 | 23s | **<0.01s** ⚡ |
| 一般问题 (未匹配) | 23s | 23s (走 ReAct) |

精确路径快查命中率 70%+, 大幅提升响应速度。

## 产品覆盖

| 产品 | Milvus Collection | Wiki 路径 |
|------|-------------------|----------|
| 王者荣耀 | `honor_of_kings` | `wiki/wangzhe/` |
| 王者荣耀:世界 | `honor_of_kings_world` | `wiki/wangzhe-world/` |
| 洛克王国世界 | `lok_world` | `wiki/luoke/` |
| 金铲铲之战 | `jcjz` | `wiki/jcczz/` |
| DNF 端游 | `dnf` | `wiki/dnf/` |
| 无畏契约手游 | `wxqy` | `wiki/valm/` |

## 风格参考库

| 来源 | 数量 | 用途 |
|------|------|------|
| 真实游戏评论 | 25,738 条 | 风格迁移 |
| strategy_cases/ | 100+ 标注 | 写作策略范本 |
| SKILL.md | 双视角蒸馏 | 句式/开篇/结尾 |

## ⚠️ 安全提示

`.env` 包含 API 密钥 (OpenAI / 智谱), 已在历史 commit 中 (commit `85902a6`)。**建议**:

1. 立即在 OpenAI / 智谱后台**轮换所有密钥**
2. 加 `.env` 到 `.gitignore` (尚未加)
3. `git rm --cached .env` + commit, 然后**重新写一份 .env.example 模板**

## 命令行

```bash
# 单元测试 Wiki 后端
python wiki_product_backend.py

# 启动服务
python app.py

# 比较 4 种检索模式
python compare_wiki_only.py
python compare_product_kb.py
python compare_mixed_kb.py
python compare_event_comments.py
```

## 文档清单

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | 本文件 (主入口) |
| [LAN_DEPLOY_README.md](LAN_DEPLOY_README.md) | LAN 部署详细方案 |
| [WIKI_BACKEND_README.md](WIKI_BACKEND_README.md) | Wiki 后端对比测试 |
| [使用手册.md](使用手册.md) | 用户使用手册 |
| [SKILL.md](SKILL.md) | 写作风格指令集 |
| [SPEC.md](SPEC.md) | 系统规格说明 |
| [compare_*.md](compare_wiki_only.md) | 4 种检索模式对比 |
| [test_e2e_hybrid_output.md](test_e2e_hybrid_output.md) | 端到端 hybrid 输出样例 |

## License

Internal Use
