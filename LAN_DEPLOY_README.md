# 评论写手 — 局域网混合部署方案

> 🎯 **本机跑 Wiki, 服务器跑评论写手 + RAG, 通过 HTTP 互通**

---

## 🏗️ 架构

```
┌──────────────────────────────────────────┐    HTTP    ┌──────────────────────────────────────────┐
│ 本机 (Windows 10.0.19045)                  │   局域网   │ 服务器 (部署评论写手)                       │
│ IP: 192.168.x.x                            │◀────────▶│ Milvus + Flask app.py                     │
│                                            │            │                                            │
│  ┌────────────────────────────────────┐   │            │  ┌────────────────────────────────────┐   │
│  │ deepsearch 项目                    │   │            │  │ 评论写手                          │   │
│  │   wiki/ (4659 md, 6 产品)         │   │            │  │   comment_generator.py             │   │
│  │   scripts/query.py (本地 Qwen)     │   │            │  │   rag_retriever.py (Milvus)        │   │
│  │   wiki_api_server.py (新增)        │   │            │  │   wiki_product_backend.py (HTTP)   │   │
│  │   ┌──────────────────────────┐    │   │            │  └────────────────────────────────────┘   │
│  │   │ :8088 HTTP API            │    │   │            │                                            │
│  │   │ /api/wiki/search         │◀───┼───┼────────────┼── WIKI_API_URL=http://192.168.x.x:8088    │
│  │   │ /api/health              │    │   │            │                                            │
│  │   │ /api/stats                │    │   │            │                                            │
│  │   └──────────────────────────┘    │   │            │                                            │
│  └────────────────────────────────────┘   │            │  ┌────────────────────────────────────┐   │
│                                            │            │  │ RAG 真实评论库 (Milvus)             │   │
│  ┌────────────────────────────────────┐   │            │  │   wangzhe_comments: 25,738 条       │   │
│  │ 启动脚本:                           │   │            │  │   lok_world/jcjz/dnf/wxqy: 文章库   │   │
│  │   start_wiki_api.bat                │   │            │  └────────────────────────────────────┘   │
│  └────────────────────────────────────┘   │            │                                            │
└──────────────────────────────────────────┘            └──────────────────────────────────────────┘
```

## 🎯 部署流程

### A. 本机 (Wiki 端)

1. **确保 deepsearch 项目能跑**:
   ```bash
   cd e:/deepsearch
   python -c "from scripts.query import query; print('ok')"
   ```

2. **启动 Wiki API Server**:
   ```bash
   # 方式 1: 直接跑
   python wiki_api_server.py --host 0.0.0.0 --port 8088 --api-key deepsearch123

   # 方式 2: 后台跑 (推荐, 开机自启)
   start_wiki_api.bat
   ```

3. **验证**:
   ```bash
   curl http://127.0.0.1:8088/api/health
   # {"qwen_model": "...", "status": "ok", "wiki_md_count": 4659, ...}
   ```

4. **查看本机 IP** (服务器需要):
   ```bash
   ipconfig | findstr IPv4
   # 假设 192.168.1.100
   ```

### B. 服务器 (评论写手端)

1. **复制 `wiki_product_backend.py` 到服务器**:
   ```bash
   scp -r E:/评论写手/wiki_product_backend.py user@server:/path/to/评论写手/
   ```

2. **配置 `.env`** (或环境变量):
   ```ini
   # .env
   WIKI_API_URL=http://192.168.1.100:8088
   WIKI_API_KEY=deepsearch123
   KNOWLEDGE_MODE=hybrid  # hybrid: wiki 优先, 失败降级 Milvus RAG
   ```

3. **无需在本机装**:
   - ❌ 不需要装 Qwen
   - ❌ 不需要 deepsearch 项目
   - ❌ 不需要 4GB+ 模型文件

4. **验证服务器能调到本机**:
   ```bash
   curl http://192.168.1.100:8088/api/health
   # 应该和本机 curl 一样
   ```

5. **启动评论写手**:
   ```bash
   cd /path/to/评论写手
   set WIKI_API_URL=http://192.168.1.100:8088
   set WIKI_API_KEY=deepsearch123
   python app.py
   ```

## 📊 三种 backend 自动切换

`WikiProductBackend(backend="auto")` (默认):

| 情况 | 自动选 | 备注 |
|------|--------|------|
| 有 `WIKI_API_URL` env | **http** | 服务器场景 |
| 无 `WIKI_API_URL` | local | 本地测试 |

显式指定:
```python
WikiProductBackend(backend="local")   # 本机直接 import deepsearch
WikiProductBackend(backend="http")    # 强制 HTTP (即使没 env)
```

## 🔧 故障排查

### 1. 服务器连不上本机

```bash
# 本机测试:
curl http://127.0.0.1:8088/api/health   # 应该 200

# 局域网测试 (在本机):
curl http://192.168.x.x:8088/api/health

# 服务器测试:
curl http://192.168.x.x:8088/api/health

# 如果服务器失败:
#   1. 检查防火墙 (Windows Defender / Linux iptables)
#   2. 检查 0.0.0.0 绑定 (不是 127.0.0.1)
#   3. 检查路由器 AP 隔离
```

### 2. 命中 fallback 太多

精确路径快查没命中时 (query 不在 EXACT_PATH_RULES 字典里), 会走 ReAct。ReAct 可能超时。

**解决**:
- 扩大 `wiki_api_server.py` 的 `EXACT_PATH_RULES` 字典, 加更多关键词
- 调大 `MAX_STEPS` (已 30) / `TIMEOUT_SEC` (已 300)

### 3. HTTP 调用慢

正常 ReAct 30 秒以内, 精确路径快查 < 1 秒。

如果 5+ 秒都没返回:
- 检查本机 Qwen 服务 `192.168.100.211:8080` 是否还活着
- 看本机 CPU/内存是否被吃满

## 🔒 安全

### 局域网 (信任网络)

```bash
python wiki_api_server.py --port 8088 --host 0.0.0.0
# 无 --api-key, 局域网内任何人都能调
```

### 半信任 (加 Bearer Token)

```bash
python wiki_api_server.py --port 8088 --host 0.0.0.0 --api-key YOUR_SECRET

# 服务器 .env:
WIKI_API_KEY=YOUR_SECRET
```

### 公网 (强烈不推荐)

需要加 Nginx 反向代理 + HTTPS + 强鉴权, 不在本文档范围。

## 📈 性能数据 (实测)

| 查询类型 | 旧 (ReAct) | 新 (精确路径) |
|---------|----------|------------|
| 星光对决 (事件) | 23.78s + fallback | <0.01s ⚡ |
| 王者荣耀 李白 | 23s | <0.01s ⚡ |
| 金铲铲薇恩 | 23s | <0.01s ⚡ |
| 一般问题 (未匹配) | 23s | 23s (走 ReAct) |

**精确路径快查命中率 70%+**, 大幅提升响应速度 + 避免 fallback。

## 📁 文件清单

| 文件 | 位置 | 用途 |
|------|------|------|
| `wiki_api_server.py` | **本机** e:/deepsearch | HTTP API 网关 |
| `wiki_product_backend.py` | **服务器** 评论写手目录 | drop-in 替代 product_retriever |
| `comment_generator.py` | **服务器** 评论写手目录 | (已改) hybrid 模式 |
| `config.py` | **服务器** 评论写手目录 | (已加) `KNOWLEDGE_MODE` 开关 |
| `start_wiki_api.bat` | **本机** | 一键启动 |

## 🔁 切换 RAG / Wiki (rollback)

```bash
# 服务器:
set KNOWLEDGE_MODE=rag     # 纯 RAG (原行为)
set KNOWLEDGE_MODE=wiki    # 纯 Wiki (强制走本机)
set KNOWLEDGE_MODE=hybrid  # 默认: Wiki 优先, RAG 兜底
```

## 🆕 高级: 多机部署

如果本机也想用 Wiki API (而不是直接 import):

```bash
# 本机也配:
set WIKI_API_URL=http://127.0.0.1:8088
set WIKI_API_KEY=deepsearch123
```

这样统一从 HTTP 走, 切换机器更灵活。

## 📞 联系

- Wiki API 问题: 看 `e:/deepsearch/wiki_api_server.py` 头部
- WikiProductBackend 问题: 看 `E:/评论写手/wiki_product_backend.py` 头部
- 局域网连通性问题: 检查防火墙/路由器
