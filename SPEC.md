# 评论写手系统 - 项目规范

## 1. 项目概述

- **项目名称**: 评论写手 (Comment Writer)
- **项目类型**: 基于LLM和RAG的智能评论生成系统
- **核心功能**: 从Milvus向量数据库检索相似评论作为参考，让LLM学习真实评论的说话风格和语气，批量生成像真人写的评论
- **目标用户**: 需要批量生成游戏/产品评论的用户

## 2. 技术架构

### 2.1 技术栈
- **向量数据库**: Milvus v2.4.0 (Docker容器, 端口19530)
- **嵌入模型**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: 支持OpenAI API兼容接口 (GPT-4o Mini等)
- **Python版本**: 3.13 (虚拟环境: commenter)
- **关键依赖**: pymilvus, sentence-transformers, openai, pandas, gradio, openpyxl

### 2.2 系统架构
```
用户输入 (话题 + 评论方向 + 数量)
       ↓
RAG检索模块 ←→ Milvus向量数据库
       ↓
LLM生成模块 (学习检索到的评论风格)
       ↓
批量生成评论输出
```

## 3. 数据结构

### 3.1 原始数据
- 位置: `D:/文案查重复.xlsx`
- 内容: 7399行 × 26列
- 格式: 交替排列的"互动量-评论"对

### 3.2 Milvus Collection: `wangzhe_comments`
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 | 主键，自增 |
| comment | VARCHAR(500) | 评论文本 |
| engagement | FLOAT | 互动量(阅读/点赞等) |
| embedding | FLOAT_VECTOR(384) | 句子嵌入向量 |

## 4. 功能模块

### 4.1 数据库构建模块
- [x] 读取Excel文件并提取评论
- [x] 文本清洗和预处理
- [x] 生成句子嵌入向量
- [x] 批量插入Milvus

### 4.2 RAG检索模块
- [x] 接收用户话题和评论方向
- [x] 检索Top-K相关评论（K = max(20, 评论数量×2)）
- [x] 去重和重排序

### 4.3 LLM生成模块
- [x] 构造Few-shot提示词
- [x] 去除"AI味"的核心策略：
  - 要求使用口语化表达
  - 禁止使用"首先、其次、综上所述"等AI典型词汇
  - 要求带有情绪和语气词
  - 要求像真人打字一样自然
- [x] 支持正性向/中性向/中正性向三种方向
- [x] 支持立场坚定但多角度思考

### 4.4 用户界面
- [x] Gradio Web界面
- [x] 支持设置话题、评论数量(1-100)、评论方向
- [x] 批量展示生成的评论

## 5. 核心Prompt设计

### 5.1 去除AI味的关键指令
```
你是一个真实的游戏玩家，你的任务是基于给定的参考评论，写出符合以下要求的评论：

1. 风格要求：
   - 必须像真人写的，不要像AI
   - 使用口语化表达，带有真实情绪
   - 可以有打字错误、重复字符等真实特征
   - 禁止使用"首先、其次、最后、综上所述、值得注意的是"等AI典型过渡词
   - 避免过于工整对仗的句式

2. 内容要求：
   - 围绕话题展开
   - 体现[正性向/中性向/中正性向]的评论方向
   - 可以有不同的视角和观点
   - 每条评论要有差异性

3. 格式要求：
   - 每条评论单独一行
   - 字数控制在50-150字之间
```

## 6. 目录结构
```
e:/评论写手/
├── SPEC.md                 # 项目规范
├── requirements.txt        # 依赖
├── .env.example           # 环境变量示例
├── config.py              # 配置文件
├── data_loader.py         # 数据加载模块
├── vector_store.py        # Milvus向量库操作
├── rag_retriever.py       # RAG检索模块
├── comment_generator.py   # LLM生成模块
├── app.py                 # Gradio界面
├── build_database.py      # 数据库构建脚本
└── run.py                 # 启动脚本
```

## 7. 配置参数

```python
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "wangzhe_comments"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20  # 默认检索数量
MAX_COMMENTS = 100  # 最大生成数量
```

## 8. 验收标准

- [x] Milvus容器正常运行
- [x] wangzhe_comments集合成功创建，包含27156条评论
- [x] 给定话题能检索出相关的真实评论
- [x] 生成的评论不包含明显AI特征
- [x] 支持1-100条评论的批量生成
- [x] 支持三种评论方向的生成
- [x] Web界面可正常运行和使用

## 9. 快速开始

### 9.1 环境准备
```bash
cd e:/评论写手
./commenter/Scripts/pip install -r requirements.txt
```

### 9.2 配置API密钥
创建 `.env` 文件：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### 9.3 启动系统
```bash
./commenter/Scripts/python run.py
```

访问 http://localhost:7860 使用系统
