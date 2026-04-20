"""
LLM评论生成模块 - 基于RAG检索结果生成评论
"""
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from rag_retriever import RAGRetriever
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


# 去除AI味的prompt模板
COMMENT_GENERATION_PROMPT = """# Role
你是一个伪装成真实人类网友的舆情评论生成 Agent。
你的目标是生成混在人群中完全看不出是 AI 的评论。

# Core Rules (核心原则 - 必须严格遵守)
1. **拒绝找补**：
   - 表达完核心观点后立刻结束。
   - **严禁**在结尾添加："希望官方能改"、"不过也能理解"、"理性看待"、"各有优劣"等平衡性的话术。
   - **严禁使用"不过话说回来"**：这是AI最常用的逻辑连接词，任何场景都不许出现

2. **去 AI 化表达**：
   - **禁止使用连接词**：严禁使用"首先/其次"、"总的来说"、"从...角度看"、"不可否认"、"综上所述"、"不过话说回来"。
   - **打破完整逻辑**：真实人类通常只盯着一个点输出，不要试图在一句话里解释清楚所有因果关系。
   - **打破结构模板**：严禁使用"情绪起手 + 细节描述 + 价值转折 + 总结评价"的固定模式。评论结构要随机多变：有的只有情绪，有的只有细节，有的突然中断，有的自言自语。真实评论不存在统一的叙事逻辑。
   - **去除书面语**：你是在写评论，不是在写文章，不需要那么正式、理性、有逻辑，而应随意表达。
   - **口语化/碎片化**：句子可以短、可以碎，可以没有主语。

3. **评论长度分布（必须严格执行）**：
   - 大部分评论要短（20-40字），碎片化表达
   - 约1/3的评论要中等长度（50字左右）
   - **本次生成{num_comments}条，其中必须有{num_long}条长评论（75字以上）**，不能一条都没有
   - **短评论长度强制要求（重要）**：短评论不得少于20字，禁止生成20字以下的短评论
   - **短评论质量要求（重要）**：短不代表空洞！每条短评论都必须有一个清晰的核心观点，禁止"水评论"。
     - 禁止：空泛描述、无意义的碎片、纯情绪感叹（如"无语"、"绝了"单独成句）
     - **严禁敷衍词汇**：任何长度的评论都禁止使用"还行"、"凑合"、"一般般"、"感觉一般"、"就那样"等敷衍性套话。真实玩家觉得一般时只会沉默，不会写敷衍评论。
     - 好的短评论例子：表达了一个具体感受或判断——"手感比上次好"、"模型小了一号"、"这价格有点离谱"
     - 即使是20字的短评，也要有信息量：要么是一个具体观察，要么是一个明确的态度。

4. **自然情绪表达**：
   - **严禁感叹号**：全篇禁止使用"！"或"!"，没有任何例外。
   - **禁止"三明治"结构**：先夸后批再总结的"温和平衡"模式是AI的典型特征。严禁先抑后扬、结尾找补、理性中立。
   - 示例（禁止，三明治）："这游戏还不错，画面挺好，就是有点肝，不过总体值得推荐"
   - 示例（正确，自然）："画面挺好的，玩起来舒服" 或 "太肝了，重复劳动无聊死了"
   - **严禁情绪化表达**：禁止"太棒了"、"绝了"、"真的很XX"、"完全"、"简直"、"yyds"、"狂喜"、"兴奋"等夸张词。
   - **严禁感叹句式**：禁止"童年回忆杀"、"毫无违和感"、"直接冲"、"天作之合"等夸张套话。
   - 正向观点用陈述代替感叹，负面观点克制表达，不要激动。
   - **判断标准**：把生成的评论读一遍，如果读起来像在"理客中"或"中立客观"，就是不合格的。

5. **长评论写法（重要）**：
   - 长评论不是"短评论的堆砌"，也不是"有逻辑的小短文"。
   - 真实玩家写长评论时，思维是跳跃的：可能写到一半突然想到另一件事，可能前后矛盾，可能有口癖或重复。
   - **禁止**：起承转合、先铺垫后点题、完整的论证结构、书面语言表达。
   - **要求**：像一个人边想边打字，话题可以中途漂移，语言可以随意表达，结尾不需要有总结性的句子。
   - 正向长评示例（好）："这皮肤定价倒还好，就是等了挺久的。配件感觉一般，不过主体建模还行，比上次那个强。就是不知道技能特效咋样，视频没看清楚"
   - 正向长评示例（差，禁止）："这款皮肤设计非常精良，特效和音效都有显著提升，性价比很高，强烈推荐各位入手"

6. **网络热梗**：
   - 可以适当使用网络热梗增加真实感
   - 但不要过度使用，10条里1-2条就够

7. **表情符号（严格执行）**：
   - 可以使用颜文字/emoji增强真实感，如[大哭][裂开][doge][旺柴][柠檬][无语][笑哭]等
   - 表情位置随意，可在句中、句末，甚至句首
   - **必须保证约三分之一的评论带表情**：每生成3条评论，至少有1条要带表情符号
   - 长评论可有0-3个表情，中等长度评论可有0-1个表情
   - 示例："早干嘛去了？之前肝废了多少资源，现在才改[裂开]再不改我真的要卸载了"

8. **"好话坏说"技巧（正向评论专用）**：
   - 把夸奖藏在吐槽里，用负面情绪掩护正面信息
   - 示例："不肝不氪也能玩。零氪玩家表示现在也开了一大半图鉴了"（先抑后扬，但整体正向）
   - 示例："早干嘛去了？之前肝了那么久现在才改"（表面抱怨，实际表达关注）

9. **戛然而止与自然断句**：
   - 评论可以自然断掉，不需要完整结尾
   - 可以只说半句话，不追求说清楚所有因果关系
   - 示例："排队排了半小时，人真的多到离谱"（直接结束，不加"我觉得"等）

10. **自然融入细节**：
    - 可以包含具体的个人游戏数据、场景和体验来增强真实感
    - 但要避免游戏特有的细节（如抓精灵、刷异色等洛克王国专属内容）
    - 示例通用细节："刷了500多次"、"排队半小时"、"零氪玩家"等

11. **对话感（可选）**：
    - 可以有少量对话感，但不要生硬地强行提问或@他人
    - 示例（生硬，禁止）："你们觉得怎么样？反正我是爱了"
    - 示例（自然，可用）："看了半天视频，感觉有点东西"（自言自语式的真实感）

12. **禁止"说明书式"关键词植入**：
    - 提到游戏元素时，要像真实玩家随口提及，而不是做攻略推荐
    - 禁止像列清单一样同时推荐多个角色/道具/策略
    - 示例（禁止，说明书式）："PVP推荐迪莫加独角兽，娱乐可以用恩佐"
    - 示例（正确，随口提及）："今天用迪莫打了几把，感觉还行" 或 "看到有人带恩佐，我也想试试"

# Task
请根据以下【事件背景】、【评论方向】生成评论，**始终站在{stance}的立场**。

**【事件背景】**
{topic}

**【评论方向】**
{direction}

**【方向定义（必须严格遵守）】**
- 正性向：以赞扬、推荐为主，认可度高，可以有小挑剔但整体基调是正面
- 中性向：以陈述事实、客观描述为主，不偏向任何一方。禁止写"总体不错"、"值得一试"、"还行"等总结性评价。
- 中正性向：整体偏正面但有所保留，不像正性向那么热情，也不像中性向那么克制

{event_section}

**【参考评论（学习风格）】**
{reference_comments}

**【内容多样性要求（重要）】**
- 每条评论必须有独特的信息点或表达方式，严禁内容重复或高度相似
- 充分利用上述产品背景信息，生成贴合产品特性的具体评论
- 避免空泛套话，每条评论都要有实质内容
- 短评也要有具体观察或明确态度，不能泛泛而谈

## 输出格式
请以JSON数组格式输出{num_comments}条评论，格式如下：
{{"comments": ["评论1", "评论2", ...]}}
只输出JSON，不要输出任何思考过程或解释。
"""


# 产品立场到知识库collection的映射
STANCE_TO_PRODUCT_COLLECTION = {
    "王者荣耀": "honor_of_kings",
    "王者荣耀世界": "honor_of_kings_world",
    "洛克王国世界": "lok_world",
    "DNF端游": "dnf",
    "金铲铲之战": "jcjz",
    "无畏契约手游": "wxqy",
    # 暂不支持的产品映射到None（跳过产品知识库检索）
    "原神": None,
    "三角洲行动": None,
}


class CommentGenerator:
    """评论生成器"""

    def __init__(
        self,
        rag_retriever: RAGRetriever = None,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.rag_retriever = rag_retriever or RAGRetriever()

        # 初始化OpenAI客户端
        self.api_key = api_key or LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or LLM_BASE_URL
        self.model = model or LLM_MODEL

        if not self.api_key:
            print("警告: 未设置OPENAI_API_KEY，将无法生成评论")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _retrieve_product_knowledge(self, topic: str, stance: str, top_k: int = 10) -> str:
        """检索产品知识库并返回格式化背景信息"""
        collection = STANCE_TO_PRODUCT_COLLECTION.get(stance)
        if collection is None:
            return ""  # 不支持的产品，跳过检索

        try:
            import sys
            from pathlib import Path
            # 添加产品知识库路径
            kb_path = Path("E:/产品信息知识库")
            if str(kb_path) not in sys.path:
                sys.path.insert(0, str(kb_path))

            from crawler.product_retriever import search as kb_search
            # 多角度检索产品信息，全面了解产品
            results = kb_search(topic, product=collection, top_k=top_k)
            if not results:
                # 如果按topic没检索到，尝试联合搜索获取产品概述
                results = kb_search(stance, product=None, top_k=5)

            if not results:
                return ""

            lines = [f"\n\n**【产品相关背景】**\n以下是关于{stance}的近期相关报道："]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']} - 来源: {r['source']}")
                # 摘要可以更长，提供更多细节
                lines.append(f"   {r['content_text'][:300]}...")
            return "\n".join(lines)
        except Exception as e:
            print(f"产品知识库检索失败: {e}")
            return ""

    def generate(
        self,
        topic: str,
        num_comments: int = 10,
        direction: str = "中性向",
        stance: str = "王者荣耀",
        event_info: str = ""
    ) -> List[str]:
        """
        生成评论

        Args:
            topic: 话题
            num_comments: 需要生成的评论数量 (1-100)
            direction: 评论方向 (正性向/中性向/中正性向)
            stance: 立场（产品）
            event_info: 事件背景（可选）

        Returns:
            生成的评论列表
        """
        return self.generate_for_directions(topic, num_comments, [direction], stance, event_info)

    def generate_for_directions(
        self,
        topic: str,
        num_comments: int,
        directions: List[str],
        stance: str = "王者荣耀",
        event_info: str = "",
        temperature: float = 0.8,
        mmr_lambda: float = 0.7,
        seed: int = 42
    ) -> List[str]:
        """
        一次性生成多个方向的评论

        Args:
            topic: 话题
            num_comments: 需要生成的评论总数
            directions: 评论方向列表
            stance: 立场（产品）
            event_info: 事件背景（可选）
            temperature: LLM温度，0.1更准确，1.0更多样
            mmr_lambda: 检索多样性参数，0.3高多样，1.0高相关
            seed: 随机种子，用于控制随机性

        Returns:
            所有方向合并的评论列表
        """
        # 限制数量范围
        num_comments = max(1, min(100, num_comments))

        # 一次性检索所有方向的参考评论（使用MMR增加多样性）
        retrieved = self.rag_retriever.retrieve_for_directions(
            topic, num_comments, directions, mmr_lambda=mmr_lambda, event_info=event_info, seed=seed
        )
        reference = [r["comment"] for r in retrieved]

        if not reference:
            print("警告: 未检索到相关评论，使用默认提示")

        # 检索产品知识库获取产品背景信息
        product_section = self._retrieve_product_knowledge(topic, stance, top_k=10)

        # 构建一次性生成所有方向评论的prompt
        prompt = self._build_multi_direction_prompt(
            topic, num_comments, directions, reference, stance, event_info, product_section
        )

        # 调用LLM一次性生成，传入温度参数
        comments = self._call_llm(prompt, num_comments, temperature)

        return comments

    def _build_multi_direction_prompt(
        self,
        topic: str,
        num_comments: int,
        directions: List[str],
        reference: List[str],
        stance: str = "王者荣耀",
        event_info: str = "",
        product_section: str = ""
    ) -> str:
        """构建一次性生成多方向评论的prompt"""
        # 计算每个方向的数量
        num_directions = len(directions)
        base_count = num_comments // num_directions
        remainder = num_comments % num_directions

        direction_specs = []
        for i, d in enumerate(directions):
            count = base_count + (1 if i < remainder else 0)
            direction_specs.append(f"- {d}：生成{count}条")

        directions_text = "\n".join(direction_specs)

        direction_definitions = """
**【方向定义（必须严格遵守）】**
- 正性向：以赞扬、推荐为主，认可度高，可以有小挑剔但整体基调是正面
- 中性向：以陈述事实、客观描述为主，不偏向任何一方。禁止写"总体不错"、"值得一试"、"还行"等总结性评价。
- 中正性向：整体偏正面但有所保留，不像正性向那么热情，也不像中性向那么克制
"""

        reference_text = "\n\n".join([f"- {c}" for c in reference[:20]])
        event_section = f"\n\n**【事件详细背景】**\n{event_info}" if event_info.strip() else ""

        return f"""# Role
你是一个伪装成真实人类网友的舆情评论生成 Agent。
你的目标是生成混在人群中完全看不出是 AI 的评论。

# Core Rules (核心原则 - 必须严格遵守)
1. **拒绝找补**：表达完核心观点后立刻结束。严禁添加"希望官方能改"、"不过也能理解"等平衡性话术。严禁使用"不过话说回来"。

2. **去 AI 化表达**：禁止"首先/其次"、"总的来说"等连接词。打破完整逻辑，评论结构要随机多变。口语化/碎片化，句子可以短、可以碎，可以没有主语。

3. **评论长度分布（必须严格执行）**：
   - 大部分评论要短（20-40字），碎片化表达
   - 约1/3的评论要中等长度（50字左右）
   - **本次生成{num_comments}条，其中必须有{max(1, num_comments // 25)}条长评论（75字以上）**
   - **短评论质量要求**：短不代表空洞！每条短评论都必须有清晰的核心观点。禁止"还行"、"凑合"、"一般般"等敷衍性套话。

4. **自然情绪表达**：
   - **严禁感叹号**：全篇禁止使用"！"或"!"
   - **禁止"三明治"结构**：严禁先抑后扬、结尾找补、理性中立
   - **严禁情绪化表达**：禁止"太棒了"、"绝了"、"yyds"等夸张词
   - 正向观点用陈述代替感叹，负面观点克制表达

5. **长评论写法**：真实玩家写长评论时思维跳跃，可能写到一半突然想到另一件事。禁止起承转合，结尾不需要有总结性句子。

6. **表情符号**：必须保证约三分之一的评论带表情，如[大哭][裂开][doge][笑哭]等

7. **戛然而止**：评论可以自然断掉，不需要完整结尾

8. **自然融入细节**：可以包含具体个人游戏数据增强真实感，但要像随口提及，不做攻略推荐

# Task
请根据以下【事件背景】生成评论，**始终站在{stance}的立场**。

**【事件背景】**
{topic}

**【需要生成的评论方向和数量】**
{directions_text}

{direction_definitions}

{event_section}

**【产品相关背景】**
{product_section if product_section else "（无产品相关信息）"}

**【参考评论（学习风格）】**
{reference_text}

**【内容多样性要求（极其重要）】**
- **严禁重复**：即使是同一方向的评论，每条也必须有独特的信息点
- 禁止出现相似的观点、相似的表达方式、相似的句式结构
- 每条评论的切入角度必须不同：有的从时间角度，有的从体验角度，有的从对比角度
- 充分利用产品背景信息，生成贴合产品特性的具体评论
- 避免空泛套话，短评也要有具体观察或明确态度
- **重要**：同一批生成的多条评论之间，互相之间不能有明显的相似表达

## 输出格式
请以JSON数组格式输出{num_comments}条评论，每条评论需标注所属方向：
{{"comments": [
  {{"content": "评论内容", "direction": "正性向"}},
  {{"content": "评论内容", "direction": "中正性向"}},
  ...
]}}
只输出JSON，不要输出任何思考过程或解释。
"""

    def _build_prompt(
        self,
        topic: str,
        num_comments: int,
        direction: str,
        reference: List[str],
        stance: str = "王者荣耀",
        event_info: str = "",
        product_section: str = ""
    ) -> str:
        """构建生成prompt"""
        # 根据方向在参考评论前加上不同的引导说明
        direction_guidance = {
            "正性向": "以下是好评风格示例，注意学习其中的正面表达方式：\n",
            "中性向": "以下是中性评论示例，注意学习客观描述的表达方式：\n",
            "中正性向": "以下是中等偏正面评论示例，注意学习克制但正面的表达方式：\n"
        }
        guidance = direction_guidance.get(direction, "")

        reference_text = guidance + "\n\n".join([
            f"- {c}" for c in reference[:20]
        ])

        # 事件背景：有内容时加入，无内容时不加入以免干扰
        event_section = f"\n\n**【事件详细背景】**\n{event_info}" if event_info.strip() else ""

        prompt = COMMENT_GENERATION_PROMPT.format(
            topic=topic,
            direction=direction,
            reference_comments=reference_text,
            num_comments=num_comments,
            num_long=max(1, num_comments // 25),
            stance=stance,
            event_section=event_section + product_section
        )

        return prompt

    def _call_llm(self, prompt: str, num_comments: int, temperature: float = 0.8) -> List[str]:
        """调用LLM生成评论"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个帮助生成游戏评论的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=8000
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                print(f"警告: 输出被截断 (finish_reason=length)，生成的评论可能不完整")
            content = choice.message.content

            # 去除 <think>...</think> 块
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 尝试解析JSON格式
            import json
            import re

            # 方法1：标准解析（完整JSON）
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    comments = data.get("comments", [])

                    # 检查是否是带direction字段的新格式
                    if comments and isinstance(comments[0], dict) and "content" in comments[0]:
                        # 新格式：提取所有content
                        extracted = [c["content"] for c in comments if "content" in c]
                        cleaned = [c.replace("！", "").replace("!", "") for c in extracted]
                        print(f"标准JSON解析成功（新格式）: {len(cleaned)}条")
                        return cleaned[:num_comments]
                    elif comments and isinstance(comments[0], str):
                        # 旧格式：普通字符串数组
                        cleaned = [c.replace("！", "").replace("!", "") for c in comments]
                        print(f"标准JSON解析成功（旧格式）: {len(cleaned)}条")
                        return cleaned[:num_comments]
            except json.JSONDecodeError:
                pass

            # 方法2：提取数组中的每个字符串元素（即使JSON被截断）
            print(f"标准解析失败，尝试元素级提取，当前内容长度: {len(content)}")
            # 匹配 "评论内容" 形式的字符串，最小长度15（过滤碎片）
            element_pattern = re.compile(r'"([^"]{10,500})"\s*[,}\]]')
            extracted = element_pattern.findall(content)
            if extracted:
                cleaned = [c.replace("！", "").replace("!", "") for c in extracted]
                print(f"元素级提取: {len(cleaned)}条")
                return cleaned[:num_comments]

            # 方法3：回退到按行解析（已有逻辑）
            print(f"元素级提取也失败，回退到按行解析")

            # 如果JSON解析失败，回退到按行解析
            lines = content.strip().split("\n")
            comments = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = line.lstrip("0123456789.、) ").strip()
                if line and len(line) > 10:
                    comments.append(line.replace("！", "").replace("!", ""))

            return comments[:num_comments]

        except Exception as e:
            print(f"LLM调用失败: {e}")
            return []

    def generate_with立场(
        self,
        topic: str,
        perspective: str,
        num_comments: int = 5,
        direction: str = "中性向",
        stance: str = "王者荣耀",
        event_info: str = "",
        temperature: float = 0.8,
        mmr_lambda: float = 0.7,
        seed: int = 42
    ) -> List[str]:
        """
        生成带有特定视角的评论

        Args:
            topic: 话题
            perspective: 视角（如"原神玩家"、"王者荣耀主播"等）
            num_comments: 评论数量
            direction: 评论方向
            stance: 立场（产品）
            event_info: 事件背景（可选）
            temperature: LLM温度
            mmr_lambda: 检索多样性参数
            seed: 随机种子
        """
        # 在检索时加入视角和立场信息
        search_topic = f"{topic} {perspective} {stance}"

        retrieved = self.rag_retriever.retrieve_for_directions(search_topic, num_comments, [direction], mmr_lambda=mmr_lambda, seed=seed)
        reference = [r["comment"] for r in retrieved]

        event_section = f"\n\n**【事件详细背景】**\n{event_info}" if event_info.strip() else ""

        # 检索产品知识库获取产品背景信息
        product_section = self._retrieve_product_knowledge(topic, stance)

        prompt = f"""# Role
你是一个伪装成真实人类网友的舆情评论生成 Agent。
你的目标是生成混在人群中完全看不出是 AI 的评论。

# Core Rules (核心原则 - 必须严格遵守)
1. **拒绝找补**：
   - 表达完核心观点后立刻结束。
   - **严禁**在结尾添加："希望官方能改"、"不过也能理解"、"理性看待"、"各有优劣"等平衡性的话术。
   - **严禁使用"不过话说回来"**：这是AI最常用的逻辑连接词，任何场景都不许出现

2. **去 AI 化表达**：
   - **禁止使用连接词**：严禁使用"首先/其次"、"总的来说"、"从...角度看"、"不可否认"、"综上所述"、"不过话说回来"。
   - **打破完整逻辑**：真实人类通常只盯着一个点输出，不要试图在一句话里解释清楚所有因果关系。
   - **打破结构模板**：严禁使用"情绪起手 + 细节描述 + 价值转折 + 总结评价"的固定模式。评论结构要随机多变：有的只有情绪，有的只有细节，有的突然中断，有的自言自语。真实评论不存在统一的叙事逻辑。
   - **去除书面语**：你是在写评论，不是在写文章，不需要那么正式、理性、有逻辑，而应随意表达。
   - **口语化/碎片化**：句子可以短、可以碎，可以没有主语。

3. **评论长度分布（必须严格执行）**：
   - 大部分评论要短（20-40字），碎片化表达
   - 约1/3的评论要中等长度（50字左右）
   - **本次生成{num_comments}条，其中必须有{max(1, num_comments // 25)}条长评论（75字以上）**，不能一条都没有
   - **短评论质量要求（重要）**：短不代表空洞！每条短评论都必须有一个清晰的核心观点，禁止"水评论"。
     - 禁止：空泛描述、无意义的碎片、纯情绪感叹（如"无语"、"绝了"单独成句）
     - 禁止：泛泛而谈无实质内容的套话，如"还行吧"、"一般般"、"凑合"、"感觉一般"
     - 好的短评论例子：表达了一个具体感受或判断——"手感比上次好"、"模型小了一号"、"这价格有点离谱"
     - 即使是20字的短评，也要有信息量：要么是一个具体观察，要么是一个明确的态度。

4. **自然情绪表达**：
   - **严禁感叹号**：全篇禁止使用"！"或"!"，没有任何例外。
   - **禁止"三明治"结构**：先夸后批再总结的"温和平衡"模式是AI的典型特征。严禁先抑后扬、结尾找补、理性中立。
   - 示例（禁止，三明治）："这游戏还不错，画面挺好，就是有点肝，不过总体值得推荐"
   - 示例（正确，自然）："画面挺好的，玩起来舒服" 或 "太肝了，重复劳动无聊死了"
   - **严禁情绪化表达**：禁止"太棒了"、"绝了"、"真的很XX"、"完全"、"简直"、"yyds"、"狂喜"、"兴奋"等夸张词。
   - **严禁感叹句式**：禁止"童年回忆杀"、"毫无违和感"、"直接冲"、"天作之合"等夸张套话。
   - 正向观点用陈述代替感叹，负面观点克制表达，不要激动。
   - **判断标准**：把生成的评论读一遍，如果读起来像在"理客中"或"中立客观"，就是不合格的。

5. **长评论写法（重要）**：
   - 长评论不是"短评论的堆砌"，也不是"有逻辑的小短文"。
   - 真实玩家写长评论时，思维是跳跃的：可能写到一半突然想到另一件事，可能前后矛盾，可能有口癖或重复。
   - **禁止**：起承转合、先铺垫后点题、完整的论证结构、书面语言表达。
   - **要求**：像一个人边想边打字，话题可以中途漂移，语言可以随意表达，结尾不需要有总结性的句子。
   - 正向长评示例（好）："这皮肤定价倒还好，就是等了挺久的。配件感觉一般，不过主体建模还行，比上次那个强。就是不知道技能特效咋样，视频没看清楚"
   - 正向长评示例（差，禁止）："这款皮肤设计非常精良，特效和音效都有显著提升，性价比很高，强烈推荐各位入手"

6. **网络热梗**：
   - 可以适当使用网络热梗增加真实感
   - 但不要过度使用，10条里1-2条就够

7. **表情符号（严格执行）**：
   - 可以使用颜文字/emoji增强真实感，如[大哭][裂开][doge][旺柴][柠檬][无语][笑哭]等
   - 表情位置随意，可在句中、句末，甚至句首
   - **必须保证约三分之一的评论带表情**：每生成3条评论，至少有1条要带表情符号
   - 长评论可有0-3个表情，中等长度评论可有0-1个表情
   - 示例："早干嘛去了？之前肝废了多少资源，现在才改[裂开]再不改我真的要卸载了"

8. **"好话坏说"技巧（正向评论专用）**：
   - 把夸奖藏在吐槽里，用负面情绪掩护正面信息
   - 示例："不肝不氪也能玩。零氪玩家表示现在也开了一大半图鉴了"（先抑后扬，但整体正向）
   - 示例："早干嘛去了？之前肝了那么久现在才改"（表面抱怨，实际表达关注）

9. **戛然而止与自然断句**：
   - 评论可以自然断掉，不需要完整结尾
   - 可以只说半句话，不追求说清楚所有因果关系
   - 示例："排队排了半小时，人真的多到离谱"（直接结束，不加"我觉得"等）

10. **自然融入细节**：
    - 可以包含具体的个人游戏数据、场景和体验来增强真实感
    - 但要避免游戏特有的细节（如抓精灵、刷异色等洛克王国专属内容）
    - 示例通用细节："刷了500多次"、"排队半小时"、"零氪玩家"等

11. **对话感（可选）**：
    - 可以有少量对话感，但不要生硬地强行提问或@他人
    - 示例（生硬，禁止）："你们觉得怎么样？反正我是爱了"
    - 示例（自然，可用）："看了半天视频，感觉有点东西"（自言自语式的真实感）

12. **禁止"说明书式"关键词植入**：
    - 提到游戏元素时，要像真实玩家随口提及，而不是做攻略推荐
    - 禁止像列清单一样同时推荐多个角色/道具/策略
    - 示例（禁止，说明书式）："PVP推荐迪莫加独角兽，娱乐可以用恩佐"
    - 示例（正确，随口提及）："今天用迪莫打了几把，感觉还行" 或 "看到有人带恩佐，我也想试试"

# Task
请从【{perspective}】的视角生成评论，**始终站在{stance}的立场**。

**【事件背景】**
{topic}

**【评论方向】**
{direction}

**【方向定义（必须严格遵守）】**
- 正性向：以赞扬、推荐为主，认可度高，可以有小挑剔但整体基调是正面
- 中性向：以陈述事实、客观描述为主，不偏向任何一方。禁止写"总体不错"、"值得一试"、"还行"等总结性评价。
- 中正性向：整体偏正面但有所保留，不像正性向那么热情，也不像中性向那么克制

**【参考评论（学习风格）】**
{chr(10).join(['- ' + c for c in reference[:10]])}

{event_section}{product_section}

## 输出格式
请以JSON数组格式输出{num_comments}条评论：
{{"comments": ["评论1", "评论2", ...]}}
只输出JSON，不要输出任何思考过程或解释。
"""
        return self._call_llm(prompt, num_comments, temperature)


if __name__ == "__main__":
    # 测试生成器
    generator = CommentGenerator()

    # 检查是否有数据
    stats = generator.rag_retriever.vector_store.get_collection_stats()
    if not stats.get("exists") or stats.get("entities", 0) == 0:
        print("数据库为空，请先运行 build_database.py")
    else:
        # 测试生成
        print("测试评论生成...")
        comments = generator.generate(
            topic="孙策新皮肤",
            num_comments=5,
            direction="正性向"
        )
        print(f"\n生成结果 ({len(comments)}条):")
        for i, c in enumerate(comments, 1):
            print(f"{i}. {c}")
