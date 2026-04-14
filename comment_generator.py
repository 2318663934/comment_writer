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

2. **去 AI 化表达**：
   - **禁止使用连接词**：严禁使用"首先/其次"、"总的来说"、"从...角度看"、"不可否认"、"综上所述"。
   - **打破完整逻辑**：真实人类通常只盯着一个点输出，不要试图在一句话里解释清楚所有因果关系。
   - **去除书面语**：你是在写评论，不是在写文章，不需要那么正式、理性、有逻辑，而应随意表达。
   - **口语化/碎片化**：句子可以短、可以碎，可以没有主语。

3. **评论长度分布（必须严格执行）**：
   - 大部分评论要短（20-40字），碎片化表达
   - 约1/3的评论要中等长度（50字左右）
   - **本次生成{num_comments}条，其中必须有{num_long}条长评论（75字以上）**，不能一条都没有
   - **短评论质量要求（重要）**：短不代表空洞！每条短评论都必须有一个清晰的核心观点，禁止"水评论"。
     - 禁止：空泛描述、无意义的碎片、纯情绪感叹（如"无语"、"绝了"单独成句）
     - 禁止：泛泛而谈无实质内容的套话，如"还行吧"、"一般般"、"凑合"、"感觉一般"
     - 好的短评论例子：表达了一个具体感受或判断——"手感比上次好"、"模型小了一号"、"这价格有点离谱"
     - 即使是20字的短评，也要有信息量：要么是一个具体观察，要么是一个明确的态度。

4. **克制表达（重中之重）**：
   - **严禁感叹号**：全篇禁止使用"！"或"!"，没有任何例外。真实随意的评论极少用感叹号。
   - **严禁情绪化表达**：禁止使用"太棒了"、"绝了"、"真的很XX"、"完全"、"简直"、"太对了"、"yyds"、"狂喜"、"兴奋"、"整破防了"等夸张或激动的词。
   - **严禁感叹句式**：禁止"童年回忆杀"、"毫无违和感"、"直接冲"、"天作之合"等夸张套话。
   - 正向观点要隐晦，用陈述代替感叹。例如不要写"特效绝了"，而是写"特效这次换了个思路，感觉不一样"。
   - 负面观点同理，克制、平淡，不要激动。
   - 你在描述一个事实或感受，不是在写种草文案，也不是在写投诉信。
   - **判断标准**：把生成的评论读一遍，如果读起来像在"安利"或"宣泄情绪"，就是不合格的。

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

## 输出格式
请以JSON数组格式输出{num_comments}条评论，格式如下：
{{"comments": ["评论1", "评论2", ...]}}
只输出JSON，不要输出任何思考过程或解释。
"""


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
        # 限制数量范围
        num_comments = max(1, min(100, num_comments))

        # 检索相关评论作为参考
        retrieved = self.rag_retriever.retrieve(topic, num_comments, direction)
        reference = [r["comment"] for r in retrieved]

        if not reference:
            print("警告: 未检索到相关评论，使用默认提示")

        # 构建prompt
        prompt = self._build_prompt(topic, num_comments, direction, reference, stance, event_info)

        # 调用LLM生成
        comments = self._call_llm(prompt, num_comments)

        return comments

    def _build_prompt(
        self,
        topic: str,
        num_comments: int,
        direction: str,
        reference: List[str],
        stance: str = "王者荣耀",
        event_info: str = ""
    ) -> str:
        """构建生成prompt"""
        reference_text = "\n\n".join([
            f"- {c}" for c in reference[:10]  # 限制参考数量
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
            event_section=event_section
        )

        return prompt

    def _call_llm(self, prompt: str, num_comments: int) -> List[str]:
        """调用LLM生成评论"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个帮助生成游戏评论的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # 较高的temperature增加多样性
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
                    if comments:
                        comments = [c.replace("！", "").replace("!", "") for c in comments]
                        print(f"标准JSON解析成功: {len(comments)}条")
                        return comments[:num_comments]
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
        event_info: str = ""
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
        """
        # 在检索时加入视角和立场信息
        search_topic = f"{topic} {perspective} {stance}"

        retrieved = self.rag_retriever.retrieve(search_topic, num_comments, direction)
        reference = [r["comment"] for r in retrieved]

        event_section = f"\n\n**【事件详细背景】**\n{event_info}" if event_info.strip() else ""

        prompt = f"""# Role
你是一个伪装成真实人类网友的舆情评论生成 Agent。
你的目标是生成混在人群中完全看不出是 AI 的评论。

# Core Rules (核心原则 - 必须严格遵守)
1. **拒绝找补**：
   - 表达完核心观点后立刻结束。
   - **严禁**在结尾添加："希望官方能改"、"不过也能理解"、"理性看待"、"各有优劣"等平衡性的话术。

2. **去 AI 化表达**：
   - **禁止使用连接词**：严禁使用"首先/其次"、"总的来说"、"从...角度看"、"不可否认"、"综上所述"。
   - **打破完整逻辑**：真实人类通常只盯着一个点输出，不要试图在一句话里解释清楚所有因果关系。
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

4. **克制表达（重中之重）**：
   - **严禁感叹号**：全篇禁止使用"！"或"!"，没有任何例外。真实随意的评论极少用感叹号。
   - **严禁情绪化表达**：禁止使用"太棒了"、"绝了"、"真的很XX"、"完全"、"简直"、"太对了"、"yyds"、"狂喜"、"兴奋"、"整破防了"等夸张或激动的词。
   - **严禁感叹句式**：禁止"童年回忆杀"、"毫无违和感"、"直接冲"、"天作之合"等夸张套话。
   - 正向观点要隐晦，用陈述代替感叹。例如不要写"特效绝了"，而是写"特效这次换了个思路，感觉不一样"。
   - 负面观点同理，克制、平淡，不要激动。
   - 你在描述一个事实或感受，不是在写种草文案，也不是在写投诉信。
   - **判断标准**：把生成的评论读一遍，如果读起来像在"安利"或"宣泄情绪"，就是不合格的。

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

{event_section}

## 输出格式
请以JSON数组格式输出{num_comments}条评论：
{{"comments": ["评论1", "评论2", ...]}}
只输出JSON，不要输出任何思考过程或解释。
"""
        return self._call_llm(prompt, num_comments)


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
