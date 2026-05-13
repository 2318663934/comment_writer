"""
LLM评论生成模块 - 基于RAG检索结果生成评论
v2.0: 简化prompt、两阶段生成(角度→评论)、玩家行为常识
"""
import os
import json
import re
import time
import random
from typing import List, Dict, Any, Optional
from openai import OpenAI
from rag_retriever import RAGRetriever
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


# ============================================================
# Prompt 模板 — v2.1 精简版
# ============================================================

SYSTEM_PROMPT = """你是一个真实的游戏玩家，正在网上随手发表评论。

## 说话方式
- 想到什么说什么，一次只盯一个点，句子可短可碎可无主语
- 禁止感叹号"！"、夸张表达（太棒了/绝了/yyds/强烈推荐）
- 禁止书面连接词（首先其次/总的来说/不过话说回来）
- 禁止三明治结构（先夸后批再总结），表达完立刻结束
- 禁止AI式结尾："希望官方能改""不过也能理解""你们觉得怎么样"

## 评论内容规律
真实玩家评论不会只围绕一个主题转。同一个人可能：
- 说着活动突然吐槽起匹配机制
- 讨论皮肤时想起自己当年的某次抽卡
- 聊着新版本怀念起旧版本的好
- 从KFC联动想到上次联动的社死经历
部分评论应该和主题只有松散关联，甚至完全跑题聊自己的事。
不要所有评论都只讨论事件背景提到的点。

## 评论结构多样性
每条评论从以下风格随机选取，同批不能重复：
纯感受 / 细节碎片 / 自言自语 / 对比吐槽 / 话说一半 / 带个人数据

## 玩家常识
- 关注焦点：限时活动、版本更新、平衡调整、肝度氪度
- 常驻活动（每日签到等）不是讨论焦点，不编造数据，不做攻略推荐
- 约1/3评论带表情([doge][笑哭][裂开][大哭][旺柴])，位置随意

## 长度要求
- 短评(15-40字)：必须观点明确，禁止"还行/凑合/一般般"
- 中评(40-70字)、长评(75字+)：思维可跳跃、不需完整结尾
"""

# 方向定义（去重）
DIRECTION_DEFINITIONS = """**【方向定义】**
- 正性向：赞扬、认可为主，可有小挑剔但整体正面
- 中性向：客观陈述为主，禁止总结性评价
- 中正性向：偏正面但克制，不如正性向热情
"""

# JSON 输出指令（去重）
JSON_OUTPUT_INSTRUCTION = "以JSON数组输出，只输出JSON，不要任何解释。"

# 观众视角指令（去重）
VIEWER_MODE_PROMPT = """**【重要：你的身份是刚看完这个内容的观众】**
你刚看了上面的内容，像一个在评论区留言的人发表看法：
- 可以针对具体画面、瞬间、细节来写
- 提到内容要像"刚看过"一样自然（"看到XX的时候""那个镜头"），禁止用"视频中显示""根据画面"等书面语
- 抓住值得讨论的细节来写，不要泛泛而谈
"""

# 头脑风暴 prompt（在角度生成和评论生成之间，让 LLM 基于事件+产品自由发散）
BRAINSTORM_PROMPT = """你是一个资深游戏玩家和舆情观察者。请基于以下信息做一次头脑风暴。

【产品名称】{stance}
【事件背景】{event_info}
{product_section}

请从以下每个维度各想1-2个切入点，总共{num_angles}个：

1. **玩家情绪/心理**：事件会触动玩家的什么感受？期待、失望、好奇、怀旧、焦虑、兴奋？
2. **产品/玩法层面**：对游戏本身有什么影响？平衡性、肝度、社交体验、新手友好度？
3. **商业/运营视角**：官方为什么这么做？定价策略、营销手法、时机选择？
4. **社区/舆论角度**：玩家圈子里会怎么讨论？不同群体的看法会有什么分歧？
5. **对比/联想**：和过去类似的事件、其他游戏的类似操作有什么可比之处？
6. **发散/跑题**：玩家讨论这件事时，可能会联想到哪些不直接相关的话题？
   例如：从联动想起以前的联动、从某个道具想起自己的游戏经历、从活动规则想起日常吐槽点

要求：每个切入点必须具体、有讨论价值。允许合理推测，但要标注"可能""或许""会不会"。
严禁编造不存在的活动、数据、角色。

请以JSON格式输出：
{{"brainstorm": ["切入点1", "切入点2", ...]}}
{json_out}"""

# 角度生成 prompt（两阶段生成的第一阶段）
ANGLE_GENERATION_PROMPT = """针对以下话题，列出{num_angles}个不同的评论切入点（角度）。
每个切入点必须具体独特，不能是泛泛的类别。
好的例子："排队时间影响体验"、"新皮建模比原画好"、"零氪玩家资源跟不上"
{used_angles_section}
**【话题】**{topic}
**【事件背景】**{event_info}
**【立场】**{stance}
**【评论方向】**{directions}

{json_out}"""


# 产品立场到知识库collection的映射
STANCE_TO_PRODUCT_COLLECTION = {
    "王者荣耀": "honor_of_kings",
    "王者荣耀世界": "honor_of_kings_world",
    "洛克王国世界": "lok_world",
    "DNF端游": "dnf",
    "金铲铲之战": "jcjz",
    "无畏契约手游": "wxqy",
    "原神": None,
    "三角洲行动": None,
}


class CommentGenerator:
    """评论生成器 v2.0"""

    def __init__(
        self,
        rag_retriever: RAGRetriever = None,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.rag_retriever = rag_retriever or RAGRetriever()

        self.api_key = api_key or LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or LLM_BASE_URL
        self.model = model or LLM_MODEL

        if not self.api_key:
            print("警告: 未设置OPENAI_API_KEY，将无法生成评论")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 会话级角度追踪：避免重复生成相同角度的评论
        self._session_angles: Dict[str, List[tuple]] = {}  # {topic: [(angle, timestamp), ...]}
        self._angle_max_size = 100
        self._angle_ttl = 1800  # 30分钟后角度过期

    # ============================================================
    # 产品知识库检索
    # ============================================================

    def _retrieve_product_knowledge(self, topic: str, stance: str, top_k: int = 10) -> str:
        """检索产品知识库并返回格式化背景信息"""
        collection = STANCE_TO_PRODUCT_COLLECTION.get(stance)
        if collection is None:
            return ""

        try:
            import sys
            from pathlib import Path
            kb_path = Path("E:/产品信息知识库")
            if str(kb_path) not in sys.path:
                sys.path.insert(0, str(kb_path))

            from crawler.product_retriever import search as kb_search
            results = kb_search(topic, product=collection, top_k=top_k)
            if not results:
                results = kb_search(stance, product=None, top_k=5)

            if not results:
                return ""

            # 按话题热度排序：限时/新上内容优先
            hot_keywords = ["限时", "新上", "首周", "限定", "新版本", "登场", "爆料", "上线"]
            def hotness(r):
                score = 0
                text = r.get("title", "") + r.get("content_text", "")
                for kw in hot_keywords:
                    if kw in text:
                        score += 1
                return -score  # 负分让热度高的排在前面

            results = sorted(results, key=hotness)

            lines = [f"\n\n**【产品相关背景】**\n以下是关于{stance}的近期相关报道："]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']} - 来源: {r['source']}")
                lines.append(f"   {r['content_text'][:300]}...")
            return "\n".join(lines)
        except Exception as e:
            print(f"产品知识库检索失败: {e}")
            return ""

    # ============================================================
    # 两阶段生成：Stage 1 — 生成评论角度
    # ============================================================

    def _get_used_angles(self, topic: str) -> set:
        """获取当前会话中已使用过的角度（含过期清理）"""
        now = time.time()
        if topic in self._session_angles:
            self._session_angles[topic] = [
                (a, ts) for a, ts in self._session_angles[topic]
                if now - ts < self._angle_ttl
            ]
            if not self._session_angles[topic]:
                del self._session_angles[topic]
                return set()
            return {a for a, _ in self._session_angles[topic]}
        return set()

    def _add_used_angles(self, topic: str, angles: List[str]):
        """记录已使用的角度"""
        now = time.time()
        if topic not in self._session_angles:
            self._session_angles[topic] = []
        for a in angles:
            self._session_angles[topic].insert(0, (a, now))
        if len(self._session_angles[topic]) > self._angle_max_size:
            self._session_angles[topic] = self._session_angles[topic][:self._angle_max_size]

    def _generate_comment_angles(
        self,
        topic: str,
        num_angles: int,
        directions: List[str],
        stance: str,
        event_info: str = ""
    ) -> List[str]:
        """Stage 1: 生成多样化的评论切入角度"""
        context = topic if topic.strip() else (event_info[:200] if event_info else stance)
        cache_key = topic or f"_{hash(context) & 0x7FFFFFFF:x}_"

        used_angles = self._get_used_angles(cache_key)
        used_section = ""
        if used_angles:
            used_list = "\n".join([f"- {a}" for a in list(used_angles)[:15]])
            used_section = f"\n**【已使用过的角度，请避开这些内容】**:\n{used_list}"

        directions_str = "、".join(directions)
        event_text = event_info if event_info.strip() else context

        prompt = ANGLE_GENERATION_PROMPT.format(
            num_angles=num_angles,
            topic=topic,
            event_info=event_text[:1500],
            stance=stance,
            directions=directions_str,
            used_angles_section=used_section,
            json_out=JSON_OUTPUT_INSTRUCTION
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏舆情分析师。请以JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,  # 较低温度保证角度质量
                max_tokens=1000
            )

            content = response.choices[0].message.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 解析JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
                angles = data.get("angles", [])
                if angles:
                    print(f"角度生成成功: {len(angles)}个切入点")
                    return angles

            # 回退：按行解析
            lines = [l.strip().lstrip("0123456789.、- ") for l in content.split("\n") if l.strip()]
            angles = [l for l in lines if len(l) >= 4 and len(l) <= 50]
            if angles:
                print(f"角度生成（按行回退）: {len(angles)}个切入点")
                return angles[:num_angles]

            print("角度生成失败，将使用默认模式")
            return []
        except Exception as e:
            print(f"角度生成出错: {e}")
            return []

    def _brainstorm(
        self,
        stance: str,
        event_info: str,
        product_section: str = "",
        num_points: int = 10
    ) -> List[str]:
        """
        Stage 1.5: 基于事件信息 + 产品知识，LLM 头脑风暴讨论切入点

        与角度生成的区别：
        - 角度生成是轻量的"标签"（如"排队时间影响体验"）
        - 头脑风暴是深度的"讨论方向"，结合了事件细节和产品特性
        """
        if not event_info.strip():
            return []

        prompt = BRAINSTORM_PROMPT.format(
            stance=stance,
            event_info=event_info[:2000],
            product_section=product_section[:1500] if product_section else "（无产品背景信息）",
            num_angles=num_points,
            json_out=JSON_OUTPUT_INSTRUCTION
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏舆情分析师。请以JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
                points = data.get("brainstorm", [])
                if points:
                    print(f"头脑风暴完成: {len(points)}个切入点")
                    return points

            print("头脑风暴 JSON 解析失败，回退为空")
            return []
        except Exception as e:
            print(f"头脑风暴失败: {e}")
            return []

    # ============================================================
    # 主生成方法
    # ============================================================

    def generate(
        self,
        topic: str,
        num_comments: int = 10,
        direction: str = "中性向",
        stance: str = "王者荣耀",
        event_info: str = ""
    ) -> List[str]:
        """生成评论（单方向便捷方法）"""
        return self.generate_for_directions(topic, num_comments, [direction], stance, event_info)

    def generate_for_directions(
        self,
        topic: str = "",
        num_comments: int = 10,
        directions: List[str] = None,
        stance: str = "王者荣耀",
        event_info: str = "",
        temperature: float = 0.8,
        use_rag: bool = True,
        use_kb: bool = True
    ) -> List[str]:
        """两阶段生成：先定角度，再写评论。topic 为空时从 event_info 推导。"""
        if directions is None:
            directions = ["中性向"]
        num_comments = max(1, min(100, num_comments))

        # topic 为空时从 event_info 推导
        if not topic or not topic.strip():
            topic = event_info[:80] if event_info else ""
        context = topic or stance

        # Stage 0: RAG 检索（可选关闭以对比无数据库效果）
        if use_rag and self.rag_retriever:
            retrieved = self.rag_retriever.retrieve_for_directions(
                context, num_comments, directions, mmr_lambda=0.3,
                event_info=event_info, seed=0
            )
            reference = [r["comment"] for r in retrieved]
        else:
            reference = []

        # Stage 1: 生成角度
        num_angles = min(num_comments + 5, num_comments * 2, 30)
        angles = self._generate_comment_angles(topic, num_angles, directions, stance, event_info)
        if not angles:
            angles = []

        # Stage 1.5: 头脑风暴（基于事件+产品知识自由发散讨论点）
        product_section = ""
        if use_kb:
            product_section = self._retrieve_product_knowledge(topic or stance, stance, top_k=10)
        brainstorm = self._brainstorm(
            stance=stance,
            event_info=event_info,
            product_section=product_section,
            num_points=max(num_comments // 2, 5)
        )

        # Stage 2: 生成评论
        prompt = self._build_comment_prompt(
            topic=topic,
            num_comments=num_comments,
            directions=directions,
            reference=reference,
            stance=stance,
            event_info=event_info,
            product_section=product_section,
            angles=angles,
            brainstorm=brainstorm
        )

        comments = self._call_llm(prompt, num_comments, temperature)
        if angles and comments:
            cache_key = topic or f"_{hash(event_info) & 0x7FFFFFFF:x}_"
            self._add_used_angles(cache_key, angles[:len(comments)])

        # 简单去重：过滤掉内容高度相似的评论
        comments = self._deduplicate_comments(comments)

        return comments

    def _deduplicate_comments(self, comments: List[str]) -> List[str]:
        """简单的内容去重：移除完全相同或高度相似的评论"""
        if len(comments) <= 1:
            return comments
        seen = set()
        result = []
        for c in comments:
            # 归一化：取前30字作为快速指纹
            normalized = c[:30].strip()
            if normalized not in seen:
                seen.add(normalized)
                result.append(c)
        return result

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_comment_prompt(
        self,
        topic: str,
        num_comments: int,
        directions: List[str],
        reference: List[str],
        stance: str = "王者荣耀",
        event_info: str = "",
        product_section: str = "",
        angles: List[str] = None,
        brainstorm: List[str] = None,
        extra_header: str = ""
    ) -> str:
        """构建评论生成的 user prompt"""

        # 方向规格
        direction_specs = []
        num_directions = len(directions)
        base_count = num_comments // num_directions
        remainder = num_comments % num_directions
        for i, d in enumerate(directions):
            count = base_count + (1 if i < remainder else 0)
            direction_specs.append(f"- {d}：生成{count}条")

        # 媒体内容 → 观众视角
        is_media = event_info and "[从媒体提取的信息]" in event_info
        if is_media:
            event_section = f"\n**【你刚看了一个视频/图片，以下是内容详情】**\n{event_info}\n{VIEWER_MODE_PROMPT}"
        elif event_info.strip():
            event_section = f"\n**【事件详细背景】**\n{event_info}"
        else:
            event_section = ""

        # 角度（打乱顺序 + 不强制一一对应，避免 LLM 按序映射导致重复）
        if angles:
            import random as _random
            shuffled = list(angles)
            _random.shuffle(shuffled)
            angle_list = "\n".join([f"- {a}" for a in shuffled])
            angle_section = f"\n**【可参考的评论切入点】**\n{angle_list}\n\n以上是灵感来源。可以混合使用、自由跳转，不要去按顺序逐个覆盖。不同评论之间可以有交集但角度要错开。"
        else:
            angle_section = "\n**【内容多样性要求】**\n每条评论必须有独特的信息点和切入角度，严禁内容重复。"

        # 头脑风暴结果（打乱顺序，避免 LLM 按序对应）
        brainstorm_section = ""
        if brainstorm:
            import random as _random
            shuffled = list(brainstorm)
            _random.shuffle(shuffled)
            b_list = "\n".join([f"- {b}" for b in shuffled])
            brainstorm_section = f"""
**【头脑风暴——以下是从事件中衍生出的值得讨论的点】**
（这些是灵感来源，自由跳转使用。注意：带"可能""或许""会不会"的是推测，写评论时要保持不确定性）
{b_list}
"""

        # 参考评论（提供 2x 生成量，让 LLM 自由选择想参考的）
        ref_samples = reference[:num_comments * 2] if reference else []
        ref_text = "\n".join([f"- {c}" for c in ref_samples]) if ref_samples else "（无参考评论，请自由发挥）"
        ref_hint = "\n（以上是真实玩家评论的参考样本，只需学习其风格和用语习惯，不要抄袭内容）" if ref_samples else ""

        # 长度分布
        num_long = max(1, num_comments // 20)
        num_mid = num_comments // 3
        num_short = num_comments - num_long - num_mid

        topic_section = f"**【话题】**\n{topic}\n" if topic and topic.strip() else ""
        return f"""{topic_section}{extra_header}
**【立场】**
站在{stance}玩家的角度

**【需要生成的评论】**
{"\n".join(direction_specs)}
共{num_comments}条

{DIRECTION_DEFINITIONS}

**【长度分布】**
{num_short}条短评(15-40字)、{num_mid}条中评(40-70字)、{num_long}条长评(75字+)
{event_section}
{product_section if product_section else ""}
{angle_section}{brainstorm_section}
**【参考评论（仅学习风格）】**
{ref_text}{ref_hint}

**【输出格式】**
{{"comments": [{{"content": "...", "direction": "正性向"}}, ...]}}
{JSON_OUTPUT_INSTRUCTION}"""

    def _build_v2_prompt(self, **kwargs):
        """兼容旧接口"""
        return self._build_comment_prompt(**kwargs)

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
        """构建单方向 prompt（保留兼容旧接口）"""
        return self._build_v2_prompt(
            topic=topic,
            num_comments=num_comments,
            directions=[direction],
            reference=reference,
            stance=stance,
            event_info=event_info,
            product_section=product_section
        )

    # ============================================================
    # LLM 调用
    # ============================================================

    def _call_llm(self, prompt: str, num_comments: int, temperature: float = 0.85) -> List[str]:
        """调用LLM生成评论"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 方法1：标准JSON解析
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    comments = data.get("comments", [])

                    if comments and isinstance(comments[0], dict) and "content" in comments[0]:
                        extracted = [c["content"] for c in comments if "content" in c]
                        cleaned = [c.replace("！", "").replace("!", "") for c in extracted]
                        print(f"JSON解析成功: {len(cleaned)}条")
                        return cleaned[:num_comments]
                    elif comments and isinstance(comments[0], str):
                        cleaned = [c.replace("！", "").replace("!", "") for c in comments]
                        print(f"JSON解析成功(旧格式): {len(cleaned)}条")
                        return cleaned[:num_comments]
            except json.JSONDecodeError:
                pass

            # 方法2：正则提取
            print(f"标准JSON解析失败，尝试元素级提取，内容长度: {len(content)}")
            element_pattern = re.compile(r'"([^"]{10,500})"\s*[,}\]]')
            extracted = element_pattern.findall(content)
            if extracted:
                cleaned = [c.replace("！", "").replace("!", "") for c in extracted]
                print(f"元素级提取: {len(cleaned)}条")
                return cleaned[:num_comments]

            # 方法3：按行回退
            print("元素级提取失败，按行回退")
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

    # ============================================================
    # 多视角生成
    # ============================================================

    def generate_with立场(
        self,
        topic: str = "",
        perspective: str = "",
        num_comments: int = 5,
        direction: str = "中性向",
        stance: str = "王者荣耀",
        event_info: str = "",
        temperature: float = 0.8,
        seed: int = 42
    ) -> List[str]:
        """
        生成带有特定视角的评论 — 复用 _build_comment_prompt
        """
        context = topic if topic and topic.strip() else (event_info[:80] if event_info else stance)
        search_topic = f"{context} {perspective} {stance}"

        # Stage 0: RAG 检索
        retrieved = self.rag_retriever.retrieve_for_directions(
            search_topic, num_comments, [direction],
            mmr_lambda=0.3, seed=0
        )
        reference = [r["comment"] for r in retrieved]

        # Stage 1: 生成角度
        angles = self._generate_comment_angles(
            topic=f"{context}（从{perspective}视角）",
            num_angles=min(num_comments + 3, 15),
            directions=[direction],
            stance=stance,
            event_info=event_info
        )

        # Stage 1.5: 头脑风暴
        product_section = self._retrieve_product_knowledge(topic, stance)
        brainstorm = self._brainstorm(
            stance=stance, event_info=event_info,
            product_section=product_section,
            num_points=max(num_comments // 2, 5)
        )

        # Stage 2: 复用公共 prompt 构建
        prompt = self._build_comment_prompt(
            topic=topic,
            num_comments=num_comments,
            directions=[direction],
            reference=reference,
            stance=stance,
            event_info=event_info,
            product_section=product_section,
            angles=angles,
            brainstorm=brainstorm,
            extra_header=f'\n**【模拟视角】**\n从"{perspective}"的视角发表评论\n'
        )

        return self._call_llm(prompt, num_comments, temperature)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    generator = CommentGenerator()

    stats = generator.rag_retriever.vector_store.get_collection_stats()
    if not stats.get("exists") or stats.get("entities", 0) == 0:
        print("数据库为空，请先运行 build_database.py")
    else:
        print("测试评论生成...")
        comments = generator.generate(
            topic="孙策新皮肤",
            num_comments=5,
            direction="正性向"
        )
        print(f"\n生成结果 ({len(comments)}条):")
        for i, c in enumerate(comments, 1):
            print(f"{i}. {c}")
