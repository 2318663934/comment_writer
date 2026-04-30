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
# Prompt 模板 — v2.0 精简版：正向引导 + 示例驱动
# ============================================================

SYSTEM_PROMPT = """你是一个真实的游戏玩家，正在网上随手发表评论。你的评论读起来完全不像AI写的。

## 表达习惯
- 想到什么说什么，一次只盯一个点，不追求面面俱到
- 句子可短可碎可无主语，用口语而不是书面语
- 用陈述句表达感受，禁止使用感叹号"！"
- 禁止"太棒了/绝了/yyds/强烈推荐/不容错过/天作之合"等夸张表达
- 禁止"首先其次/总的来说/不过话说回来/综上所述"等书面连接词
- 禁止"三明治"结构（先夸后批再总结）
- 表达完核心观点立刻结束，拒绝平衡性找补话术

## 评论结构多样性（极其重要）
每条评论从以下风格中随机选取，同批评论风格不能重复：
- 纯感受："玩了几把，手感确实比上赛季好"
- 细节碎片："模型好像小了一圈，不知道是不是错觉"
- 自言自语："什么时候修那个bug，都好几个版本了"
- 对比吐槽："比上次的皮肤强，上次那个真的没法看"
- 话说一半："排队半小时进去秒退，这谁顶得住"
- 带个人数据："刷了两三百次了，毛都没见到"

## 玩家常识
- 焦点：限时活动、版本更新内容、平衡性调整、肝度氪度、社交体验
- 常驻活动（每日签到、常驻商店）不是讨论焦点，不要围绕它们写
- 不编造具体数值，不围绕冷门玩法展开讨论
- 提到游戏内容要像随口提及，不做攻略式推荐

## 长度要求
- 短评(15-40字)：必须有清晰观点，禁止"还行/凑合/一般般/感觉一般/就那样"
- 中评(40-70字)：可略展开但不要写成小作文
- 长评(75字+)：思维跳跃、话题可漂移、不需要完整结尾

## 表情符号
约1/3评论带表情：[doge][笑哭][裂开][大哭][无语][柠檬][旺柴]
位置随意，句中句末句首都可以。

## 禁止的AI特征
严禁出现以下词汇或句式：
"希望官方能改" "不过也能理解" "理性看待" "各有优劣"
"你们觉得怎么样" "有没有一样的" "反正我是爱了"
"""

# 角度生成 prompt（两阶段生成的第一阶段）
ANGLE_GENERATION_PROMPT = """针对以下话题，列出{num_angles}个不同的评论切入点（角度）。

要求：
- 每个切入点必须具体、独特，是一个明确的观察或感受
- 不能是泛泛的分类标签
- 好的例子："排队时间影响体验"、"新皮建模比原画好"、"零氪玩家资源跟不上"
- 坏的例子（太泛，不可用）："游戏体验"、"皮肤评价"、"玩家感受"、"更新内容"
{used_angles_section}
**【话题】**
{topic}

**【事件背景】**
{event_info}

**【立场】**
{stance}

**【评论方向】**
{directions}

请以JSON格式输出：
{{"angles": ["切入点1", "切入点2", ...]}}
只输出JSON，不要任何解释。"""


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
        """
        Stage 1: 生成多样化的评论切入角度

        Args:
            topic: 话题
            num_angles: 需要生成的角度数量（略多于评论数以确保多样性）
            directions: 评论方向列表
            stance: 产品立场
            event_info: 事件背景

        Returns:
            角度列表
        """
        # 获取已使用过的角度
        used_angles = self._get_used_angles(topic)
        used_section = ""
        if used_angles:
            used_list = "\n".join([f"- {a}" for a in list(used_angles)[:15]])
            used_section = f"\n**【已使用过的角度，请避开这些内容】**:\n{used_list}"

        directions_str = "、".join(directions)
        event_text = event_info if event_info.strip() else topic

        prompt = ANGLE_GENERATION_PROMPT.format(
            num_angles=num_angles,
            topic=topic,
            event_info=event_text[:1500],
            stance=stance,
            directions=directions_str,
            used_angles_section=used_section
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个游戏舆情分析师，擅长从不同角度分析玩家关注点。请以JSON格式输出。"},
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
        两阶段生成：先定角度，再写评论

        Stage 1: 生成多样化的评论切入点（角度）
        Stage 2: 基于角度逐一生成评论，确保内容不重复
        """
        num_comments = max(1, min(100, num_comments))
        random.seed(seed)

        # ---- Stage 0: RAG 检索参考评论 ----
        retrieved = self.rag_retriever.retrieve_for_directions(
            topic, num_comments, directions, mmr_lambda=mmr_lambda,
            event_info=event_info, seed=seed
        )
        reference = [r["comment"] for r in retrieved]

        # ---- Stage 1: 生成评论角度 ----
        # 生成略多于需求的角度数，确保有足够多样性
        num_angles = min(num_comments + 5, num_comments * 2, 30)
        angles = self._generate_comment_angles(topic, num_angles, directions, stance, event_info)

        if not angles:
            # 回退：使用自动角度（无角度引导时给LLM更多自由）
            angles = []
            print("未生成角度，将使用自由生成模式")

        # ---- Stage 2: 基于角度生成评论 ----
        # 检索产品知识库
        product_section = self._retrieve_product_knowledge(topic, stance, top_k=10)

        # 构建prompt
        prompt = self._build_v2_prompt(
            topic=topic,
            num_comments=num_comments,
            directions=directions,
            reference=reference,
            stance=stance,
            event_info=event_info,
            product_section=product_section,
            angles=angles
        )

        comments = self._call_llm(prompt, num_comments, temperature)

        # 记录已使用的角度
        if angles and comments:
            self._add_used_angles(topic, angles[:len(comments)])

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

    def _build_v2_prompt(
        self,
        topic: str,
        num_comments: int,
        directions: List[str],
        reference: List[str],
        stance: str = "王者荣耀",
        event_info: str = "",
        product_section: str = "",
        angles: List[str] = None
    ) -> str:
        """构建 v2.0 精简 prompt"""

        # 方向规格
        num_directions = len(directions)
        base_count = num_comments // num_directions
        remainder = num_comments % num_directions

        direction_specs = []
        for i, d in enumerate(directions):
            count = base_count + (1 if i < remainder else 0)
            direction_specs.append(f"- {d}：生成{count}条")
        directions_text = "\n".join(direction_specs)

        # 方向定义
        direction_defs = """**【方向定义】**
- 正性向：赞扬、认可、推荐为主，可有小挑剔但整体正面
- 中性向：客观陈述为主，不偏向任一方。禁止"总体不错"、"值得一试"等总结
- 中正性向：偏正面但克制，不如正性向热情
"""

        # 事件背景
        event_section = f"\n**【事件详细背景】**\n{event_info}" if event_info.strip() else ""

        # 角度（Stage 1 的输出）
        angle_section = ""
        if angles:
            angle_list = "\n".join([f"{i+1}. {a}" for i, a in enumerate(angles)])
            angle_section = f"""
**【必须覆盖的评论切入点】**
以下每个切入点至少生成一条评论，确保每条评论对应不同的切入点：
{angle_list}

重要：每条评论必须从一个独特的切入点出发，不同评论之间不能有相似的角度。
"""
        else:
            angle_section = """
**【内容多样性要求】**
每条评论必须有独特的信息点和切入角度，严禁内容重复或高度相似。
"""

        # 参考评论
        ref_samples = reference[:15] if reference else []
        ref_text = "\n".join([f"- {c}" for c in ref_samples]) if ref_samples else "（无参考评论，请根据话题自由发挥）"

        # 长度分布计算
        num_long = max(1, num_comments // 20)
        num_mid = num_comments // 3
        num_short = num_comments - num_long - num_mid

        prompt = f"""**【话题】**
{topic}

**【立场】**
站在{stance}玩家的角度

**【需要生成的评论】**
{directions_text}
共{num_comments}条

{direction_defs}

**【长度分布要求】**
本批{num_comments}条中：{num_short}条短评(15-40字)、{num_mid}条中评(40-70字)、{num_long}条长评(75字以上)
短评不是水评，每条必须有清晰的实质内容。
{event_section}
{product_section if product_section else ""}
{angle_section}

**【参考评论（仅学习风格，不抄袭内容）】**
{ref_text}

**【输出格式】**
以JSON数组输出{num_comments}条评论，每条标注方向：
{{"comments": [{{"content": "评论内容", "direction": "正性向"}}, ...]}}
只输出JSON。"""

        return prompt

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
        生成带有特定视角的评论（v2.0 两阶段生成）
        """
        search_topic = f"{topic} {perspective} {stance}"

        # Stage 0: RAG 检索
        retrieved = self.rag_retriever.retrieve_for_directions(
            search_topic, num_comments, [direction],
            mmr_lambda=mmr_lambda, seed=seed
        )
        reference = [r["comment"] for r in retrieved]

        # Stage 1: 生成角度（带视角标记）
        angles = self._generate_comment_angles(
            topic=f"{topic}（从{perspective}视角）",
            num_angles=min(num_comments + 3, 15),
            directions=[direction],
            stance=stance,
            event_info=event_info
        )

        # 产品知识库
        product_section = self._retrieve_product_knowledge(topic, stance)

        # Stage 2: 构建 prompt 并生成
        angle_section = ""
        if angles:
            angle_list = "\n".join([f"{i+1}. {a}" for i, a in enumerate(angles)])
            angle_section = f"\n**【必须覆盖的切入点】**\n{angle_list}"

        ref_text = "\n".join([f"- {c}" for c in reference[:10]])

        event_text = f"\n**【事件背景】**\n{event_info}" if event_info.strip() else ""

        num_long = max(1, num_comments // 10)
        num_mid = num_comments // 3
        num_short = num_comments - num_long - num_mid

        prompt = f"""**【话题】**
{topic}

**【模拟视角】**
从"{perspective}"的视角发表评论

**【立场】**
始终站在{stance}的立场

**【评论方向】**
{direction}

方向定义：{"赞扬、认可、推荐为主" if direction == "正性向" else ("客观陈述为主，禁止总结性评价" if direction == "中性向" else "偏正面但克制，不热情")}

**【长度分布】**
{num_short}条短评(15-40字)、{num_mid}条中评(40-70字)、{num_long}条长评(75字+)
{event_text}
{product_section if product_section else ""}
{angle_section}

**【参考评论风格】**
{ref_text}

**【输出格式】**
以JSON数组输出{num_comments}条评论：
{{"comments": ["评论1", "评论2", ...]}}
只输出JSON。"""

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
