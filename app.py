"""
Gradio Web界面 - 评论写手系统
"""
import sys
sys.path.insert(0, "e:/评论写手")

import gradio as gr
from rag_retriever import RAGRetriever
from comment_generator import CommentGenerator
from vector_store import VectorStore


class CommentWriterApp:
    """评论写手应用"""

    def __init__(self):
        self.vector_store = None
        self.rag_retriever = None
        self.generator = None
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        try:
            self.vector_store = VectorStore()
            stats = self.vector_store.get_collection_stats()

            if stats.get("exists") and stats.get("entities", 0) > 0:
                self.rag_retriever = RAGRetriever(self.vector_store)
                self.generator = CommentGenerator(self.rag_retriever)
                print(f"系统就绪: 数据库包含 {stats['entities']} 条评论")
            else:
                print("警告: 数据库为空，请先运行 build_database.py")
        except Exception as e:
            print(f"初始化失败: {e}")

    def is_ready(self) -> bool:
        """检查系统是否就绪"""
        return self.generator is not None

    def generate_comments(
        self,
        topic: str,
        num_comments: int,
        directions: list,
        stance: str,
        stance_custom: str,
        event_info: str,
        temperature: float = 0.8,
        diversity: float = 0.7,
        seed: int = 42
    ) -> str:
        """
        生成评论

        Args:
            topic: 话题
            num_comments: 数量
            directions: 方向列表（可多选）
            stance: 立场（产品）
            stance_custom: 自定义产品名称（"其他"选项时使用）
            event_info: 事件背景（可选）
            temperature: LLM温度
            diversity: 检索多样性（MMR lambda），越低越多样
            seed: 随机种子

        Returns:
            格式化后的评论文本
        """
        if not self.is_ready():
            return "系统未就绪，请确保已运行 build_database.py 构建数据库"

        if not topic or not topic.strip():
            return "请输入话题"

        # 处理"其他"选项
        if stance == "其他":
            if not stance_custom or not stance_custom.strip():
                return "请输入产品名称"
            stance = stance_custom.strip()

        try:
            # 一次性生成所有方向的评论
            all_comments = self.generator.generate_for_directions(
                topic=topic.strip(),
                num_comments=num_comments,
                directions=directions,
                stance=stance,
                event_info=event_info.strip() if event_info else "",
                temperature=temperature,
                mmr_lambda=diversity,
                seed=seed
            )

            if not all_comments:
                return "生成失败，请检查API配置或重试"

            # 格式化输出
            result = []
            for i, c in enumerate(all_comments, 1):
                result.append(f"{i}. {c}")

            return "\n".join(result)

        except Exception as e:
            return f"生成失败: {str(e)}"

    def generate_with_perspective(
        self,
        topic: str,
        perspective: str,
        num_comments: int,
        directions: list,
        stance: str,
        stance_custom: str,
        event_info: str,
        temperature: float = 0.8,
        diversity: float = 0.7,
        seed: int = 42
    ) -> str:
        """
        带视角生成评论

        Args:
            topic: 话题
            perspective: 视角
            num_comments: 数量
            directions: 方向列表（可多选）
            stance: 立场（产品）
            stance_custom: 自定义产品名称（"其他"选项时使用）
            event_info: 事件背景（可选）
            temperature: LLM温度
            diversity: 检索多样性（MMR lambda）
            seed: 随机种子

        Returns:
            格式化后的评论文本
        """
        if not self.is_ready():
            return "系统未就绪，请确保已运行 build_database.py 构建数据库"

        if not topic or not topic.strip():
            return "请输入话题"

        if not perspective or not perspective.strip():
            return "请输入视角"

        # 处理"其他"选项
        if stance == "其他":
            if not stance_custom or not stance_custom.strip():
                return "请输入产品名称"
            stance = stance_custom.strip()

        try:
            # 计算每个方向的评论数量
            all_comments = []
            num_directions = len(directions)
            base_count = num_comments // num_directions
            remainder = num_comments % num_directions

            for i, direction in enumerate(directions):
                # 分配评论数量，轮流分配多出来的评论
                dir_count = base_count + (1 if i < remainder else 0)
                if dir_count == 0:
                    continue

                comments = self.generator.generate_with立场(
                    topic=topic.strip(),
                    perspective=perspective.strip(),
                    num_comments=dir_count,
                    direction=direction,
                    stance=stance,
                    event_info=event_info.strip() if event_info else "",
                    temperature=temperature,
                    mmr_lambda=diversity,
                    seed=seed
                )

                if comments:
                    all_comments.extend(comments)

            if not all_comments:
                return "生成失败，请检查API配置或重试"

            result = []
            for i, c in enumerate(all_comments, 1):
                result.append(f"{i}. {c}")

            return "\n".join(result)

        except Exception as e:
            return f"生成失败: {str(e)}"

    def get_status(self) -> str:
        """获取系统状态"""
        if not self.vector_store:
            return "❌ 未连接到Milvus"

        try:
            stats = self.vector_store.get_collection_stats()
            if stats.get("exists"):
                entities = stats.get("entities", 0)
                return f"✅ 就绪 - 数据库包含 {entities} 条评论"
            else:
                return "❌ 数据库集合不存在"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"


def create_app() -> gr.Blocks:
    """创建Gradio应用"""

    app = CommentWriterApp()

    with gr.Blocks(title="评论写手") as demo:
        gr.Markdown("# 评论写手系统")
        gr.Markdown("基于LLM和RAG的智能评论生成系统")

        # 状态显示
        status = app.get_status()
        status_box = gr.Textbox(label="系统状态", value=status, interactive=False)

        with gr.Tab("基础生成"):
            topic_input = gr.Textbox(
                label="话题",
                placeholder="事件标签，例如：#王者你已急哭头像框#、#洛克王国世界元宵喜乐会#...",
                lines=2
            )

            with gr.Row():
                with gr.Column(scale=1):
                    num_input = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                        label="评论数量"
                    )
                with gr.Column(scale=1):
                    direction_checkbox = gr.CheckboxGroup(
                        choices=["正性向", "中性向", "中正性向"],
                        value=["正性向"],
                        label="评论方向（可多选）"
                    )
                with gr.Column(scale=1):
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="LLM温度",
                        info="0.1=更准确, 1.0=更多样"
                    )
                with gr.Column(scale=1):
                    diversity_slider = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="检索多样性",
                        info="0.3=高多样, 1.0=高相关"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    stance_dropdown = gr.Dropdown(
                        choices=["王者荣耀", "DNF端游", "金铲铲之战", "无畏契约手游", "洛克王国世界", "王者荣耀世界", "其他"],
                        value="王者荣耀",
                        label="立场（产品）"
                    )
                with gr.Column(scale=1):
                    stance_custom_input = gr.Textbox(
                        label='产品名称（选"其他"时填写）',
                        placeholder="输入产品名称",
                        lines=1,
                        visible=False
                    )
                with gr.Column(scale=1):
                    seed_input = gr.Number(
                        value=42,
                        label="随机种子",
                        info="同一种子可复现结果"
                    )

            def update_stance_visibility(stance):
                return gr.update(visible=(stance == "其他"))

            stance_dropdown.change(
                fn=update_stance_visibility,
                inputs=[stance_dropdown],
                outputs=[stance_custom_input]
            )

            event_info_input = gr.Textbox(
                label="事件背景（可选）",
                placeholder="可粘贴事件相关文章、补充事件的来龙去脉等，越详细生成的多样性越丰富",
                lines=5
            )

            generate_btn = gr.Button("生成评论", variant="primary")

            output_box = gr.Textbox(
                label="生成的评论",
                lines=15
            )

            generate_btn.click(
                fn=app.generate_comments,
                inputs=[topic_input, num_input, direction_checkbox, stance_dropdown, stance_custom_input, event_info_input, temperature_slider, diversity_slider, seed_input],
                outputs=output_box
            )

        with gr.Tab("多视角生成"):
            gr.Markdown("### 带视角的评论生成")
            gr.Markdown("可以从不同人群的视角思考，但始终站在所选产品的立场")

            with gr.Row():
                with gr.Column(scale=2):
                    topic_input2 = gr.Textbox(
                        label="话题",
                        placeholder="事件标签，例如：#王者你已急哭头像框#、#洛克王国世界元宵喜乐会#...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    perspective_input = gr.Textbox(
                        label="视角",
                        placeholder="例如：原神玩家、王者荣耀主播...",
                        lines=2
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    num_input2 = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=5,
                        step=1,
                        label="评论数量"
                    )
                with gr.Column(scale=1):
                    direction_checkbox2 = gr.CheckboxGroup(
                        choices=["正性向", "中性向", "中正性向"],
                        value=["中正性向"],
                        label="评论方向（可多选）"
                    )
                with gr.Column(scale=1):
                    temperature_slider2 = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="LLM温度",
                        info="0.1=更准确, 1.0=更多样"
                    )
                with gr.Column(scale=1):
                    diversity_slider2 = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="检索多样性",
                        info="0.3=高多样, 1.0=高相关"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    stance_dropdown2 = gr.Dropdown(
                        choices=["王者荣耀", "DNF端游", "金铲铲之战", "无畏契约手游", "洛克王国世界", "王者荣耀世界", "其他"],
                        value="王者荣耀",
                        label="立场（产品）"
                    )
                with gr.Column(scale=1):
                    stance_custom_input2 = gr.Textbox(
                        label='产品名称（选"其他"时填写）',
                        placeholder="输入产品名称",
                        lines=1,
                        visible=False
                    )
                with gr.Column(scale=1):
                    seed_input2 = gr.Number(
                        value=42,
                        label="随机种子",
                        info="同一种子可复现结果"
                    )

            def update_stance_visibility2(stance):
                return gr.update(visible=(stance == "其他"))

            stance_dropdown2.change(
                fn=update_stance_visibility2,
                inputs=[stance_dropdown2],
                outputs=[stance_custom_input2]
            )

            event_info_input2 = gr.Textbox(
                label="事件背景（可选）",
                placeholder="可粘贴事件相关文章、补充事件的来龙去脉等，越详细生成的多样性越丰富",
                lines=5
            )

            generate_btn2 = gr.Button("生成评论", variant="primary")

            output_box2 = gr.Textbox(
                label="生成的评论",
                lines=15
            )

            generate_btn2.click(
                fn=app.generate_with_perspective,
                inputs=[topic_input2, perspective_input, num_input2, direction_checkbox2, stance_dropdown2, stance_custom_input2, event_info_input2, temperature_slider2, diversity_slider2, seed_input2],
                outputs=output_box2
            )

        with gr.Tab("使用说明"):
            gr.Markdown("""
## 使用说明

### 1. 数据库构建
首次使用请先运行数据库构建脚本：
```bash
cd e:/评论写手
./commenter/Scripts/python build_database.py
```

### 2. API配置
确保在项目根目录创建 `.env` 文件：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### 3. 基础生成
- 输入话题（必填）
- 调整评论数量（1-100）
- 选择评论方向
- 点击生成

### 4. 多视角生成
- 输入话题
- 输入要模拟的视角（如"原神玩家"）
- 系统会站在该视角生成评论，但仍支持王者荣耀

### 5. 评论方向说明
- **正性向**: 正面、赞扬、支持
- **中性向**: 客观分析、理性评价
- **中正性向**: 中立但偏正面
            """)

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
