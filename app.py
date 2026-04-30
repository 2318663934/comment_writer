"""
Gradio Web界面 - 评论写手系统
"""
import sys
sys.path.insert(0, "e:/评论写手")

import gradio as gr
from rag_retriever import RAGRetriever
from comment_generator import CommentGenerator
from vector_store import VectorStore
from multimodal_extractor import create_extractor
from starlette.middleware.base import BaseHTTPMiddleware

class AllowIframeMiddleware(BaseHTTPMiddleware):
    """移除 X-Frame-Options 头，允许被 AI Hub iframe 嵌入"""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.pop('x-frame-options', None)
        response.headers.pop('X-Frame-Options', None)
        return response


class CommentWriterApp:
    """评论写手应用"""

    def __init__(self):
        self.vector_store = None
        self.rag_retriever = None
        self.generator = None
        self.extractor = None
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

        # 初始化多模态提取器（可选）
        self.extractor = create_extractor()

    def is_ready(self) -> bool:
        """检查系统是否就绪"""
        return self.generator is not None

    # ============================================================
    # 媒体信息提取（独立步骤，返回给用户审核）
    # ============================================================

    def extract_media_info(self, media_path: str, video_url: str = "", stance: str = "") -> str:
        """
        提取图片/视频中的关键信息，返回给用户审核编辑
        stance: 产品立场，用于过滤无关内容
        """
        if not self.extractor:
            return ("【错误】Ollama 多模态未启用。\n"
                    "请在 .env 中设置 OLLAMA_ENABLED=true 并重启服务。")

        if not media_path and not (video_url and video_url.strip()):
            return ""

        is_weibo = video_url and ('weibo.com' in video_url or 'weibo.cn' in video_url or 't.cn' in video_url)
        is_douyin = video_url and 'douyin.com' in video_url

        try:
            result = self.extractor.extract(
                media_path=media_path if media_path else None,
                video_url=video_url.strip() if video_url else None,
                focus=stance if stance and stance != "其他" else ""
            )
            if result:
                print(f"[媒体提取] 成功，{len(result)} 字")
                return result
            else:
                if is_weibo:
                    return ("【提取失败】微博视频需要登录态。\n\n"
                            "1. 浏览器登录 weibo.com\n"
                            "2. 用 EditThisCookie 扩展导出 cookies.txt\n"
                            "3. .env 中设置 YTDLP_COOKIE_FILE=路径/cookies.txt\n"
                            "4. 重启服务\n\n或直接下载视频后上传本地文件。")
                if is_douyin:
                    return ("【提取失败】抖音视频提取失败。\n\n"
                            "请检查链接格式：\n"
                            "- 正确格式：包含 /video/{数字ID} 或 modal_id={数字ID}\n"
                            "- 不支持：搜索页/列表页链接（如 /jingxuan/search/）\n\n"
                            "如何获取正确的视频链接：\n"
                            "1. 在抖音网页版点击进入具体视频\n"
                            "2. 复制浏览器地址栏中的 URL\n"
                            "3. URL 应包含 /video/ 后跟一串数字\n\n"
                            "或将视频下载到本地后上传文件。")
                return "【提取失败】未能从媒体中提取信息。请检查链接是否有效，或下载后上传本地文件。"
        except Exception as e:
            return f"【提取出错】{str(e)}"

    # ============================================================
    # 评论生成
    # ============================================================

    def _merge_media_context(self, media_extracted_text: str, event_info: str) -> str:
        """将用户确认的媒体提取内容合并到事件背景"""
        if not media_extracted_text or not media_extracted_text.strip():
            return event_info

        parts = [f"[从媒体提取的信息]\n{media_extracted_text.strip()}"]
        if event_info.strip():
            parts.append(f"[用户提供的事件背景]\n{event_info.strip()}")
        return "\n\n".join(parts)

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
        seed: int = 42,
        media_extracted_text: str = ""
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
            diversity: 检索多样性
            seed: 随机种子
            media_extracted_text: 用户确认的媒体提取内容（可选）
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

        # 合并媒体提取内容到事件背景
        merged_event = self._merge_media_context(media_extracted_text, event_info)

        try:
            all_comments = self.generator.generate_for_directions(
                topic=topic.strip(),
                num_comments=num_comments,
                directions=directions,
                stance=stance,
                event_info=merged_event.strip() if merged_event else "",
                temperature=temperature,
                mmr_lambda=diversity,
                seed=seed
            )

            if not all_comments:
                return "生成失败，请检查API配置或重试"

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
        seed: int = 42,
        media_extracted_text: str = ""
    ) -> str:
        """
        带视角生成评论
        """
        if not self.is_ready():
            return "系统未就绪，请确保已运行 build_database.py 构建数据库"

        if not topic or not topic.strip():
            return "请输入话题"

        if not perspective or not perspective.strip():
            return "请输入视角"

        if stance == "其他":
            if not stance_custom or not stance_custom.strip():
                return "请输入产品名称"
            stance = stance_custom.strip()

        # 合并媒体提取内容到事件背景
        merged_event = self._merge_media_context(media_extracted_text, event_info)

        try:
            all_comments = []
            num_directions = len(directions)
            base_count = num_comments // num_directions
            remainder = num_comments % num_directions

            for i, direction in enumerate(directions):
                dir_count = base_count + (1 if i < remainder else 0)
                if dir_count == 0:
                    continue

                comments = self.generator.generate_with立场(
                    topic=topic.strip(),
                    perspective=perspective.strip(),
                    num_comments=dir_count,
                    direction=direction,
                    stance=stance,
                    event_info=merged_event.strip() if merged_event else "",
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
                extra = ""
                if self.extractor:
                    extra += " | Ollama 多模态已连接"
                return f"✅ 就绪 - 数据库包含 {entities} 条评论{extra}"
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

        # ============================================================
        # Tab 1: 基础生成
        # ============================================================
        with gr.Tab("基础生成"):
            topic_input = gr.Textbox(
                label="话题",
                placeholder="事件标签，例如：#王者你已急哭头像框#、#洛克王国世界元宵喜乐会#...",
                lines=2
            )

            with gr.Row():
                with gr.Column(scale=1):
                    num_input = gr.Slider(
                        minimum=1, maximum=100, value=10, step=1,
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
                        minimum=0.1, maximum=1.0, value=0.8, step=0.1,
                        label="LLM温度", info="0.1=更准确, 1.0=更多样"
                    )
                with gr.Column(scale=1):
                    diversity_slider = gr.Slider(
                        minimum=0.3, maximum=1.0, value=0.7, step=0.1,
                        label="检索多样性", info="0.3=高多样, 1.0=高相关"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    stance_dropdown = gr.Dropdown(
                        choices=["王者荣耀", "DNF端游", "金铲铲之战", "无畏契约手游", "洛克王国世界", "王者荣耀世界", "其他"],
                        value="王者荣耀", label="立场（产品）"
                    )
                with gr.Column(scale=1):
                    stance_custom_input = gr.Textbox(
                        label='产品名称（选"其他"时填写）',
                        placeholder="输入产品名称", lines=1, visible=False
                    )
                with gr.Column(scale=1):
                    seed_input = gr.Number(
                        value=42, label="随机种子", info="同一种子可复现结果"
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

            with gr.Accordion("图片/视频辅助（可选，需 Ollama）", open=False):
                gr.Markdown("上传图片/视频文件，或粘贴视频链接，点击「提取信息」后审核编辑")
                media_input = gr.File(
                    label="上传图片或视频（文件）",
                    file_types=["image", "video"],
                    type="filepath"
                )
                media_url_input = gr.Textbox(
                    label="或输入视频链接（B站/抖音），其他平台请下载后上传",
                    placeholder="https://...",
                    lines=1
                )
                with gr.Row():
                    extract_btn = gr.Button("提取信息", variant="secondary", size="sm")
                    clear_media_btn = gr.Button("清空", size="sm")

                media_extracted_output = gr.Textbox(
                    label="提取到的信息（可编辑）",
                    placeholder="点击「提取信息」后，模型提取的内容将显示在这里，你可以修改后再生成评论",
                    lines=8, interactive=True
                )

            generate_btn = gr.Button("生成评论", variant="primary")

            output_box = gr.Textbox(label="生成的评论", lines=15)

            # 事件绑定
            extract_btn.click(
                fn=app.extract_media_info,
                inputs=[media_input, media_url_input, stance_dropdown],
                outputs=[media_extracted_output]
            )
            clear_media_btn.click(
                fn=lambda: ("", "", ""),
                inputs=[],
                outputs=[media_input, media_url_input, media_extracted_output]
            )
            generate_btn.click(
                fn=app.generate_comments,
                inputs=[
                    topic_input, num_input, direction_checkbox,
                    stance_dropdown, stance_custom_input, event_info_input,
                    temperature_slider, diversity_slider, seed_input,
                    media_extracted_output
                ],
                outputs=output_box
            )

        # ============================================================
        # Tab 2: 多视角生成
        # ============================================================
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
                        minimum=1, maximum=100, value=5, step=1,
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
                        minimum=0.1, maximum=1.0, value=0.8, step=0.1,
                        label="LLM温度", info="0.1=更准确, 1.0=更多样"
                    )
                with gr.Column(scale=1):
                    diversity_slider2 = gr.Slider(
                        minimum=0.3, maximum=1.0, value=0.7, step=0.1,
                        label="检索多样性", info="0.3=高多样, 1.0=高相关"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    stance_dropdown2 = gr.Dropdown(
                        choices=["王者荣耀", "DNF端游", "金铲铲之战", "无畏契约手游", "洛克王国世界", "王者荣耀世界", "其他"],
                        value="王者荣耀", label="立场（产品）"
                    )
                with gr.Column(scale=1):
                    stance_custom_input2 = gr.Textbox(
                        label='产品名称（选"其他"时填写）',
                        placeholder="输入产品名称", lines=1, visible=False
                    )
                with gr.Column(scale=1):
                    seed_input2 = gr.Number(
                        value=42, label="随机种子", info="同一种子可复现结果"
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

            with gr.Accordion("图片/视频辅助（可选，需 Ollama）", open=False):
                gr.Markdown("上传图片/视频文件，或粘贴视频链接，点击「提取信息」后审核编辑")
                media_input2 = gr.File(
                    label="上传图片或视频（文件）",
                    file_types=["image", "video"],
                    type="filepath"
                )
                media_url_input2 = gr.Textbox(
                    label="或输入视频链接（B站/抖音），其他平台请下载后上传",
                    placeholder="https://...",
                    lines=1
                )
                with gr.Row():
                    extract_btn2 = gr.Button("提取信息", variant="secondary", size="sm")
                    clear_media_btn2 = gr.Button("清空", size="sm")

                media_extracted_output2 = gr.Textbox(
                    label="提取到的信息（可编辑）",
                    placeholder="点击「提取信息」后，模型提取的内容将显示在这里，你可以修改后再生成评论",
                    lines=8, interactive=True
                )

            generate_btn2 = gr.Button("生成评论", variant="primary")

            output_box2 = gr.Textbox(label="生成的评论", lines=15)

            # 事件绑定
            extract_btn2.click(
                fn=app.extract_media_info,
                inputs=[media_input2, media_url_input2, stance_dropdown2],
                outputs=[media_extracted_output2]
            )
            clear_media_btn2.click(
                fn=lambda: ("", "", ""),
                inputs=[],
                outputs=[media_input2, media_url_input2, media_extracted_output2]
            )
            generate_btn2.click(
                fn=app.generate_with_perspective,
                inputs=[
                    topic_input2, perspective_input, num_input2, direction_checkbox2,
                    stance_dropdown2, stance_custom_input2, event_info_input2,
                    temperature_slider2, diversity_slider2, seed_input2,
                    media_extracted_output2
                ],
                outputs=output_box2
            )

        # ============================================================
        # Tab 3: 使用说明
        # ============================================================
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

### 6. 图片/视频辅助（需 Ollama）
1. 展开"图片/视频辅助"折叠区
2. 上传游戏截图或视频文件
3. 点击「提取信息」——系统通过多模态模型提取关键事件信息
4. 审核并修改提取到的内容
5. 点击「生成评论」——修改后的内容将作为事件背景的一部分
6. 需要先在 .env 中设置 `OLLAMA_ENABLED=true`
            """)

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.app.add_middleware(AllowIframeMiddleware)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
