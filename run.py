"""
评论写手系统 - 启动脚本
"""
import sys
import os

# 确保项目路径在sys.path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app


def main():
    """启动评论写手系统"""
    print("=" * 60)
    print("评论写手系统")
    print("=" * 60)

    # 检查环境变量
    from config import LLM_API_KEY
    if not LLM_API_KEY:
        print("\n警告: 未设置OPENAI_API_KEY环境变量")
        print("请在项目根目录创建.env文件并设置API密钥")
        print("或者设置系统环境变量: set OPENAI_API_KEY=your_key")

    # 创建并启动Gradio应用
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
