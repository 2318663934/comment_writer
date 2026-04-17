"""
评论写手系统 - 启动脚本
"""
import sys
import os
import subprocess

# 确保项目路径在sys.path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 检查是否为PM2环境（通过环境变量判断）
if os.environ.get("PM2_HOME") or os.environ.get("PM2_LIST"):
    # PM2环境下，隐藏控制台窗口
    import ctypes
    SEM_NOGPFAULTERRORBOX = 0x0002
    ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
    info = subprocess.STARTUPINFO()
    info.dmFlags = subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0  # SW_HIDE = 0

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
