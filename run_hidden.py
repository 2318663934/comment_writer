import subprocess
import sys
import os
import ctypes

# Windows API to hide console window
SEM_NOGPFAULTERRORBOX = 0x0002
ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)

# Hide the console window
info = subprocess.STARTUPINFO()
info.dwFlags = subprocess.STARTF_USESHOWWINDOW
info.wShowWindow = 0  # SW_HIDE = 0

# Get the virtual environment python path
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commenter', 'Scripts', 'python.exe')

# Run the original run.py with console fully hidden
process = subprocess.Popen(
    [venv_python, 'run.py'],
    startupinfo=info,
    cwd=os.path.dirname(os.path.abspath(__file__)),
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
process.wait()
