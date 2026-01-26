@echo off
setlocal
cd /d "%~dp0"
call venv\Scripts\activate
python -m src.ai_toolbox.tui.app
pause
