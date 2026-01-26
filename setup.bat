@echo off
REM ========================================
REM   AI TOOLBOX - Setup Script
REM ========================================

setlocal enabledelayedexpansion

echo.
echo  ========================================
echo     AI TOOLBOX - Setup
echo  ========================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add to PATH" during installation!
    pause
    exit /b 1
)

REM Show Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [INFO] Found %%i

echo.
echo [STEP 1/4] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment exists, updating...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo [STEP 2/4] Activating environment...
call venv\Scripts\activate.bat

echo.
echo [STEP 3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo [STEP 4/4] Installing AI Toolbox and components...
pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install AI Toolbox
    pause
    exit /b 1
)

pip install -e gguf-converter --quiet
if errorlevel 1 (
    echo [WARNING] Could not install GGUF Converter as editable
    echo [INFO] Installing dependencies...
    pip install rich click questionary huggingface-hub transformers safetensors tokenizers sentencepiece tqdm psutil gguf --quiet
)

echo.
echo  ========================================
echo     SETUP COMPLETE!
echo  ========================================
echo.
echo  To start AI Toolbox, run:
echo.
echo     toolbox.bat
echo.
echo  Or for GGUF Converter directly:
echo.
echo     gguf-converter\run.bat
echo.
echo  ========================================
echo.

pause
