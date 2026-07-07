@echo off
REM ========================================
REM   AI TOOLBOX - Windows Launcher
REM ========================================

setlocal enabledelayedexpansion

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [INFO] First-time setup - Creating virtual environment...
    echo.
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo [ERROR] Make sure Python 3.9+ is installed and in PATH
        pause
        exit /b 1
    )

    echo [INFO] Installing dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip --quiet
    pip install -e . --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Setup complete!
    echo.
) else (
    call venv\Scripts\activate.bat
)

REM Check if reinstall is needed (marker file)
if exist ".reinstall_marker" (
    echo [INFO] Updating package with new modules...
    pip install -e . --quiet
    del .reinstall_marker
)

REM Run AI Toolbox
python -m ai_toolbox.main %*

REM Keep window open if no arguments
if "%~1"=="" (
    echo.
)
