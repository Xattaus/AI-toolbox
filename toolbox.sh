#!/bin/bash
# ========================================
#   AI TOOLBOX - Linux/macOS Launcher
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}[ERROR] Python is not installed${NC}"
        echo "Please install Python 3.9 or higher"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}[INFO] Using Python $PYTHON_VERSION${NC}"
}

# Setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}[INFO] First-time setup - Creating virtual environment...${NC}"
        $PYTHON_CMD -m venv venv

        echo -e "${YELLOW}[INFO] Installing dependencies...${NC}"
        source venv/bin/activate
        pip install --upgrade pip --quiet
        pip install -e . --quiet

        echo -e "${GREEN}[SUCCESS] Setup complete!${NC}"
    else
        source venv/bin/activate
    fi
}

# Main
check_python
setup_venv

# Run AI Toolbox
python -m ai_toolbox.main "$@"
