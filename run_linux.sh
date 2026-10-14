#!/bin/bash
# Omok++ Linux Quick Launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed."
    echo "Please run: sudo apt update && sudo apt install -y python3 python3-tk python3-pip"
    exit 1
fi

echo "========================================================"
echo "  Launching Omok++ AI Arena (AlphaZero 2x Champion)    "
echo "  Working Directory: $DIR"
echo "========================================================"

python3 "$DIR/play.py"
