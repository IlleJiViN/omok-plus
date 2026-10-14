#!/bin/bash
set -e

echo "=========================================================="
echo "  Setting up Omok++ AI Arena Environment on Linux/Ubuntu  "
echo "=========================================================="

echo "[1/3] Installing Python 3, Tkinter GUI, and pip..."
sudo apt update
sudo apt install -y python3 python3-pip python3-tk python-is-python3 unzip

echo "[2/3] Installing PyTorch, NumPy, Numba..."
# Detect if NVIDIA GPU is present
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip3 install torch
else
    echo "CPU only detected. Installing CPU PyTorch..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu || pip3 install torch
fi

pip3 install numpy numba

echo "[3/3] Setting execution permissions..."
chmod +x run_linux.sh

echo "=========================================================="
echo "  SUCCESS! Environment setup is complete.               "
echo "  To start the game, run: ./run_linux.sh                "
echo "  Or simply: python3 play.py                            "
echo "=========================================================="
