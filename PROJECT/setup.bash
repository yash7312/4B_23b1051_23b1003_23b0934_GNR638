#!/bin/bash

set -e

echo "=============================="
echo "Setting up Project 1 Environment"
echo "=============================="

# -------------------------------
# 0. Create models directory
# -------------------------------
mkdir -p models

# -------------------------------
# 1. Create environment
# -------------------------------
conda create -n gnr_project_env python=3.11 -y

# -------------------------------
# 2. Activate environment
# -------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gnr_project_env

# -------------------------------
# 3. Install dependencies
# -------------------------------
# -------------------------------
# 3. Install dependencies
# -------------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas tqdm pillow opencv-python

# Install Qwen dependencies
pip install "transformers>=4.45.0" accelerate huggingface_hub qwen-vl-utils

# Install the CUDA compiler toolkit into your conda environment
conda install -c nvidia cuda-toolkit=12.1 -y

# Point CUDA_HOME to the conda environment
export CUDA_HOME=$CONDA_PREFIX

# Install build dependencies
pip install ninja packaging

# Install flash-attn
pip install flash-attn --no-build-isolation

# Download Qwen2-VL-2B
echo "Downloading Qwen2-VL model into ./models..."
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="models/qwen",
    local_dir_use_symlinks=False,
    force_download=True
)
print("Qwen2-VL downloaded successfully.")
EOF

echo "Setup completed!"