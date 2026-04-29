#!/usr/bin/env bash
# setup_wsl.sh — Full environment setup + simulation runner for WSL2 (Ubuntu x86_64)
#
# Run this ONCE to install everything, then use simulation.sh for daily use.
#
# Usage:
#   cd /mnt/e/AGX-BACKUP/VLN-DAPO-main
#   bash setup_wsl.sh
#
# After setup, run the simulation:
#   bash simulation.sh --scene-id 1LXtFkjw3qL \
#       --instruction "Walk to the kitchen counter." --show-window

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

CONDA_ENV="RL"
PYTHON_VER="3.10"
VLLM_VER="0.8.4"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO}/global_step_450/actor/huggingface}"

echo "================================================================"
echo " VLN-DAPO WSL2 Setup"
echo " Repo   : $REPO"
echo " Env    : $CONDA_ENV  (Python $PYTHON_VER)"
echo "================================================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[setup] Installing system dependencies..."
echo "equanimity" | sudo -S apt-get update -qq
echo "equanimity" | sudo -S apt-get install -y -qq \
    wget curl git unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libjpeg-dev libpng-dev ffmpeg \
    build-essential cmake

# ── 2. Miniconda (if not present) ─────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[setup] Installing Miniconda3..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "[setup] Miniconda installed."
else
    eval "$(conda shell.bash hook)"
    echo "[setup] conda found: $(conda --version)"
fi

# ── 3. Create / refresh RL conda env ──────────────────────────────────────────
if conda env list | grep -qw "^${CONDA_ENV}"; then
    echo "[setup] Conda env '${CONDA_ENV}' already exists — updating."
    conda activate "$CONDA_ENV"
else
    echo "[setup] Creating conda env '${CONDA_ENV}' (Python ${PYTHON_VER})..."
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
    conda activate "$CONDA_ENV"
fi

# ── 4. habitat-sim (Linux x86_64 — aihabitat conda channel) ──────────────────
if ! python -c "import habitat_sim" &>/dev/null 2>&1; then
    echo "[setup] Installing habitat-sim 0.2.5 via conda (aihabitat channel)..."
    conda install -y \
        -c aihabitat \
        -c conda-forge \
        "habitat-sim=0.2.5" \
        "withbullet" \
        "headless"
else
    echo "[setup] habitat-sim already installed."
fi

# ── 5. habitat-lab ────────────────────────────────────────────────────────────
if ! python -c "import habitat" &>/dev/null 2>&1; then
    echo "[setup] Installing habitat-lab 0.2.5..."
    pip install "habitat-lab==0.2.5" --quiet
else
    echo "[setup] habitat-lab already installed."
fi

# ── 6. Core Python dependencies ───────────────────────────────────────────────
echo "[setup] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    "openai>=1.0.0" \
    "qwen-vl-utils>=0.0.8" \
    "opencv-python" \
    "pillow" \
    "imageio" \
    "imageio[ffmpeg]" \
    "tqdm" \
    "requests" \
    "httpx" \
    "transformers>=4.47.0" \
    "accelerate" \
    "hydra-core" \
    "omegaconf" \
    "ray[default]" \
    "flask" \
    "dtw-python" \
    "fastdtw"

# ── 7. PyTorch (CUDA — skip if already present) ───────────────────────────────
if ! python -c "import torch; assert torch.cuda.is_available()" &>/dev/null 2>&1; then
    echo "[setup] Installing PyTorch with CUDA 12.1..."
    pip install --quiet \
        "torch==2.3.1" \
        "torchvision==0.18.1" \
        --index-url https://download.pytorch.org/whl/cu121
else
    echo "[setup] PyTorch+CUDA already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# ── 8. vLLM ───────────────────────────────────────────────────────────────────
if ! python -c "import vllm" &>/dev/null 2>&1; then
    echo "[setup] Installing vLLM ${VLLM_VER}..."
    pip install --quiet "vllm==${VLLM_VER}"
else
    echo "[setup] vLLM already installed: $(python -c 'import vllm; print(vllm.__version__)')"
fi

# ── 9. Install repo packages (vlnce_server + VLN_CE) ─────────────────────────
echo "[setup] Installing vlnce_server package..."
pip install --quiet -e "${REPO}/vlnce_server" --no-deps

echo "[setup] Setting PYTHONPATH..."
PPATH="${REPO}:${REPO}/vlnce_server"
if ! grep -q "VLN-DAPO PYTHONPATH" "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<EOF

# VLN-DAPO PYTHONPATH
export PYTHONPATH="${PPATH}\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
fi
export PYTHONPATH="${PPATH}${PYTHONPATH:+:$PYTHONPATH}"

# ── 10. Download R2R VLN-CE dataset (if missing) ─────────────────────────────
R2R_DIR="${REPO}/data/datasets/R2R_VLNCE_v1-3_preprocessed"
if [[ ! -d "$R2R_DIR" ]]; then
    echo "[setup] Downloading R2R VLN-CE v1-3 preprocessed dataset..."
    mkdir -p "${REPO}/data/datasets"
    cd "${REPO}/data/datasets"
    # Official VLN-CE dataset from Facebook AI Research
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/habitat/data/datasets/vln/r2r/v1-3/R2R_VLNCE_v1-3_preprocessed.zip" \
        -O R2R_VLNCE_v1-3_preprocessed.zip
    echo "[setup] Extracting dataset..."
    unzip -q R2R_VLNCE_v1-3_preprocessed.zip
    rm R2R_VLNCE_v1-3_preprocessed.zip
    cd "$REPO"
    echo "[setup] R2R dataset ready."
else
    echo "[setup] R2R dataset already present at ${R2R_DIR}"
fi

# ── 11. Verify MP3D scenes exist ──────────────────────────────────────────────
MP3D_DIR="${REPO}/data/scene_datasets/mp3d"
GLB_COUNT=$(find "$MP3D_DIR" -name "*.glb" 2>/dev/null | wc -l)
echo "[setup] MP3D scenes found: ${GLB_COUNT} .glb files"
if [[ $GLB_COUNT -eq 0 ]]; then
    echo "[setup] WARNING: No MP3D .glb files found at ${MP3D_DIR}"
    echo "         Copy your MP3D data from Windows:"
    echo "         cp -r /mnt/e/AGX-BACKUP/VLN-DAPO-main/data/scene_datasets/mp3d ${MP3D_DIR}"
fi

# ── 12. Quick import check ────────────────────────────────────────────────────
echo ""
echo "[setup] Verifying imports..."
python - <<'PY'
import sys
ok = True
for mod in ["habitat", "openai", "cv2", "PIL", "qwen_vl_utils", "vllm", "torch"]:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except ImportError as e:
        print(f"  ✗ {mod}  ({e})")
        ok = False
import torch
cuda = torch.cuda.is_available()
print(f"  {'✓' if cuda else '✗'} CUDA available: {cuda}")
if cuda:
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
sys.exit(0 if ok else 1)
PY

echo ""
echo "================================================================"
echo " Setup complete!"
echo ""
echo " To run simulation:"
echo ""
echo "   conda activate RL"
echo "   cd $REPO"
echo "   bash simulation.sh \\"
echo "       --scene-id 1LXtFkjw3qL \\"
echo '       --instruction "Walk to the kitchen counter and stop." \'
echo "       --max-episodes 1 --show-window"
echo "================================================================"
