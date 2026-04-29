#!/usr/bin/env bash
# Sets up miniforge3 at ~/miniforge3 and creates the 'RL' conda env with vllm.
# Safe to re-run: skips steps already done.
set -euo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
MINIFORGE_INSTALLER="/tmp/Miniforge3-aarch64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
ENV_YAML="$(dirname "$0")/environment.yaml"

# ── 1. Install miniforge if missing ─────────────────────────────────────────
if [[ ! -f "$MINIFORGE_DIR/bin/conda" ]]; then
    echo "[setup] Downloading Miniforge3 for aarch64…"
    curl -fsSL "$MINIFORGE_URL" -o "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
    rm -f "$MINIFORGE_INSTALLER"
    echo "[setup] Miniforge3 installed at $MINIFORGE_DIR"
else
    echo "[setup] Miniforge3 already present at $MINIFORGE_DIR"
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# ── 2. Create / update RL env ────────────────────────────────────────────────
if conda env list | grep -q "^RL "; then
    echo "[setup] Conda env 'RL' already exists — updating from $ENV_YAML"
    conda env update -n RL -f "$ENV_YAML" --prune
else
    echo "[setup] Creating conda env 'RL' from $ENV_YAML"
    conda env create -f "$ENV_YAML"
fi

conda activate RL

# ── 3. Install vllm wheel for JetPack / aarch64 ──────────────────────────────
# vllm 0.8.4 requires a pre-built wheel for aarch64 + CUDA 11.x (JetPack 5).
# Try pip first (works if a CUDA-capable wheel is available); fall back to
# the Jetson Community wheels hosted by dusty-nv.
if ! python -c "import vllm" 2>/dev/null; then
    echo "[setup] vllm not importable — attempting installation…"
    JETSON_VLLM_URL="https://github.com/dusty-nv/jetson-containers/raw/master/packages/llm/vllm/wheels/vllm-0.8.4+cu118-cp310-cp310-linux_aarch64.whl"
    if curl -fsSL --head "$JETSON_VLLM_URL" 2>/dev/null | grep -q "200"; then
        pip install "$JETSON_VLLM_URL" --no-deps
    else
        echo "[setup] Jetson wheel not found — falling back to pip vllm (may need CUDA torch)"
        pip install "vllm==0.8.4"
    fi
else
    echo "[setup] vllm already importable in 'RL' env."
fi

echo ""
echo "[setup] Done. Activate with:"
echo "  source $MINIFORGE_DIR/etc/profile.d/conda.sh && conda activate RL"
