#!/usr/bin/env bash
set -e

CONDA_SH="/home/chitti/miniforge3/etc/profile.d/conda.sh"
ENV_NAME="RL"
SDK_DIR="/home/chitti/VLN-DAPO-main/unitree/unitree_legged_sdk"
SCRIPT="$SDK_DIR/run_dlink.py"

source "$CONDA_SH"
conda activate "$ENV_NAME"

PYTHON=$(conda run -n "$ENV_NAME" which python)
PY_TAG=$("$PYTHON" -c "import sys; print('cpython-{}{}'.format(*sys.version_info[:2]))")
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
SO="$SDK_DIR/lib/python/$ARCH/robot_interface.${PY_TAG}-${ARCH}-linux-gnu.so"

if [ ! -f "$SO" ]; then
    echo ">>> Building robot_interface for $PY_TAG ($ARCH)..."
    BUILD_DIR="$SDK_DIR/build_py"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    conda run -n "$ENV_NAME" cmake "$SDK_DIR/python_wrapper" \
        -DPYTHON_EXECUTABLE="$PYTHON" \
        -DCMAKE_BUILD_TYPE=Release
    conda run -n "$ENV_NAME" make -j"$(nproc)"
    echo ">>> Build complete: $SO"
else
    echo ">>> robot_interface already built for $PY_TAG"
fi

# Install pynput if missing
conda run -n "$ENV_NAME" python -c "import pynput" 2>/dev/null || \
    conda run -n "$ENV_NAME" pip install pynput -q

echo ">>> Launching run_dlink.py..."
cd "$SDK_DIR"
conda run -n "$ENV_NAME" python "$SCRIPT"
