#!/usr/bin/env bash
# VLN inference launcher — runs web UI + model inference together.
# Writes log to /tmp/infer_live.log (do NOT pipe this script's output to tee).
#
# Usage:
#   bash infer.sh                 # default instruction, robot live
#   bash infer.sh --dry-run       # no motor commands
#   INSTRUCTION="Go to kitchen" bash infer.sh --dry-run
#   UI_PORT=8080 CAMERA_ID=1 bash infer.sh --dry-run

set -euo pipefail

source /home/chitti/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-RL}"
conda activate "$CONDA_ENV"
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "[infer.sh] Using conda env: $CONDA_ENV"

LOG=/tmp/infer_live.log
INSTR_FILE=/tmp/vln_instruction.txt
RESTART_FLAG=/tmp/vln_restart.flag
FRAME_FILE=/tmp/vln_latest_frame.jpg
UI_PORT="${UI_PORT:-5000}"
VLLM_PORT="${VLLM_PORT:-8003}"
CAMERA_ID="${CAMERA_ID:-0}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_STEPS="${MAX_STEPS:-120}"
SHARED_FRAME_FPS="${SHARED_FRAME_FPS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./global_step_450/actor/huggingface}"
USE_VLLM=1
PY_SHIM_DIR="$PWD/.infer_pyshim"
# Leave INSTRUCTION unset to start paused and let the web UI provide one.
INSTRUCTION="${INSTRUCTION-}"

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    echo "[infer.sh] Refusing to start in dry-run mode. Remove --dry-run to run the real robot."
    exit 1
  fi
done

cleanup_existing_processes() {
  echo "[infer.sh] Cleaning up old processes…"
  # SIGTERM first, then SIGKILL to ensure vllm EngineCoreProc subprocesses
  # (which hold ~17 GiB of unified memory) are fully released.
  pkill -f "infer\.py" 2>/dev/null || true
  pkill -f "web_ui_voice\.py" 2>/dev/null || true
  pkill -f "vllm" 2>/dev/null || true
  pkill -f "EngineCoreProc" 2>/dev/null || true

  for i in 1 2 3 4 5; do
    pgrep -f "infer\.py\|web_ui_voice\.py\|vllm\|EngineCoreProc" > /dev/null 2>&1 || break
    sleep 1
  done

  # Force-kill anything still alive.
  pkill -9 -f "vllm\|EngineCoreProc" 2>/dev/null || true

  # Force-free the selected camera and UI port if anything still holds them.
  fuser -k "/dev/video${CAMERA_ID}" 2>/dev/null || true
  fuser -k "${UI_PORT}/tcp" 2>/dev/null || true
  fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
  sleep 2
}

wait_for_vllm() {
  echo "[infer.sh] Waiting for vLLM endpoint on port ${VLLM_PORT}…"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[infer.sh] vLLM is ready."
      return 0
    fi
    sleep 1
  done
  echo "[infer.sh] vLLM did not become ready in time."
  return 1
}

prepare_torch_env() {
  local py_ver site_packages current_cuda nvidia_cuda
  py_ver="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  site_packages="$CONDA_PREFIX/lib/python${py_ver}/site-packages"

  if [[ ! -d "$site_packages/~orch" ]]; then
    return 0
  fi

  current_cuda="$("$PYTHON_BIN" - <<PY
from pathlib import Path
scope = {}
ver = Path(r"$site_packages/torch/version.py")
if ver.exists():
    exec(ver.read_text(), scope)
print(scope.get("cuda"))
PY
)"
  nvidia_cuda="$("$PYTHON_BIN" - <<PY
from pathlib import Path
scope = {}
ver = Path(r"$site_packages/~orch/version.py")
if ver.exists():
    exec(ver.read_text(), scope)
print(scope.get("cuda"))
PY
)"

  if [[ "$current_cuda" == "None" && "$nvidia_cuda" != "None" ]]; then
    mkdir -p "$PY_SHIM_DIR"
    ln -sfn "$site_packages/~orch" "$PY_SHIM_DIR/torch"
    if [[ -d "$site_packages/~orchgen" ]]; then
      ln -sfn "$site_packages/~orchgen" "$PY_SHIM_DIR/torchgen"
    fi
    export PYTHONPATH="$PY_SHIM_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "[infer.sh] Detected CPU-only torch shadowing NVIDIA torch. Using shim from $site_packages/~orch"
  fi
}

require_gpu() {
  "$PYTHON_BIN" - <<'PY'
import os
import sys
import torch
print(f"[infer.sh] python={sys.executable}")
print(f"[infer.sh] torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"[infer.sh] py_path0={sys.path[0]}")
print(f"[infer.sh] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
if not torch.cuda.is_available():
    if torch.version.cuda:
        print("[infer.sh] CUDA build detected but device initialization failed.")
    raise SystemExit("[infer.sh] GPU is not available in the vln_infer environment. Refusing to start.")
print(f"[infer.sh] GPU device={torch.cuda.get_device_name(0)}")
PY
}

# ── 1. Verify GPU / kill leftovers ──────────────────────────────────────────
prepare_torch_env
require_gpu
cleanup_existing_processes

# ── 2. Reset shared files ────────────────────────────────────────────────────
: > "$LOG"
rm -f "$RESTART_FLAG" "$FRAME_FILE" /tmp/vln_paused.flag

if [[ -z "$INSTRUCTION" ]]; then
  # No instruction → start paused; UI will set one.
  : > "$INSTR_FILE"
  echo "1" > /tmp/vln_paused.flag
  INSTRUCTION="(awaiting instruction from UI)"
  AWAITING=1
  echo "[infer.sh] No instruction provided — starting PAUSED (set one in the UI)"
else
  echo "$INSTRUCTION" > "$INSTR_FILE"
  AWAITING=0
fi

# ── 3. Start web UI in background ────────────────────────────────────────────
"$PYTHON_BIN" -u web_ui_voice.py --port "$UI_PORT" --log "$LOG" >> "$LOG" 2>&1 &
UI_PID=$!
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
HOST_IP="${HOST_IP:-127.0.0.1}"
echo "[infer.sh] Web UI PID=$UI_PID  →  http://$HOST_IP:$UI_PORT"

# ── 4. Trap Ctrl+C to clean up web UI ────────────────────────────────────────
trap 'kill $UI_PID 2>/dev/null || true; exit 0' INT TERM EXIT

# ── 5. Run inference. Single tee writes to $LOG. ─────────────────────────────
if [[ "$AWAITING" -eq 1 ]]; then
  echo "[infer.sh] Starting REAL inference (paused — waiting for UI instruction)"
else
  echo "[infer.sh] Starting REAL inference (instruction: $INSTRUCTION)"
fi
VLLM_ARGS=()
if [[ "$USE_VLLM" -eq 1 ]]; then
  VLLM_ARGS=(--use-vllm --vllm-port "$VLLM_PORT" --vllm-log "$LOG")
fi

"$PYTHON_BIN" -u infer.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --instruction "$INSTRUCTION" \
    --camera-id "$CAMERA_ID" \
    --max-turns "$MAX_TURNS" \
    --max-steps "$MAX_STEPS" \
    --shared-frame-fps "$SHARED_FRAME_FPS" \
    --headless \
    "${VLLM_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$LOG"
