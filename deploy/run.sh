#!/usr/bin/env bash
# ActiveVLN launcher (from repo root: bash deploy/run.sh)
#
# Usage:
#   bash deploy/run.sh
#   bash deploy/run.sh --dry-run
#   INSTRUCTION="Go to the brown chair" bash deploy/run.sh
#   bash deploy/run.sh --test
#   bash deploy/run.sh --diadem          # Diadem AGV via USB instead of Go1
#
# New feature flags (set env vars or pass as CLI args):
#   ENABLE_OBSTACLE_REPLANNING=1         # detect depth obstacles → replan
#   OBSTACLE_DEPTH_THRESHOLD=0.8        # metres (default 0.8)
#   ENABLE_MEMORY=1                     # inject spatial memory into VLM prompts
#   ENABLE_LANDMARK_TRACKING=1          # track named landmarks from instruction
#
# Examples:
#   ENABLE_MEMORY=1 ENABLE_LANDMARK_TRACKING=1 bash deploy/run.sh
#   ENABLE_OBSTACLE_REPLANNING=1 OBSTACLE_DEPTH_THRESHOLD=0.6 bash deploy/run.sh

set -euo pipefail

source /home/chitti/miniforge3/etc/profile.d/conda.sh
conda activate RL
cd "$(dirname "$0")/.."   # repo root

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Source micro-ROS workspace (disable strict mode around ROS2 setup scripts
# which use unbound variables internally and trip set -euo pipefail)
set +euo pipefail
source /home/chitti/uros_ws/install/setup.bash 2>/dev/null || true
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
INSTRUCTION="${INSTRUCTION:-}"
VLLM_PORT="${VLLM_PORT:-8003}"
UI_PORT="${UI_PORT:-5000}"
CAMERA_ID="${CAMERA_ID:-0}"
SETTLE_TIME="${SETTLE_TIME:-0.4}"
FRAME_TIMEOUT="${FRAME_TIMEOUT:-4.0}"
ACTION_SPACE="${ACTION_SPACE:-r2r}"      # r2r or rxr

# Checkpoint: defaults to the step-499 joint-trained model when available,
# falling back to the original step-450 R2R checkpoint.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/chitti/ActiveVLN/global_step_450/actor/huggingface}"

NAV_LOG="/tmp/activevln.log"
VLLM_LOG="deploy/vllm_server.log"

# Training / simulation alignment parameters
VLLM_MAX_MODEL_LEN=32768

# ── New feature flags ─────────────────────────────────────────────────────────
# Closed-loop obstacle replanning (requires RealSense depth stream).
ENABLE_OBSTACLE_REPLANNING="${ENABLE_OBSTACLE_REPLANNING:-0}"
OBSTACLE_DEPTH_THRESHOLD="${OBSTACLE_DEPTH_THRESHOLD:-0.8}"

# Spatial memory module: dead-reckoning map injected into every VLM prompt.
ENABLE_MEMORY="${ENABLE_MEMORY:-0}"

# Semantic landmark tracking from instruction text.
ENABLE_LANDMARK_TRACKING="${ENABLE_LANDMARK_TRACKING:-0}"

# ── Kill stale processes ──────────────────────────────────────────────────────
echo "[run.sh] Cleaning up stale processes…"
pkill -f "go1_nav\.py"    2>/dev/null || true
pkill -f "web_ui\.py"     2>/dev/null || true
pkill -f "vllm"           2>/dev/null || true
pkill -f "EngineCoreProc" 2>/dev/null || true
sleep 1
pkill -9 -f "vllm\|EngineCoreProc" 2>/dev/null || true
fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
fuser -k "${UI_PORT}/tcp"   2>/dev/null || true
sleep 1

# ── Verify GPU ────────────────────────────────────────────────────────────────
python3 -c "
import torch, sys
print(f'[run.sh] torch={torch.__version__}  cuda={torch.version.cuda}')
if not torch.cuda.is_available():
    sys.exit('[run.sh] ERROR: No CUDA GPU available.')
print(f'[run.sh] GPU: {torch.cuda.get_device_name(0)}')
"

: > "$NAV_LOG"
: > "$VLLM_LOG"
rm -f /tmp/vln_paused.flag /tmp/vln_restart.flag /tmp/vln_estop.flag \
      /tmp/vln_instruction.txt /tmp/vln_latest_frame.jpg /tmp/vln_history.json \
      /tmp/vln_memory.json /tmp/vln_obstacle.flag
rm -rf /tmp/vln_obs

# ── Start Web UI in background ────────────────────────────────────────────────
python3 -u deploy/web_ui.py --port "$UI_PORT" --log "$NAV_LOG" &
UI_PID=$!
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "[run.sh] Web UI PID=$UI_PID  →  http://${HOST_IP:-127.0.0.1}:$UI_PORT"
trap 'kill $UI_PID 2>/dev/null || true' EXIT INT TERM

echo "[run.sh] Instruction         : ${INSTRUCTION:-'(waiting for UI)'}"
echo "[run.sh] Obstacle replanning : $ENABLE_OBSTACLE_REPLANNING (threshold ${OBSTACLE_DEPTH_THRESHOLD}m)"
echo "[run.sh] Spatial memory      : $ENABLE_MEMORY"
echo "[run.sh] Landmark tracking   : $ENABLE_LANDMARK_TRACKING"
echo "[run.sh] Starting navigation…"

# ── Build navigation argument list ────────────────────────────────────────────
declare -a NAV_ARGS
NAV_ARGS=(
    --use-vllm
    --checkpoint-path "$CHECKPOINT_PATH"
    --vllm-port       "$VLLM_PORT"
    --vllm-log        "$VLLM_LOG"
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
    --camera-id       "$CAMERA_ID"
    --settle-time     "$SETTLE_TIME"
    --frame-timeout   "$FRAME_TIMEOUT"
    --action-space    "$ACTION_SPACE"
    --rs
    --obstacle-depth-threshold "$OBSTACLE_DEPTH_THRESHOLD"
)

[[ -n "$INSTRUCTION" ]] && NAV_ARGS+=(--instruction "$INSTRUCTION")

# Enable optional features when the corresponding env var is set to 1
[[ "$ENABLE_OBSTACLE_REPLANNING" == "1" ]] && NAV_ARGS+=(--enable-obstacle-replanning)
[[ "$ENABLE_MEMORY"              == "1" ]] && NAV_ARGS+=(--enable-memory)
[[ "$ENABLE_LANDMARK_TRACKING"   == "1" ]] && NAV_ARGS+=(--enable-landmark-tracking)

# ── Pass through CLI overrides ────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --test)                        NAV_ARGS+=(--test-mode) ;;
        --diadem)                      NAV_ARGS+=(--diadem) ;;
        --obstacle-replanning)         NAV_ARGS+=(--enable-obstacle-replanning) ;;
        --memory)                      NAV_ARGS+=(--enable-memory) ;;
        --landmark-tracking)           NAV_ARGS+=(--enable-landmark-tracking) ;;
        --all-features)
            NAV_ARGS+=(--enable-obstacle-replanning --enable-memory --enable-landmark-tracking)
            ;;
        *) NAV_ARGS+=("$arg") ;;
    esac
done

# ── Run navigation (stdout → log + terminal) ──────────────────────────────────
python3 -u deploy/go1_nav.py \
    "${NAV_ARGS[@]}" 2>&1 | tee -a "$NAV_LOG"
