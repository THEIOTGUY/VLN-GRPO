#!/usr/bin/env bash
# Move the Diadem AGV over ROS2 /cmd_vel.
# Run diadem/agent.sh in another terminal first.
#
# Usage:
#   bash diadem/diadem.sh --dist <cm>           # forward (negative = backward)
#   bash diadem/diadem.sh --deg <degrees>        # turn (positive = left, negative = right)
#   bash diadem/diadem.sh --dist 50 --deg 90     # forward then turn
#   bash diadem/diadem.sh                        # stop (sends zero velocity)
#
# Examples:
#   bash diadem/diadem.sh --dist 50              # forward 50 cm
#   bash diadem/diadem.sh --dist -30             # backward 30 cm
#   bash diadem/diadem.sh --deg 90               # turn left 90°
#   bash diadem/diadem.sh --deg -45              # turn right 45°
#   bash diadem/diadem.sh --dist 50 --deg -90    # forward 50 cm then turn right 90°

LINEAR_SPD=0.3
ANGULAR_SPD=0.6
RATE=10

DIST_CM=0
DEG=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --dist) DIST_CM="$2"; shift 2 ;;
    --deg)  DEG="$2";     shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

unset PYTHONHOME
# ros2 uses #!/usr/bin/python3 (system Python3); system numpy is broken on this
# Jetson. Put conda's working numpy first so system Python3 finds it instead.
CONDA_SP="${CONDA_PREFIX:-/home/chitti/miniforge3/envs/RL}/lib/python3.10/site-packages"
export PYTHONPATH="${CONDA_SP}${PYTHONPATH:+:$PYTHONPATH}"
source /home/chitti/uros_ws/install/setup.bash
export ROS_DOMAIN_ID=101

_pub() {
  ros2 topic pub --times "$3" --rate "$RATE" /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: $1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: $2}}"
}

_stop() {
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
}

# ── Forward / backward ────────────────────────────────────────────────────────
MOVED=0
DIST_ABS=$(python3 -c "print(abs(float('$DIST_CM')))")
if python3 -c "import sys; sys.exit(0 if float('$DIST_ABS') > 0.001 else 1)"; then
  LIN=$(python3 -c "d=float('$DIST_CM'); print($LINEAR_SPD if d>0 else -$LINEAR_SPD)")
  TIMES=$(python3 -c "print(max(1, round(float('$DIST_ABS')/100.0 / $LINEAR_SPD * $RATE)))")
  DIR=$(python3 -c "print('forward' if float('$DIST_CM')>0 else 'backward')")
  echo "[diadem] $DIR ${DIST_CM} cm  (${TIMES} msgs @ ${RATE} Hz)"
  _pub "$LIN" "0.0" "$TIMES"
  _stop
  sleep 0.3
  MOVED=1
fi

# ── Turn ──────────────────────────────────────────────────────────────────────
DEG_ABS=$(python3 -c "print(abs(float('$DEG')))")
if python3 -c "import sys; sys.exit(0 if float('$DEG_ABS') > 0.01 else 1)"; then
  ANG=$(python3 -c "d=float('$DEG'); print($ANGULAR_SPD if d>0 else -$ANGULAR_SPD)")
  TIMES=$(python3 -c "import math; print(max(1, round(math.radians(float('$DEG_ABS')) / $ANGULAR_SPD * $RATE)))")
  DIR=$(python3 -c "print('left' if float('$DEG')>0 else 'right')")
  echo "[diadem] turn $DIR ${DEG}°  (${TIMES} msgs @ ${RATE} Hz)"
  _pub "0.0" "$ANG" "$TIMES"
  _stop
  MOVED=1
fi

# ── Stop only ─────────────────────────────────────────────────────────────────
if [[ $MOVED -eq 0 ]]; then
  echo "[diadem] Sending stop..."
  _stop
fi

echo "[diadem] Done."
