#!/bin/bash
REPO="/usershome/cs671_user6/.nv/ComputeCache/RR"
export PATH="$REPO/miniconda3/envs/server/bin:$PATH"
export PYTHONPATH="$REPO/vlnce_server:$REPO/nav_ws/habitat-lab:$REPO"
cd "$REPO"

nohup python3 -m vlnce_server.server \
    vlnce.gpus=[3] \
    vlnce.r2r_gpu_plan=[32] \
    vlnce.rxr_gpu_plan=[0] \
    >> server.log 2>&1 &
echo "Server started PID: $!"
