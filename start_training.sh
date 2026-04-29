#!/bin/bash
REPO="/usershome/cs671_user6/.nv/ComputeCache/RR"
CONDA_SITE="$REPO/miniconda3/envs/RL/lib/python3.10/site-packages"
export PATH="$REPO/miniconda3/envs/RL/bin:$PATH"
export PYTHONPATH="$CONDA_SITE:$REPO"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Set WANDB_API_KEY in your environment before running this script.
# export WANDB_API_KEY=<your_key>
NVIDIA_LIBS="$CONDA_SITE/nvidia"
TORCH_LIBS="$CONDA_SITE/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIBS:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$REPO/miniconda3/envs/RL/lib:${LD_LIBRARY_PATH}"
cd "$REPO"

# Reset any orphaned simulator environments from a previous crashed run
echo "Resetting simulator pool..."
curl -s -X POST http://127.0.0.1:5001/admin/reset_all | grep -o '"message":"[^"]*"' || true
echo "Simulator pool reset complete."

nohup bash examples/vlnce/r2r.sh >> training.log 2>&1 &
echo "Training started PID: $!"
