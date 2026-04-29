set -x

# Joint R2R + RxR training.
#
# Both datasets are concatenated via Hydra's list syntax so the trainer draws
# from both corpora every epoch.  R2R-trained and RxR-trained SFT checkpoints
# are merged into a single reference model before RL (see REF_MODEL_PATH).
#
# Key differences vs. single-dataset scripts:
#   - Dual parquet files passed to data.train_files
#   - Longer 499-step RL run to accommodate the larger combined dataset
#   - Stricter α=1.1 early stopping (vs ActiveVLN α=2.0)
#   - Guaranteed success re-sampling (max_sample_attempts=3)

PROJECT_NAME="activevln"
EXPERIMENT_NAME="r2r_rxr_joint"

export SAVE_CHECKPOINT_DIR="verl_checkpoints"

DATASET_R2R=data/r2r_4000_train.parquet
DATASET_RXR=data/rxr_4000_train_new.parquet
DATASET_VAL=data/r2r_val_tiny.parquet

# Use the RxR-fine-tuned model as reference; it was trained on both R2R and
# RxR data via envdrop multiturn SFT, making it a good starting point for
# joint RL.
REF_MODEL_PATH=Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_rxr_multiturn

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/vlnce"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path $CONFIG_PATH --config-name train_vlnce_4gpus.yaml \
    "data.train_files=[${DATASET_R2R},${DATASET_RXR}]" \
    data.val_files=[${DATASET_VAL}] \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.rollout.agent.base_url="http://127.0.0.1:5001" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.agent.timeout=600 \
    actor_rollout_ref.rollout.agent.gen_length_tolerance=2.0 \
    actor_rollout_ref.rollout.agent.experiment_name=${EXPERIMENT_NAME} \
    actor_rollout_ref.rollout.agent.reward.reward_type=weighted_success_ndtw \
    actor_rollout_ref.rollout.agent.reward.success_reward_base=15 \
    actor_rollout_ref.rollout.agent.reward.ndtw_reward_base=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.logger=[console,tensorboard,wandb] \
    trainer.total_training_steps=499 \
    actor_rollout_ref.rollout.agent.smooth_alpha=1.1 \
    actor_rollout_ref.rollout.agent.enable_dynamic_sampling=true \
    actor_rollout_ref.rollout.agent.max_sample_attempts=3
