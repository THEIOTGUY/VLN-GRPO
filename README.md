# VLN-GRPO: Multi-Turn Reinforcement Learning for Indoor Navigation

**AR525 — Reinforcement Learning for Robotics | Group 2**  
Ayush Vaidande · Harsh Vardhan Saxena · Sushant Sharma

---

## Overview

VLN-GRPO applies Group Relative Policy Optimization (GRPO) to Vision-Language Navigation (VLN), training a 3B-parameter multimodal model to follow natural-language navigation instructions in photo-realistic indoor environments. The system outperforms the prior state-of-the-art ActiveVLN baseline with **59.4% success rate** and **4.09 m navigation error** on RxR Val-Unseen — using the same 3B model and the same 177K training trajectories.

The key insight is that imitation learning teaches the agent *what* a human expert would do, but GRPO teaches it *what actually works*, allowing it to discover faster and more efficient navigation strategies through self-generated rollouts.

---

## Results

| Method (Model Size) | NE ↓ | SR ↑ | SPL ↑ | nDTW ↑ |
|---|---|---|---|---|
| NaVILA (8B) | 6.77 | 49.3 | 44.0 | 58.8 |
| UniNaVid (7B) | 6.24 | 48.7 | 40.9 | — |
| StreamVLN (7B) | 6.22 | **52.9** | **46.0** | **61.9** |
| ActiveVLN (3B, w/o RL) | 7.25 | 41.0 | 34.1 | — |
| ActiveVLN (3B) | 5.84 | 50.7 | 41.2 | 58.1 |
| **VLN-GRPO (3B)** | **4.09** | **59.4** | — | — |

> NE: Navigation Error (m) ↓, SR: Success Rate (%) ↑, SPL: Success weighted by Path Length ↑, nDTW: normalized Dynamic Time Warping ↑

**Training progression over 499 steps:**

| Phase | Steps | SR (%) | Oracle SR (%) | Avg Reward | Dist to Goal (m) |
|---|---|---|---|---|---|
| 1 | 1–125 | 41.8 ± 8.9 | 43.4 ± 9.1 | 3.6 ± 0.9 | 6.3 ± 1.4 |
| 2 | 125–250 | 45.6 ± 10.4 | 47.4 ± 10.5 | 4.0 ± 1.2 | 5.5 ± 1.3 |
| 3 | 250–375 | 48.6 ± 10.8 | 50.9 ± 11.1 | 4.5 ± 1.2 | 5.4 ± 1.5 |
| 4 | 375–499 | 53.1 ± 11.1 | 54.8 ± 11.0 | 5.1 ± 1.3 | 4.9 ± 1.4 |

Final: **+19.1 pp SR gain**, **−2.40 m distance reduction**, **+2.04 reward improvement** from initialization.

---

## Architecture

Training proceeds in two stages:

### Stage 1 — Imitation Learning (Bootstrap)

The model is initialized on 167,600 expert trajectories from R2R/RxR using cross-entropy loss, learning action syntax from human demonstrations. This produces an initial policy capable of navigating but constrained to memorized trajectories.

### Stage 2 — GRPO Reinforcement Learning

The IL-initialized policy is fine-tuned via GRPO using 4,000 expert reference trajectories per dataset. At each step, the model generates `n=4` rollouts per prompt. Rollouts better than the group average receive positive advantage; worse ones receive negative advantage. The policy update is clipped at `[0.2, 0.28]`.

**Reward function** ([`vlnce_server/env.py`](vlnce_server/env.py)):
```
R = success_reward × soft_success
  + ndtw_reward   × nDTW
  + format_reward × format_correct
  + landmark_reward × landmarks_found
```
Where `soft_success = 1.0 if dist_to_goal < 3.0m else 0.0`.

---

## Key Contributions — 8 Improvements Over ActiveVLN

### Training Strategy

**1. Stricter Early Stopping** ([`examples/vlnce/train_vlnce_4gpus.yaml`](examples/vlnce/train_vlnce_4gpus.yaml))  
`smooth_alpha: 1.1` vs ActiveVLN's `α = 2.0`. Prunes failed rollouts more aggressively before they waste training signal, keeping the advantage estimates tighter.

**2. Guaranteed Re-sampling** ([`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py))  
Dynamic re-sampling with `max_sample_attempts: 3` ensures at least one successful rollout per group, so the reward signal is never all-negative and the policy always has a positive example to learn from.

**3. Longer RL Training** ([`examples/vlnce/train_vlnce_4gpus.yaml`](examples/vlnce/train_vlnce_4gpus.yaml))  
499 steps vs ActiveVLN's ~350 convergence point. The success rate curve continues to improve monotonically through step 499 with no sign of degradation.

**4. Dual Dataset Training** ([`examples/vlnce/r2r_rxr_joint.sh`](examples/vlnce/r2r_rxr_joint.sh))  
Jointly trains on R2R (`r2r_4000_train.parquet`) and RxR (`rxr_4000_train_new.parquet`) reference trajectories, exposing the policy to a broader instruction distribution and improving generalization.

### System Design

**5. Closed-Loop Obstacle Replanning** ([`vlnce_server/env.py`](vlnce_server/env.py), [`deploy/go1_nav.py`](deploy/go1_nav.py))  
Reads the RealSense depth stream after each action. If the central 1/3 of the frame contains any depth reading below 0.8 m, the VLM is called again mid-trajectory with a `[Replanning — obstacle detected]` prefix — enabling real-time adaptation to newly observed obstacles rather than waiting for the next episode.

**6. Spatial Memory Module** ([`vlnce_server/memory.py`](vlnce_server/memory.py), [`deploy/go1_nav.py`](deploy/go1_nav.py))  
The agent builds a running top-down occupancy grid across all steps, injecting a text summary of visited positions and discovered landmarks into every VLM prompt. This enables long-horizon planning and reduces repetitive back-tracking. In simulation, position is read from Habitat's state; in deployment, it is dead-reckoned from forward/turn primitives.

**7. Curriculum Difficulty Scheduling** ([`vlnce_server/curriculum.py`](vlnce_server/curriculum.py), [`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py))  
Episodes are filtered by GT-action count. Early training exposes the agent only to short corridors (min 10 actions); the allowed maximum ramps linearly up to the full distribution by 70% of training. This prevents the policy from encountering hard long-horizon episodes before it has mastered basic navigation, leading to faster convergence and better generalization.

**8. Semantic Landmark Rewards** ([`vlnce_server/env.py`](vlnce_server/env.py), [`vlnce_server/memory.py`](vlnce_server/memory.py))  
Named objects are extracted from the instruction (e.g., "brown chair", "red door") via regex. GT waypoints are distributed across the instruction's landmark phrases. The agent receives a bonus reward each time it enters proximity of a waypoint, directly encouraging visual grounding of semantic targets from the instruction.

---

## Repository Layout

```
VLN-GRPO/
├── vlnce_server/          # Simulation environment and training components
│   ├── env.py             # VLNCEEnv: Habitat wrapper, reward, obstacle detection
│   ├── env_config.py      # VLNCEConfig dataclass with all feature flags
│   ├── memory.py          # SpatialMemory: 2D occupancy grid + landmark tracking
│   ├── curriculum.py      # CurriculumScheduler: linear action-count ramp
│   ├── prompt.py          # VLM prompt templates (with memory/replan variants)
│   ├── constants.py       # Shared string constants and default thresholds
│   └── server.py          # Ray-based environment server
│
├── examples/vlnce/        # Training launch scripts
│   ├── train_vlnce_4gpus.yaml  # Main Hydra config (all hyperparameters)
│   ├── r2r.sh             # R2R-only training
│   ├── rxr.sh             # RxR-only training
│   └── r2r_rxr_joint.sh   # Dual-dataset training (R2R + RxR)
│
├── verl/trainer/ppo/
│   └── ray_trainer.py     # GRPO trainer (curriculum filtering, dynamic sampling)
│
├── deploy/                # Hardware deployment
│   ├── run.sh             # Main launcher — starts vLLM + web UI + navigation
│   ├── go1_nav.py         # Navigation loop (Unitree Go1 / Diadem AGV)
│   ├── web_ui.py          # Flask web UI with live video, controls, memory stats
│   ├── odom.py            # Dead-reckoning manual control (cm + degrees)
│   └── cmd_vel_input.py   # Raw cmd_vel publisher (m/s + rad/s)
│
├── data/
│   ├── r2r_4000_train.parquet
│   └── rxr_4000_train_new.parquet
│
├── checkpoints/
│   └── Qwen2.5-VL-3B_rl_r2r_4000/    # IL-trained reference model
│
├── global_step_450/       # Saved GRPO checkpoint (step 450)
└── eval/vlnce/            # Evaluation scripts
    ├── eval_vlnce.py
    └── analyze_results.py
```

---

## Training

### Prerequisites

```bash
conda env create -f conda_rl_env.yml
conda activate RL
bash scripts/install_activevln.sh
```

GPU requirements: 4× NVIDIA GPU with ≥40 GB VRAM (tested on A100 80GB). Training takes ~20 hours.

### Single Dataset (R2R)

```bash
bash examples/vlnce/r2r.sh
```

### Dual Dataset (R2R + RxR)

```bash
bash examples/vlnce/r2r_rxr_joint.sh
```

### Key hyperparameters ([`examples/vlnce/train_vlnce_4gpus.yaml`](examples/vlnce/train_vlnce_4gpus.yaml))

| Parameter | Value | Note |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | |
| Learning rate | 1e-6 | |
| Batch size | 8 prompts × 4 rollouts | |
| Clip ratio | [0.2, 0.28] | |
| Temperature | 1.2 (train) / 0.2 (val) | |
| Total steps | 499 | |
| smooth_alpha | 1.1 | stricter early stopping vs ActiveVLN's 2.0 |
| KL coefficient | 0.0 | no KL penalty |

---

## Evaluation

```bash
bash examples/vlnce/eval_r2r.sh
bash examples/vlnce/eval_rxr.sh
python eval/vlnce/analyze_results.py --results-dir <path>
```

---

## Hardware Deployment

### Quick Start

```bash
# Full system with all features
bash deploy/run.sh --all-features

# Diadem AGV instead of Go1
bash deploy/run.sh --all-features --diadem

# Custom instruction at launch
INSTRUCTION="Go to the brown chair and stop" bash deploy/run.sh --all-features
```

The launcher ([`deploy/run.sh`](deploy/run.sh)):
1. Kills stale vLLM / UI processes
2. Starts Flask web UI on port 5000
3. Launches vLLM server on port 8003 with the GRPO checkpoint
4. Runs the navigation loop ([`deploy/go1_nav.py`](deploy/go1_nav.py))

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_PATH` | `global_step_450/actor/huggingface` | Path to HF model |
| `INSTRUCTION` | *(wait for UI)* | Override instruction at launch |
| `ENABLE_OBSTACLE_REPLANNING` | `1` | Depth obstacle → VLM replan |
| `OBSTACLE_DEPTH_THRESHOLD` | `0.8` | Metres from camera |
| `ENABLE_MEMORY` | `1` | Inject dead-reckoning memory into prompts |
| `ENABLE_LANDMARK_TRACKING` | `1` | Track named objects from instruction |
| `ACTION_SPACE` | `r2r` | `r2r` (4 actions) or `rxr` (continuous) |
| `VLLM_PORT` | `8003` | vLLM server port |
| `UI_PORT` | `5000` | Web UI port |

### Web UI ([`deploy/web_ui.py`](deploy/web_ui.py))

Access at `http://<robot-ip>:5000`. Features:
- Live camera feed with JPEG streaming
- Instruction input with voice (Web Speech API)
- Step counter, last action, LLM latency, spatial memory state, obstacle alert
- Pause / Emergency Stop / Restart controls
- Real-time log stream via Server-Sent Events

### Manual robot control

```bash
# Velocity control (m/s, rad/s)
python deploy/cmd_vel_input.py

# Distance/angle odometry (cm, degrees)
python deploy/odom.py
```

---

## Hardware Notes

The project was deployed on two platforms:

**Unitree Go1** (quadruped): payload limitations caused odometry drift during long-horizon navigation. Camera and compute were mounted on the body; the RealSense D435i provided RGB + depth streams.

**Diadem AGV** (wheeled): better payload handling; migrated to this platform after Go1 instability. ROS2/micro-ROS bridge via `/cmd_vel` topic. AGV SDK compatibility required custom ROS2 wrappers.

Both platforms use the same [`deploy/go1_nav.py`](deploy/go1_nav.py) navigator; select with `--diadem` flag for the Diadem-specific low-level interface.

---

## Why PPO Failed and GRPO Works

Early experiments used trajectory-based dense reward with PPO. The model found a shortcut — oscillating left-right to avoid step penalties without navigating — a classic reward hacking failure. PPO had no positive signal to learn from once the heuristic was discovered.

GRPO solves this by normalizing rewards *within a group of rollouts*. Even if all rewards are low, the relative ranking provides a meaningful gradient. Guaranteed re-sampling further ensures at least one rollout per group succeeds, so the policy always has a positive example to reinforce.

---

## References

1. ActiveVLN: Chen et al. "ActiveVLN: Active Vision-Language Navigation with Self-Generated Rollouts." arXiv 2024.
2. Qwen2.5-VL: Wang et al. "Qwen2.5-VL Technical Report." Alibaba Group, 2025.
3. RxR Dataset: Ku et al. "Room-Across-Room: Multilingual VLN with Dense Spatiotemporal Grounding." EMNLP 2020.
4. R2R Dataset: Anderson et al. "Vision-and-Language Navigation." CVPR 2018.
5. Habitat: Savva et al. "Habitat: A Platform for Embodied AI Research." ICCV 2019.
6. GRPO: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning." arXiv:2402.03300, 2024.
7. DAgger: Ross et al. "A Reduction of Imitation Learning to No-Regret Online Learning." AISTATS 2011.
