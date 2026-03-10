#!/bin/bash
# Launch all 12 baseline eval jobs: 3 models x 2 datasets x 2 modes (tools/notool)
# Each job uses 2 nodes in colocate mode (all GPUs for rollout).
#
# Usage:
#   WANDB_KEY=<your_key> bash examples/retool/launch_eval_sweep.sh
#
# Requires: WANDB_KEY environment variable

set -euo pipefail

if [ -z "${WANDB_KEY:-}" ]; then
    echo "ERROR: WANDB_KEY must be set"
    exit 1
fi

MOUNT_DIR="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime"
CODE_DATA="${MOUNT_DIR}/data/new_code_processed/test/2_python_filter_by_qwen3_4b_5791_4b_tool_eval_scan_train.jsonl"
MATH_DATA="${MOUNT_DIR}/data/math/5_mathematical_processed_data_v2_prompt_test.jsonl"

NODES=2

# Common eval-only args: colocate, no training rollouts, eval immediately
EVAL_COMMON="--colocate --eval-interval 1 --n-samples-per-eval-prompt 8 --lr-decay-iters 1 --lr-warmup-iters 0 --eval-max-response-len 16384 --eval-temperature 0.6 --eval-top-p 0.95 --eval-top-k 20"

echo "=== Launching 12 Eval Jobs ==="

# --- Qwen3-4B ---
SLURM_4B="examples/retool/code-execute-qwen3-4b-rl-gcp.slurm"

# 4B + Code + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_4b_code_tools $SLURM_4B

# 4B + Code + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_4b_code_notool $SLURM_4B

# 4B + Math + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_4b_math_tools $SLURM_4B

# 4B + Math + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_4b_math_notool $SLURM_4B

# --- Qwen3-8B ---
SLURM_8B="examples/retool/code-execute-qwen3-8b-rl-gcp.slurm"

# 8B + Code + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_8b_code_tools $SLURM_8B

# 8B + Code + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_8b_code_notool $SLURM_8B

# 8B + Math + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_8b_math_tools $SLURM_8B

# 8B + Math + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_8b_math_notool $SLURM_8B

# --- Qwen3-30B-A3B ---
SLURM_30B="examples/retool/code-execute-qwen3-30ba3b-rl-gcp.slurm"

# 30B-A3B + Code + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_30b_code_tools $SLURM_30B

# 30B-A3B + Code + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$CODE_DATA \
    EVAL_DATA="code $CODE_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_30b_code_notool $SLURM_30B

# 30B-A3B + Math + Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON" \
    sbatch --nodes=$NODES --job-name=eval_30b_math_tools $SLURM_30B

# 30B-A3B + Math + No Tools
WANDB_KEY=$WANDB_KEY TRAIN_NODES=$NODES NUM_ROLLOUT=0 \
    TRAIN_DATA=$MATH_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="math $MATH_DATA" \
    ADD_ARGS="$EVAL_COMMON --disable-tool-use" \
    sbatch --nodes=$NODES --job-name=eval_30b_math_notool $SLURM_30B

echo "=== All 12 eval jobs submitted ==="
