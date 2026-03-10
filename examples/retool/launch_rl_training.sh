#!/bin/bash
# Launch all 6 GRPO RL training jobs: 3 models x 2 modes (tools/notool)
# Mixed code+math training data. Eval every 20 steps, save every 20 steps.
#
# Node allocation:
#   4B:       4 nodes (2 train + 2 rollout)
#   8B:       8 nodes (2 train + 6 rollout)
#   30B-A3B:  8 nodes (2 train + 6 rollout)
#
# Usage:
#   WANDB_KEY=<your_key> bash examples/retool/launch_rl_training.sh
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
CODE_EVAL="${MOUNT_DIR}/data/new_code_processed/test/2_python_filter_by_qwen3_4b_5791_4b_tool_eval_scan_train.jsonl"
MATH_EVAL="${MOUNT_DIR}/data/math/5_mathematical_processed_data_v2_prompt_test.jsonl"

echo "=== Launching 6 RL Training Jobs ==="

# --- T1: Qwen3-4B + Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="" \
    sbatch --nodes=4 --job-name=rl_4b_tools \
    examples/retool/code-execute-qwen3-4b-rl-gcp.slurm

# --- T2: Qwen3-4B + No Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="--disable-tool-use" \
    sbatch --nodes=4 --job-name=rl_4b_notool \
    examples/retool/code-execute-qwen3-4b-rl-gcp.slurm

# --- T3: Qwen3-8B + Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="" \
    sbatch --nodes=8 --job-name=rl_8b_tools \
    examples/retool/code-execute-qwen3-8b-rl-gcp.slurm

# --- T4: Qwen3-8B + No Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="--disable-tool-use" \
    sbatch --nodes=8 --job-name=rl_8b_notool \
    examples/retool/code-execute-qwen3-8b-rl-gcp.slurm

# --- T5: Qwen3-30B-A3B + Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="" \
    sbatch --nodes=8 --job-name=rl_30b_tools \
    examples/retool/code-execute-qwen3-30ba3b-rl-gcp.slurm

# --- T6: Qwen3-30B-A3B + No Tools ---
WANDB_KEY=$WANDB_KEY TRAIN_NODES=2 NUM_ROLLOUT=500 \
    TRAIN_DATA=$CODE_DATA MATH_DATA=$MATH_DATA \
    EVAL_DATA="code $CODE_EVAL math $MATH_EVAL" \
    ADD_ARGS="--disable-tool-use" \
    sbatch --nodes=8 --job-name=rl_30b_notool \
    examples/retool/code-execute-qwen3-30ba3b-rl-gcp.slurm

echo "=== All 6 RL training jobs submitted ==="
