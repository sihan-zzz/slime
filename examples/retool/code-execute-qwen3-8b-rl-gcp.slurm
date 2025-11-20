#!/bin/bash
#
#SBATCH --job-name=qwen3-8b
#SBATCH --nodes=1                      # set how many nodes you want
#SBATCH --ntasks-per-node=1            # one launcher per node (procid 0..N-1)
#SBATCH --gpus-per-node=8              # GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --account=cpt
#
# NOTE:
# - Adjust nodes/gpus/cpus/time/partition to your GCP/Slurm cluster.
# - This script launches a Ray cluster using the Slurm allocation and
#   submits the original ray job from the head (procid 0).
# - Alternative: below we also show how to use Slurm-native rendezvous
#   with torchrun if you want to avoid Ray's cluster startup.

set -ex

# --- user tweakables ---
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}  # fallback if your cluster exposes env var differently
PYTHON=python
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
WORKSPACE_DIR="$(realpath "${SLIME_REPO_DIR}/..")"
DEFAULT_MEGATRON_REPO="${WORKSPACE_DIR}/Megatron-LM"
MEGATRON_REPO_DIR="${MEGATRON_REPO_DIR:-${DEFAULT_MEGATRON_REPO}}"

if [ ! -d "${MEGATRON_REPO_DIR}" ]; then
  echo "ERROR: Could not locate Megatron-LM checkout at ${MEGATRON_REPO_DIR}. Set MEGATRON_REPO_DIR to override." >&2
  exit 1
fi

PYTHONPATH_BASE="${MEGATRON_REPO_DIR}:${SLIME_REPO_DIR}:${SCRIPT_DIR}"
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH_VALUE="${PYTHONPATH_BASE}:${PYTHONPATH}"
else
  PYTHONPATH_VALUE="${PYTHONPATH_BASE}"
fi

export CUDA_HOME=/usr/local/cuda-12.8
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

export LOG_DIR=/home/sihanzeng_meta_com/uv/logs
export UNIQUE_JOB_NAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
export LOG_PREFIX=${LOG_DIR}/${UNIQUE_JOB_NAME}_$(date +"%Y%m%d_%H%M%S")
echo "Creating log directory at $LOG_PREFIX"
mkdir -p "$LOG_PREFIX"
exec > "$LOG_PREFIX/sbatch.out" 2>&1


which python

# torch.compile in PyTorch 2.4 + Triton 3.4 in this env hits a duplicate template bug.
# Force Dynamo off so we fall back to eager mode unless explicitly overridden.
: "${TORCHDYNAMO_DISABLE:=1}"
export TORCHDYNAMO_DISABLE

# load modules / conda if necessary (example)
# module load cuda/12.1
# source /path/to/venv/bin/activate

# helper: head hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 20000))}   # stable-ish per-job port

echo "SLURM_JOBID: $SLURM_JOB_ID"
echo "NODES: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# clean up any stale processes (only on nodes in this allocation)
pkill -9 sglang || true
pkill -9 ray || true
pkill -9 python || true
sleep 2

# detect NVLink if you need to set NCCL flags
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# set env used by the application
export PYTHONBUFFERED=16
export MASTER_ADDR
export MASTER_PORT

# common runtime env used for ray job submit (tweak PYTHONPATH)
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${PYTHONPATH_VALUE}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_SHARP_DISABLE": "1",
    "NCCL_DEBUG": "INFO"
  }
}
EOF
)

#
# Start Ray cluster inside the Slurm allocation
# We start one ray head on SLURM_PROCID==0 and join from others.
#
if [ "${SLURM_PROCID}" -eq 0 ]; then
  echo "Starting ray head on ${MASTER_ADDR}"
  ray stop --force || true
  # start head (dashboard on 8265 so we can submit the job)
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  # small wait for the head to be ready
  sleep 5
else
  # other nodes join the head
  echo "Joining ray head at ${MASTER_ADDR}"
  ray stop --force || true
  ray start --address="${MASTER_ADDR}:6379" --num-gpus ${GPUS_PER_NODE} --disable-usage-stats
fi

# wait for all ray nodes to be ready
# srun barrier: use srun to synchronize across nodes (each allocation has one SLURM_PROCID task)
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --mpi=none true

# Only on head node submit the ray job (this mirrors the original script)
if [ "${SLURM_PROCID}" -eq 0 ]; then

  # these arrays are copied/adapted from your original script; change paths as needed
  CKPT_ARGS=(
     --hf-checkpoint /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/qwen3_8b
     --ref-load /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/qwen3_8b_torch_dist
     --save /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/qwen3-8b-rl-multi-turn/
     --save-interval 20
     --rotary-base 1000000
  )

  ROLLOUT_ARGS=(
     --prompt-data /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/APPS-dataset/APPS-synthetic-train.jsonl
     --input-key prompt
     --label-key label
     --apply-chat-template
     --rollout-shuffle
     --reward-key score
     --num-rollout 3000
     --rollout-batch-size 32
     --n-samples-per-prompt 8
     --rollout-max-response-len 8192
     --rollout-temperature 0.8
     --over-sampling-batch-size 48
     --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.filter_truncated_samples
     --global-batch-size 256
     --balance-data
  )

  EVAL_ARGS=(
     --eval-interval 20
     --eval-prompt-data apps  /mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/APPS-dataset/APPS-synthetic-test.jsonl
     --n-samples-per-eval-prompt 4
     --eval-max-response-len 4096
     --eval-top-p 0.7
  )

  PERF_ARGS=(
     --tensor-model-parallel-size 4
     --sequence-parallel
     --pipeline-model-parallel-size 2
     --context-parallel-size 1
     --expert-model-parallel-size 1
     --expert-tensor-parallel-size 1
     --recompute-granularity full
     --recompute-method uniform
     --recompute-num-layers 1
     --use-dynamic-batch-size
     --max-tokens-per-gpu 9216
  )

  GRPO_ARGS=(
     --advantage-estimator grpo
     --use-kl-loss
     --kl-loss-coef 0.00
     --kl-loss-type low_var_kl
     --entropy-coef 0.00
     --eps-clip 0.2
     --eps-clip-high 0.28
  )

  OPTIMIZER_ARGS=(
     --optimizer adam
     --lr 1e-6
     --lr-decay-style constant
     --weight-decay 0.1
     --adam-beta1 0.9
     --adam-beta2 0.98
  )

  WANDB_ARGS=(
     --use-wandb
     --wandb-project slime-dapo
     --wandb-group qwen3-8B-test-multi-turn
     --wandb-key ${WANDB_KEY}
  )

  SGLANG_ARGS=(
     --rollout-num-gpus-per-engine 2
     --sglang-mem-fraction-static 0.9
  )

  MISC_ARGS=(
     --attention-dropout 0.0
     --hidden-dropout 0.0
     --accumulate-allreduce-grads-in-fp32
     --attention-softmax-in-fp32
     --attention-backend flash
  )

  CUSTOM_ARGS=(
     --custom-generate-function-path generate_with_code_execute.generate
     --custom-rm-path generate_with_code_execute.reward_func
     --rollout-health-check-timeout 45
  )

  # submit the same ray job as you had in containerized setup
  echo "Launching training directly on the head node (this will BLOCK the sbatch until finished)"
  export PYTHONPATH="${PYTHONPATH_VALUE}"
  ${PYTHON} train.py \
    --actor-num-nodes ${SLURM_NNODES} \
    --actor-num-gpus-per-node ${GPUS_PER_NODE} \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
  2>&1 | tee train.log

  TRAIN_EXIT_CODE=${PIPESTATUS[0]:-0}
  echo "train.py exited with code ${TRAIN_EXIT_CODE}"
  # ray stop only after training completes
  ray stop --force || true
  exit ${TRAIN_EXIT_CODE}
fi

# ---------------------------
# Alternative: Slurm-native rendezvous (torchrun)
# ---------------------------
# If you prefer to bypass Ray and run PyTorch's distributed backend directly using Slurm's rendezvous:
# - Use --ntasks-per-node=$GPUS_PER_NODE and launch one launcher per GPU OR
# - Use torchrun with --nproc_per_node=${GPUS_PER_NODE} and srun to allocate nodes.
# Example (uncomment & adapt if you want):
#
# if you want to use torch.distributed rendezvous:
#   export NCCL_DEBUG=INFO
#   export PYTHONPATH="${PYTHONPATH_VALUE}"
#   # Run torchrun with c10d rendezvous on MASTER_ADDR:MASTER_PORT
#   srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --mpi=none \
#     ${PYTHON} -m torch.distributed.run \
#       --nnodes=${SLURM_NNODES} \
#       --nproc_per_node=${GPUS_PER_NODE} \
#       --rdzv_backend=c10d \
#       --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#       --rdzv_id=${SLURM_JOB_ID} \
#       train.py \
#       --actor-num-nodes ${SLURM_NNODES} \
#       --actor-num-gpus-per-node ${GPUS_PER_NODE} \
#       ...
#
# Adjust train.py flags so that it uses torch.distributed (no ray).
