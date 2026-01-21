#!/bin/bash
#
#SBATCH --job-name=team_mle
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1            # one launcher per node (procid 0..N-1)
#SBATCH --gpus-per-node=8              # GPUs per node
#SBATCH --account=cpt
#SBATCH --cpus-per-task=224
#SBATCH --mem=3588G
#SBATCH --exclude=metavmds1-a4-[5,148,156,173,239,47,121,113,214,98]
#
# This is the GCP/Slurm adaptation of retool-qwen3-8b-rl.sh. It launches a Ray
# cluster inside the allocation and runs the same train.py invocation from the
# head node.

set -ex

# --- user tweakables ---
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}
PYTHON=python
WORKSPACE_DIR="/home/sihanzeng_meta_com/uv"

export LOG_DIR=${WORKSPACE_DIR}/logs
export UNIQUE_JOB_NAME=${SLURM_JOB_NAME}_${SLURM_JOB_ID}
export LOG_PREFIX=${LOG_DIR}/${UNIQUE_JOB_NAME}
echo "Creating log directory at $LOG_PREFIX"
mkdir -p "$LOG_PREFIX"
exec > "$LOG_PREFIX/sbatch.out" 2>&1
export RAY_NODE_LOG="${LOG_PREFIX}/ray_node_${SLURM_PROCID:-0}.out"
export RAY_TEMP_DIR="${LOG_PREFIX}/ray_temp_${SLURM_PROCID:-0}"
export MAIN_NODE_LOG="${LOG_PREFIX}/main_${SLURM_PROCID:-0}.out"

# Ensure we always tear down Ray on exit.
cleanup() {
  ray stop --force || true
}
trap cleanup EXIT

SLIME_REPO_DIR="${WORKSPACE_DIR}/slime"
MEGATRON_REPO_DIR="${WORKSPACE_DIR}/Megatron-LM"

if [ ! -d "${MEGATRON_REPO_DIR}" ]; then
  echo "ERROR: Could not locate Megatron-LM checkout at ${MEGATRON_REPO_DIR}. Set MEGATRON_REPO_DIR to override." >&2
  exit 1
fi

PYTHONPATH_BASE="${MEGATRON_REPO_DIR}:${SLIME_REPO_DIR}:${SLIME_REPO_DIR}/examples/retool"
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH_VALUE="${PYTHONPATH_BASE}:${PYTHONPATH}"
else
  PYTHONPATH_VALUE="${PYTHONPATH_BASE}"
fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

export CUDA_HOME=/usr/local/cuda-12.8
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

# NCCL/IB tuning inspired by gemma_4b_cpt.slurm (adjust to your fabric as needed).
export TRITON_CACHE_DIR="${HOME}/local/tmp/${SLURM_JOB_ID}/${SLURM_PROCID:-0}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_HCA=${NCCL_IB_HCA:-^docker0,lo}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp0s19,enp192s20}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-8}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-8}
export NCCL_ASYNC_ERROR_HANDLING=1

which python

: "${TORCHDYNAMO_DISABLE:=1}"
export TORCHDYNAMO_DISABLE

# helper: head hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 20000))}

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

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export PYTHONBUFFERED=16
export MASTER_ADDR
export MASTER_PORT
source ${SLIME_REPO_DIR}/scripts/models/qwen3-8B.sh

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${PYTHONPATH_VALUE}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_SHARP_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
    "TORCH_COMPILE_DISABLE": "1"
  }
}
EOF
)

#
# Start Ray cluster inside the Slurm allocation
#
if [ "${SLURM_PROCID}" -eq 0 ]; then
  echo "Starting ray head on ${MASTER_ADDR}"
  ray stop --force || true
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus ${GPUS_PER_NODE} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir ${RAY_TEMP_DIR} | tee -a "${RAY_NODE_LOG}"
  sleep 5
else
  echo "Joining ray head at ${MASTER_ADDR}"
  ray stop --force || true
  ray start --address="${MASTER_ADDR}:6379" \
    --num-gpus ${GPUS_PER_NODE} \
    --disable-usage-stats \
    --temp-dir ${RAY_TEMP_DIR} | tee -a "${RAY_NODE_LOG}"
fi

# wait for all ray nodes to be ready
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --mpi=none true

MOUNT_DIR="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime"

# Only on head node submit the training job
if [ "${SLURM_PROCID}" -eq 0 ]; then

  CKPT_ARGS=(
      --hf-checkpoint ${MOUNT_DIR}/qwen3_8b
      --ref-load ${MOUNT_DIR}/qwen3_8b_torch_dist
      # --load /root/Qwen3-4B_slime/
      --save ${MOUNT_DIR}/qwen3-8b-rl-multi-turn/
     --save-interval 20
     --rotary-base 1000000
  )

  ROLLOUT_ARGS=(
     --prompt-data ${MOUNT_DIR}/data/dapo-math-17k/dapo-math-17k.jsonl
     --input-key prompt
     --label-key label
     --apply-chat-template
     --rollout-shuffle
     --reward-key score
     --num-rollout 3000
     --rollout-batch-size 16
     --n-samples-per-prompt 4
     --rollout-max-response-len 2048
     --rollout-temperature 0.8
     --global-batch-size 64
     --balance-data
  )

  EVAL_ARGS=(
     --eval-interval 20
     --eval-prompt-data aime  ${MOUNT_DIR}/data/aime-2024/aime-2024.jsonl
     --n-samples-per-eval-prompt 4
     --eval-max-response-len 4096
     --eval-top-p 0.7
  )

  PERF_ARGS=(
     --tensor-model-parallel-size 8
     --sequence-parallel
     --pipeline-model-parallel-size 1
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
     --wandb-group slurm-qwen3-8B-test-retool-${SLURM_JOB_ID}
     --wandb-key ${WANDB_KEY}
  )

  SGLANG_ARGS=(
     --rollout-num-gpus-per-engine 1
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
     --custom-generate-function-path generate_with_retool.generate
     --custom-rm-path generate_with_retool.reward_func
     --rollout-health-check-timeout 45
  )

  echo "Launching training directly on the head node"
  echo "Model args: ${MODEL_ARGS[*]}"
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
  2>&1 | tee ${MAIN_NODE_LOG}

  TRAIN_EXIT_CODE=${PIPESTATUS[0]:-0}
  echo "train.py exited with code ${TRAIN_EXIT_CODE}"
  ray stop --force || true
  exit ${TRAIN_EXIT_CODE}
fi

