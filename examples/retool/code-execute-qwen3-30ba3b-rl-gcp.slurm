#!/bin/bash
#
#SBATCH --job-name=team_mle
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1            # one launcher per node (procid 0..N-1)
#SBATCH --gpus-per-node=8              # GPUs per node
#SBATCH --account=cpt
#SBATCH --cpus-per-task=224
#SBATCH --mem=3588G
#SBATCH --exclude=metavmds1-a4-[0]
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
export GPUS_PER_NODE
export TRAIN_NODES=${TRAIN_NODES:-2}
export ROLLOUT_NODES=$(($SLURM_NNODES - TRAIN_NODES))
echo "Training using ${TRAIN_NODES} nodes, rollout using ${ROLLOUT_NODES} nodes."
PYTHON=python
WORKSPACE_DIR="/home/sihanzeng_meta_com/uv"

export LOG_DIR=${WORKSPACE_DIR}/logs
export UNIQUE_JOB_NAME=${SLURM_JOB_ID}_${SLURM_JOB_NAME}
export LOG_PREFIX=${LOG_DIR}/${UNIQUE_JOB_NAME}
echo "Creating log directory at $LOG_PREFIX"
mkdir -p "$LOG_PREFIX"
exec > "$LOG_PREFIX/sbatch.out" 2>&1

export RAY_NODE_LOG_BASE="${LOG_PREFIX}/ray_node"
export RAY_NODE_LOG="${LOG_PREFIX}/ray_node_${SLURM_PROCID:-0}.out"
export RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_PROCID:-0}"
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
# Ray workers inherit the environment from the raylet; export PYTHONPATH before ray starts
export PYTHONPATH="${PYTHONPATH_VALUE}"

export CUDA_HOME=/usr/local/cuda-12.8
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

# NCCL/IB tuning inspired by gemma_4b_cpt.slurm (adjust to your fabric as needed).
# TorchMemorySaver used by sglang currently errors when expandable_segments is on.
# Leave unset to keep the default allocator unless you know you are not using TorchMemorySaver.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID}/${SLURM_PROCID:-0}"
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

# torch.compile in PyTorch 2.4 + Triton 3.4 in this env hits a duplicate template bug.
# Force Dynamo off so we fall back to eager mode unless explicitly overridden.
: "${TORCHDYNAMO_DISABLE:=1}"
export TORCHDYNAMO_DISABLE

# load modules / conda if necessary (example)
# module load cuda/12.1
# source /path/to/venv/bin/activate

MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Prefer the same interface used in other jobs on this cluster.
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOSTNAME" ip addr show enp192s20 | grep -Po 'inet \K[\d.]+' | head -n1)
: "${MASTER_ADDR:=${MASTER_HOSTNAME}}"
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
# MASTER_PORT=6379
export MASTER_ADDR MASTER_PORT

echo "SLURM_JOBID: $SLURM_JOB_ID"
echo "NODES: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

echo "=== HEALTH CHECK (silent unless errors) ==="

BAD_NODES=$(srun \
  --nodes=$SLURM_JOB_NUM_NODES \
  --ntasks-per-node=1 \
  --cpus-per-task=1 \
  --mpi=none \
  --kill-on-bad-exit=0 \
  bash -lc '
    H=$(hostname)

    # Try checks; if any fails, print hostname but DO NOT exit non-zero.
    failed=0

    nvidia-smi -L >/dev/null 2>&1 || failed=1
    ls /tmp >/dev/null 2>&1 || failed=1

    python3 - <<EOF >/dev/null 2>&1 || failed=1
import torch
import os
assert torch.cuda.is_available()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
EOF

    # If any check failed → print hostname
    if [[ $failed -ne 0 ]]; then
      echo "$H"
    fi

    exit 0   # <- IMPORTANT: always exit with 0 so srun never hangs
  ' | tr -d '[:space:]')

if [[ -n "$BAD_NODES" ]]; then
  echo "❌ BAD NODES DETECTED:"
  printf "%s\n" "$BAD_NODES"
  echo "Aborting before torchrun rendezvous."
  exit 1
fi

echo "=== HEALTH CHECK PASSED ==="

# clean up any stale processes (only on nodes in this allocation)
srun --nodes=$SLURM_NNODES \
     --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --mpi=none \
     bash -lc '
       echo "=== $(hostname): cleaning processes ==="
       pkill -9 sglang 2>/dev/null || true
       pkill -9 ray 2>/dev/null || true
       pkill -9 python 2>/dev/null || true
       ray stop --force 2>/dev/null || true
       echo "=== $(hostname): done ==="
     '
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
source ${SLIME_REPO_DIR}/scripts/models/qwen3-30B-A3B.sh 

# common runtime env used for ray job submit (tweak PYTHONPATH)
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${PYTHONPATH_VALUE}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_SHARP_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
    "TORCH_COMPILE_DISABLE": "1"
  }
}
EOF
)

MOUNT_DIR="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime"

CKPT_ARGS=(
   --hf-checkpoint ${MOUNT_DIR}/qwen3_30b_a3b_2507
   --ref-load ${MOUNT_DIR}/qwen3_30b_a3b_2507_torch_dist
   --save ${MOUNT_DIR}/qwen3-30b-rl-multi-turn/${UNIQUE_JOB_NAME}
   --save-interval 20
   --rotary-base 10000000
)

export NUM_ROLLOUT=${NUM_ROLLOUT:-500}

export TRAIN_DATA=${TRAIN_DATA:-${MOUNT_DIR}/data/APPS-dataset/APPS-synthetic-train.jsonl}
export MATH_DATA=${MATH_DATA:-${MOUNT_DIR}/data/math/5_mathematical_processed_data_prompt.jsonl}
export EVAL_DATA=${EVAL_DATA:-${MOUNT_DIR}/data/APPS-dataset/APPS-synthetic-test.jsonl}
ROLLOUT_ARGS=(
   --rollout-prompt-data code $TRAIN_DATA math $MATH_DATA
   --rollout-samples-per-dataset 16
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 0.6
   --rollout-top-p 0.95
   --rollout-top-k 20
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --global-batch-size 256
   --balance-data
   --rollout-num-gpus $((8 * $ROLLOUT_NODES))
   # --rollout-temperature 1.0 
   # --rollout-temperature-anneal-style cosine 
   # --rollout-temperature-anneal-end 0.6 
   # --rollout-temperature-anneal-steps 200
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-temperature 0.6
   --eval-top-p 0.95
   --eval-top-k 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 128000
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
   --wandb-group ${SLURM_JOB_ID}-${SLURM_JOB_NAME}
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.9
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-allow-auto-truncate
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # --no-fp8-wgrad
   --attention-backend flash

   # # for debug
   # --no-gradient-accumulation-fusion
   # --no-gradient-reduce-div-fusion
   # --disable-tp-comm-bulk-wgrad
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_code_execute.generate
   --custom-rm-path generate_with_code_execute.reward_func
   --rollout-health-check-timeout 45
)

TARGET="${MOUNT_DIR}/${SLURM_JOB_ID}/"
rm -r -f ${TARGET}
mkdir ${TARGET}
TRAIN_CMD=(
   ${PYTHON}
   train.py
   --actor-num-nodes ${TRAIN_NODES}
   --actor-num-gpus-per-node ${GPUS_PER_NODE}
   --output_sample_file "${TARGET}"
   --dump-details "${TARGET}/dump_details/"
)
TRAIN_CMD+=("${MODEL_ARGS[@]}")
TRAIN_CMD+=("${CKPT_ARGS[@]}")
TRAIN_CMD+=("${ROLLOUT_ARGS[@]}")
TRAIN_CMD+=("${OPTIMIZER_ARGS[@]}")
TRAIN_CMD+=("${GRPO_ARGS[@]}")
TRAIN_CMD+=("${WANDB_ARGS[@]}")
TRAIN_CMD+=("${PERF_ARGS[@]}")
TRAIN_CMD+=("${EVAL_ARGS[@]}")
TRAIN_CMD+=("${SGLANG_ARGS[@]}")
TRAIN_CMD+=("${MISC_ARGS[@]}")
TRAIN_CMD+=("${CUSTOM_ARGS[@]}")
TRAIN_CMD_STR=$(printf '%q ' "${TRAIN_CMD[@]}")
TRAIN_CMD_STR=${TRAIN_CMD_STR% }

export ADD_ARGS=${ADD_ARGS:-""}
echo "Launching training directly on the head node (this will BLOCK the sbatch until finished)"
echo "Model args: ${MODEL_ARGS[*]}"
export PYTHONPATH="${PYTHONPATH_VALUE}"

srun --nodes=${SLURM_NNODES} \
     --ntasks=${SLURM_NNODES} \
     --ntasks-per-node=1 \
     --kill-on-bad-exit=1 \
     --mpi=none \
     bash -lc "
set -euox pipefail

LOGFILE=\"${RAY_NODE_LOG_BASE}_\${SLURM_PROCID}.out\"

if [[ \$SLURM_PROCID -eq 0 ]]; then
   echo \"[HEAD] Starting Ray head on ${MASTER_ADDR}:${MASTER_PORT}\"
   ray start --head \
      --node-ip-address=${MASTER_ADDR} \
      --port=${MASTER_PORT} \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265 \
      --disable-usage-stats \
      > \$LOGFILE 2>&1 &

   sleep 5

   echo \"[HEAD] Running Python connectivity test\"
   export RAY_ADDRESS=${MASTER_ADDR}:${MASTER_PORT}
   # python examples/retool/ray_connectivity.py 2>&1 | tee ${RAY_NODE_LOG_BASE}_ray_connect.out
   ${TRAIN_CMD_STR} ${ADD_ARGS} \
   2>&1 | tee ${MAIN_NODE_LOG}

   TRAIN_EXIT_CODE=\${PIPESTATUS[0]:-0}
   echo \"python exited with code \${TRAIN_EXIT_CODE}, stopping ray across board.\"
   ray stop --force || true
   exit 1 
else
   echo \"[WORKER \$SLURM_PROCID] Starting worker and holding step open\"
   sleep 5
   nohup ray start --address=${MASTER_ADDR}:${MASTER_PORT} \
      --disable-usage-stats \
      --num-gpus ${GPUS_PER_NODE} \
      > \$LOGFILE 2>&1 &
   sleep infinity
fi
"
