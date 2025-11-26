#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORK_DIR="/home/sihanzeng_meta_com/uv"
MEGATRON_REPO_DIR="${WORK_DIR}/Megatron-LM"
SLIME_DIR="${WORK_DIR}/slime"
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

MOUNT_DIR="/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime"

CKPT_ARGS=(
   --hf-checkpoint ${MOUNT_DIR}/qwen3_8b
   --ref-load ${MOUNT_DIR}/qwen3_8b_torch_dist
   --save ${MOUNT_DIR}/qwen3-8b-rl-multi-turn/
   --save-interval 20
   --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data ${MOUNT_DIR}/data/APPS-dataset/APPS-synthetic-train.jsonl
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
   --eval-prompt-data apps  ${MOUNT_DIR}/data/APPS-dataset/APPS-synthetic-test.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 4096
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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
   --wandb-group slurm-qwen3-8B-code-local
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.9
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
     --custom-generate-function-path generate_with_code_execute.generate
     --custom-rm-path generate_with_code_execute.reward_func
     --rollout-health-check-timeout 45
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/workspace/slime\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
#   }
# }"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_REPO_DIR}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SHARP_DISABLE\": \"1\",
    \"NCCL_DEBUG\": \"WARN\",
    \"TORCH_COMPILE_DISABLE\": \"1\",
    \"TORCHDYNAMO_DISABLE\": \"1\"
  }
}"

TARGET="${MOUNT_DIR}/test/"

echo "Wrting samples to ${TARGET}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --output_sample_file "${TARGET}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}