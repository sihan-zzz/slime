# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**slime** is an LLM post-training framework for RL scaling. It connects Megatron-LM (training) with SGLang (inference/rollout) via Ray-based orchestration. It powers GLM-4.5/4.6/4.7 and supports Qwen3, DeepSeek V3/R1, Llama 3, and other model families.

## Build & Install

```bash
# Full environment setup (CUDA required, Linux)
bash build_conda.sh

# Dev install (if Megatron-LM + SGLang already installed)
pip install -e .
```

Key external dependencies pinned to specific commits: SGLang and Megatron-LM (see `build_conda.sh` for exact commits). Patches from `docker/patch/` must be applied to both.

## Linting & Code Style

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks (ruff, autoflake, isort, black)
pre-commit run --all-files --show-diff-on-failure --color=always
```

Style: black (line length 119), isort (black-compatible profile), ruff for lint errors/bugbear/pyupgrade. See `pyproject.toml` for full config.

## Running Training

Training is submitted as a Ray job. The typical pattern:

```bash
# Start Ray head node
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "/path/to/Megatron-LM/"}}' \
  -- python3 train.py <args>
```

- `train.py` — synchronous training loop
- `train_async.py` — asynchronous training loop (overlaps rollout generation with training)

Pre-configured launch scripts are in `scripts/` (e.g., `run-qwen3-4B.sh`). Model architecture configs live in `scripts/models/`.

## Tests

Tests require GPUs and a Ray cluster. Most tests are integration tests that download models and run small training jobs.

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_qwen3_0.6B_fsdp_colocated_2xGPU.py

# Run by marker
pytest -m "unit"         # unit tests only
pytest -m "not system"   # exclude system tests
```

Test markers: `unit`, `integration`, `system`, `acceptance`, `skipduringci`, `pleasefixme`.

Environment variables affecting tests: `SLIME_TEST_FEW_GPU`, `SLIME_TEST_TIGHT_DEVICE_MEMORY`.

## Architecture

### Three-Module Design

```
Training (Megatron/FSDP)  <-->  Data Buffer  <-->  Rollout (SGLang + Router)
```

1. **Training**: Reads batches from the data buffer, performs RL training (GRPO, PPO, etc.), syncs parameters to rollout after each step.
2. **Rollout**: SGLang engines generate responses, compute rewards via reward models/verifiers, and store results in the data buffer.
3. **Data Buffer**: Bridge managing prompts, rollout data, and custom data generation workflows.

### Package Structure

- **`slime/backends/`** — Training backend implementations:
  - `megatron_utils/` — Megatron-LM backend (`MegatronTrainRayActor`): tensor/pipeline/expert parallelism, checkpointing, loss computation, weight conversion (`megatron_to_hf/`)
  - `fsdp_utils/` — HuggingFace + FSDP backend (`FSDPTrainRayActor`): simpler distributed training, custom MoE kernels
  - `sglang_utils/` — SGLang inference engine arguments and utilities
- **`slime/ray/`** — Ray-based orchestration:
  - `placement_group.py` — GPU allocation with PACK strategy, creates training models and rollout manager
  - `actor_group.py` — `RayTrainGroup` coordinating distributed training actors
  - `rollout.py` — `RolloutManager` for generation, evaluation, weight sync
  - `train_actor.py` — Base `TrainRayActor` class
- **`slime/rollout/`** — Rollout data generation:
  - `generate_hub/` — Pluggable generation strategies
  - `rm_hub/` — Reward model implementations (deepscaler, f1, etc.)
  - `filter_hub/` — Dynamic data sampling filters
- **`slime/router/`** — Request routing with middleware
- **`slime/utils/`** — Utilities: argument parsing, PPO/GAE math, data processing, logging, metrics, memory management, types (`RolloutBatch`)

### Plugins

- **`slime_plugins/mbridge/`** — Model bridge adapters for non-standard architectures (GLM4, Qwen3Next, MIMO, MoE variants)
- **`slime_plugins/megatron_bridge/`** — Megatron integration bridges
- **`slime_plugins/models/`** — Custom model definitions
- **`slime_plugins/rollout_buffer/`** — Async agent trajectory generation via HTTP API with custom `Generator` classes

### Arguments

Three categories (see `slime/utils/arguments.py`):
1. **Megatron args**: Standard Megatron-LM args (e.g., `--tensor-model-parallel-size 2`)
2. **SGLang args**: Prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static 0.7`)
3. **slime-specific args**: Cluster config (`--actor-num-nodes`, `--rollout-num-gpus`, `--colocate`), training config (`--train-backend megatron|fsdp`, `--advantage-estimator grpo`), rollout config (`--rm-type`, `--n-samples-per-prompt`)

## Model Conversion Tools

Located in `tools/`:
- `convert_to_hf.py` / `convert_fsdp_to_hf.py` — Convert checkpoints to HuggingFace format
- `convert_hf_to_torch_dist.py` / `convert_torch_dist_to_hf.py` — Convert between HF and torch distributed formats
- `convert_hf_to_fp8.py` / `convert_hf_to_int4.py` — Quantization conversions

---

## Our Project: Universal Verifier RL Training

We are training a **universal verifier** — an LLM that verifies whether candidate solutions to code and math problems are correct. The model outputs `\boxed{1}` (correct) or `\boxed{0}` (incorrect), and can use a code interpreter tool during verification.

### Task Description

- **Code verification**: Given a programming problem + candidate Python solution, verify correctness. Data sourced from LiveCodeBench with GPT-generated candidate solutions.
- **Math verification**: Given a math problem + candidate solution, verify correctness. Data sourced from AIME 2025 and other math benchmarks with solutions from various models (Kimi K2, Claude Opus, etc.).
- **Training signal**: Binary reward (+1 correct, -1 incorrect) with a +0.1 bonus per valid tool call to encourage tool use.

### Data

Two JSONL datasets in the working directory (`/Users/sihanzeng/Documents/work/uv/`):

| File | Type | Rows | Description |
|------|------|------|-------------|
| `2_python_filter_by_qwen3_4b_5791_4b_tool_eval_scan_train.jsonl` | Code | 2620 | LiveCodeBench problems filtered to questions where Qwen3-8B has low accuracy. Fields: `id`, `dataset_name`, `model_name`, `program`, `label` (bool), `question`, `prompt` (chat messages) |
| `5_mathematical_processed_data_v2_prompt_test.jsonl` | Math | 1596 | Math verification problems (AIME 2025 etc.) filtered similarly. Fields: `prompt` (chat messages), `answer`, `label` (bool), `model_name`, `dataset_name`, `question` |

Both datasets are filtered from an 8B model eval scan: we only keep questions where the 8B model has low accuracy (i.e., hard examples for the verifier).

### Key Files

- **`examples/retool/generate_with_code_execute.py`** — Multi-turn generation function for code/math verification with tool calls. Handles the conversation loop: model generates → extracts `<tool_call>` → executes in sandbox → feeds result back → model gives final `\boxed{}` answer.
- **`examples/retool/generate_with_tir.py`** — Fixed 3-turn "Tool-Integrated Reasoning" (TIR) variant: Turn 1 = analysis, Turn 2 = write verification code, Turn 3 = final verdict. More structured than the free-form multi-turn approach.
- **`examples/retool/tool_sandbox.py`** — Python code sandbox (`PythonSandbox` class) and `ToolRegistry`. Executes model-generated code in subprocess with safety checks, timeout (5s), memory limits (4GB). Configurable via env vars `MAX_TURNS`, `MAX_TOOL_CALLS_PER_TURN`.
- **`examples/retool/eval_scan.py`** — Eval scan hook: runs evaluation, filters prompts where `correct_count < --eval-scan-min-correct`, and writes them to `--eval-scan-output-path`. Used for iterative data filtering (find hard problems).
- **`examples/retool/math_preprocess.py`** — Converts raw math JSONL into prompt format using templates from `slime/utils/prompting.py`.
- **`slime/utils/prompting.py`** — Prompt templates for math and code verification (with/without tool requirement).

### Generation Flow

The `generate_with_code_execute.generate()` function:
1. Formats the initial prompt with system message + tool spec (unless `--disable-tool-use`)
2. Loops up to `MAX_TURNS` (default 10):
   - Sends context to SGLang router for generation
   - Parses response for `<tool_call>` blocks or `\boxed{}` answers
   - If tool call found: executes code in sandbox, appends `<interpreter>` output, continues
   - If `\boxed{}` found: done
   - On second-to-last turn: instructs model to give final answer without more tool calls
3. Tracks `loss_masks` (0 for environment/tool output tokens, 1 for model-generated tokens)
4. Returns `Sample` with tokens, log probs, loss masks, and debug info

### Reward Function

`generate_with_code_execute.reward_func()`:
- Extracts `\boxed{0}` or `\boxed{1}` from the model's last response
- Compares against the ground truth `label` field
- Score = +1.0 (correct) or -1.0 (incorrect) + 0.1 per valid tool call

### Launching Jobs on GCP

Jobs run on a SLURM cluster. The canonical launch pattern:

```bash
WANDB_KEY=<key> \
TRAIN_DATA=/mnt/lustre/.../train.jsonl \
MATH_DATA=/path/to/math.jsonl \
EVAL_DATA="code /path/to/eval.jsonl" \
TRAIN_NODES=2 \
ADD_ARGS="--num-rollout 500 --eval-scan-output-path /path/to/output.jsonl --eval-scan-min-correct 7 --n-samples-per-eval-prompt 8" \
sbatch --nodes=8 --requeue --job-name=<name> examples/retool/code-execute-qwen3-8b-rl-gcp.slurm
```

Key environment variables for the SLURM scripts:
- `WANDB_KEY` — W&B API key for experiment tracking
- `TRAIN_DATA` — Path to code training JSONL
- `MATH_DATA` — Path to math training JSONL
- `EVAL_DATA` — Space-separated `<name> <path>` pairs for eval datasets
- `TRAIN_NODES` — Number of nodes for training (rest used for rollout)
- `NUM_ROLLOUT` — Number of rollout steps (default 500)
- `ADD_ARGS` — Extra arguments appended to the training command

Available SLURM scripts in `examples/retool/`:
- `code-execute-qwen3-4b-rl-gcp.slurm` — Qwen3-4B
- `code-execute-qwen3-8b-rl-gcp.slurm` — Qwen3-8B (primary)
- `code-execute-qwen3-30ba3b-rl-gcp.slurm` — Qwen3-30B-A3B MoE
- `code-execute-qwen3-30ba3b-inst-rl-gcp.slurm` — Qwen3-30B-A3B Instruct
- `code-execute-qwen3-32b-rl-gcp.slurm` — Qwen3-32B

### GCP Paths

- Checkpoints: `/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/`
- HF model weights: `${MOUNT_DIR}/qwen3_8b`, `${MOUNT_DIR}/qwen3_4b`, etc.
- Torch dist weights: `${MOUNT_DIR}/qwen3_8b_torch_dist`
- Training data: `${MOUNT_DIR}/data/`
- Workspace: `/home/sihanzeng_meta_com/uv/`
- Logs: `${WORKSPACE_DIR}/logs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}/`

### Custom Train Args (added in `train.py`)

- `--disable-tool-use` — Run generation without tools (pure reasoning baseline)
- `--eval-scan-output-path` — Write filtered eval prompts to this JSONL path
- `--eval-scan-min-correct` — Keep prompts where correct_count < this threshold
- `--eval-scan-overwrite` — Overwrite eval-scan output on rollout_id==0
- `--system-prompt` — Custom system prompt for the model
- `--no_loss_on_truncated` — Zero out loss on truncated examples

### Mixed Dataset Training

Training uses `--rollout-prompt-data` with named datasets to mix code and math:
```
--rollout-prompt-data code $TRAIN_DATA math $MATH_DATA
--rollout-samples-per-dataset 16
```
This samples 16 prompts from each dataset per batch, enabling joint code+math verification training.

### Local Branch Changes (vs upstream)

Key commits on our branch (oldest to newest):
- `daf9459` enable code-exe — initial code execution support
- `3fd2440` / `b2b866e` — adding 32B/4B model configs
- `80be05c` eval scan — eval scan for iterative data filtering
- `458c667` temperature annealing — rollout temperature annealing support
- `f2b8096` / `d012c8f` — math dataset preprocessing and training
- `45fce1a` enable mixing dataset — multi-dataset training support
- `1862f52` process reward — reward computation improvements
- `f384f30` tool call bug fix — tool call not enforced fix
- `824aa33` prompt tuning — various prompt template iterations
- `c037b7c` generate with TIR — fixed 3-turn TIR generation strategy
