# Fixed-pattern tool-integrated reasoning for math verification
import json
import logging
import re
from typing import Any, Optional, Tuple

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

from tool_sandbox import SEMAPHORE, tool_registry

logger: logging.Logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert mathematical verification assistant. "
    "Your task is to verify whether the candidate solution is correct."
)

TURN1_USER_TEMPLATE = (
    "You will be given a math problem and a candidate solution.\n"
    "Analyze the candidate solution step by step and identify any errors.\n"
    "Do NOT give the final verdict yet. Do NOT write any code.\n\n"
    "**Problem**\n{question}\n\n"
    "**Candidate Solution**\n{answer}\n"
)

TURN2_USER_INSTRUCTION = (
    "Now write executable Python code to validate the candidate solution.\n"
    "- Use only the Python standard library.\n"
    "- Print intermediate variables and checks step by step.\n"
    "- The code must be runnable as-is.\n"
    "Return the code inside a <tool_call> JSON with name code_interpreter.\n"
    "Example:\n"
    "<tool_call>\n"
    '{"name":"code_interpreter","arguments":{"code":"print(1+1)","stdin":""} }\n'
    "</tool_call>\n"

)

TURN3_USER_TEMPLATE = (
    "Here are the Python execution results:\n"
    "<interpreter>\n{tool_output}\n</interpreter>\n\n"
    "Using your earlier analysis and the tool output, give the final verdict.\n"
    "Output exactly one line in this exact format: \\boxed{{1}} or \\boxed{{0}}.\n"
    "Do not output any other text, explanation, punctuation, or spaces."
)


def _format_system(content: str) -> str:
    return f"<|im_start|>system\n{content}<|im_end|>\n"


def _format_user(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n"


def _format_assistant_prefix() -> str:
    return "<|im_start|>assistant\n"


def _extract_code(prediction: str) -> Tuple[str, Optional[str]]:
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    m = re.search(tool_call_pattern, prediction, re.DOTALL)
    if m:
        json_str = m.group(1)
        data = None
        try:
            data = json.loads(json_str)
        except Exception as e:
            try:
                json_str = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", json_str)
                data = json.loads(json_str)
            except Exception as e2:
                logger.warning(f"Failed to parse tool_call JSON: {e2}, {json_str=}, {prediction=}")
        if data and data.get("name") == "code_interpreter":
            args = data.get("arguments", {})
            code = (args.get("code") or "").strip()
            stdin = args.get("stdin", args.get("input", None))
            return code, stdin

    return "", None


def _log_wandb(payload: dict[str, Any]):
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


def _get_ctx_len(args, tokenizer) -> int:
    ctx_len = getattr(args, "rollout_max_context_len", None)
    if ctx_len:
        return int(ctx_len)
    model_max = getattr(tokenizer, "model_max_length", None)
    return int(model_max) if model_max else 40960


def _prepare_sampling_params(sampling_params, current_len, ctx_len):
    params = sampling_params.copy()
    max_new = params.get("max_new_tokens", None)
    allowed_new = max(0, ctx_len - current_len)
    if max_new is None:
        params["max_new_tokens"] = allowed_new
    else:
        params["max_new_tokens"] = max(0, min(max_new, allowed_new))
    return params


async def _generate_one(url, state, input_ids, sampling_params):
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    output = await post(url, payload)
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        return output, "", [], []

    token_logprobs = output["meta_info"]["output_token_logprobs"]
    token_ids = [item[1] for item in token_logprobs]
    log_probs = [item[0] for item in token_logprobs]
    text = state.tokenizer.decode(token_ids, skip_special_tokens=False)
    return output, text, token_ids, log_probs


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function."
    assert sample is not None and isinstance(sample, Sample)

    raw = (sample.metadata or {}).get("_raw_data", {})
    question = raw.get("question")
    answer = raw.get("answer")
    if question is None or answer is None:
        raise ValueError(f"Missing question/answer in sample.metadata['_raw_data'] {question=}, {answer=}")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    ctx_len = _get_ctx_len(args, state.tokenizer)

    # Turn 1: analysis
    turn1_user = TURN1_USER_TEMPLATE.format(question=question, answer=answer)
    conversation = _format_system(SYSTEM_PROMPT) + _format_user(turn1_user) + _format_assistant_prefix()
    initial_prompt = conversation
    sample.prompt = initial_prompt
    tokenizer = state.tokenizer
    prompt_tokens_ids = tokenizer(initial_prompt, add_special_tokens=False)["input_ids"]
    assistant_end_text = "<|im_end|>\n"
    assistant_prefix_text = _format_assistant_prefix()
    assistant_end_tokens = tokenizer(assistant_end_text, add_special_tokens=False)["input_ids"]
    assistant_prefix_tokens = tokenizer(assistant_prefix_text, add_special_tokens=False)["input_ids"]
    turn2_user_text = _format_user(TURN2_USER_INSTRUCTION)
    turn2_user_tokens = tokenizer(turn2_user_text, add_special_tokens=False)["input_ids"]

    response_text = ""
    response_token_ids = []
    response_logprob = []
    loss_masks = []
    debug = {"question": question, "answer": answer}
    context_token_ids = prompt_tokens_ids.copy()
    sample_index = getattr(sample, "index", None)

    def _append_non_generated(text: str, tokens: list[int] | None = None):
        nonlocal response_text, response_token_ids, response_logprob, loss_masks
        if tokens is None:
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        response_text += text
        response_token_ids += tokens
        response_logprob += [0.0] * len(tokens)
        loss_masks += [0] * len(tokens)

    params = _prepare_sampling_params(sampling_params, len(context_token_ids), ctx_len)
    if params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    output, turn1_text, turn1_tokens, turn1_logprobs = await _generate_one(url, state, context_token_ids, params)
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        sample.status = Sample.Status.ABORTED
        return sample

    context_token_ids += turn1_tokens
    response_text += turn1_text
    response_token_ids += turn1_tokens
    response_logprob += turn1_logprobs
    loss_masks += [1] * len(turn1_tokens)
    debug[0] = {"assistant": turn1_text, "finish_reason": output["meta_info"]["finish_reason"]["type"]}

    _append_non_generated(assistant_end_text, assistant_end_tokens)
    logger.info(
        "tir turn1 finished: index=%s prompt_tokens=%d response_tokens=%d finish_reason=%s",
        sample_index,
        len(prompt_tokens_ids),
        len(turn1_tokens),
        output["meta_info"]["finish_reason"]["type"],
    )
    _log_wandb(
        {
            "debug/tir_turn1_tokens": len(turn1_tokens),
            "debug/tir_prompt_tokens": len(prompt_tokens_ids),
        }
    )

    # Turn 2: code generation
    _append_non_generated(turn2_user_text, turn2_user_tokens)
    _append_non_generated(assistant_prefix_text, assistant_prefix_tokens)
    context_token_ids += assistant_end_tokens + turn2_user_tokens + assistant_prefix_tokens
    params = _prepare_sampling_params(sampling_params, len(context_token_ids), ctx_len)
    if params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    output, turn2_text, turn2_tokens, turn2_logprobs = await _generate_one(url, state, context_token_ids, params)
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        sample.status = Sample.Status.ABORTED
        return sample

    context_token_ids += turn2_tokens
    response_text += turn2_text
    response_token_ids += turn2_tokens
    response_logprob += turn2_logprobs
    loss_masks += [1] * len(turn2_tokens)

    code, stdin = _extract_code(turn2_text)
    tool_output = ""
    tool_call_count = 0
    if code:
        async with SEMAPHORE:
            args_dict: dict[str, Any] = {"code": code}
            if stdin is not None:
                args_dict["stdin"] = stdin
            tool_output = await tool_registry.execute_tool("code_interpreter", args_dict)
        tool_call_count = 1
    else:
        tool_output = "Error: No Python code found"

    _append_non_generated(assistant_end_text, assistant_end_tokens)

    tool_obs_text = f"<|im_start|>tools\n<interpreter>\n{tool_output}\n</interpreter>\n<|im_end|>\n"
    tool_obs_tokens = tokenizer(tool_obs_text, add_special_tokens=False)["input_ids"]
    _append_non_generated(tool_obs_text, tool_obs_tokens)

    debug[1] = {
        "assistant": turn2_text,
        "code": code,
        "tool_output": tool_output,
        "finish_reason": output["meta_info"]["finish_reason"]["type"],
    }
    logger.info(
        "tir turn2 finished: index=%s response_tokens=%d code_len=%d tool_output_len=%d finish_reason=%s",
        sample_index,
        len(turn2_tokens),
        len(code) if code else 0,
        len(tool_output),
        output["meta_info"]["finish_reason"]["type"],
    )
    _log_wandb(
        {
            "debug/tir_turn2_tokens": len(turn2_tokens),
            "debug/tir_code_len": len(code) if code else 0,
            "debug/tir_tool_output_len": len(tool_output),
            "debug/tir_tool_called": tool_call_count,
        }
    )

    # Turn 3: final verdict
    turn3_user = TURN3_USER_TEMPLATE.format(tool_output=tool_output)
    turn3_user_text = _format_user(turn3_user)
    turn3_user_tokens = tokenizer(turn3_user_text, add_special_tokens=False)["input_ids"]
    _append_non_generated(turn3_user_text, turn3_user_tokens)
    _append_non_generated(assistant_prefix_text, assistant_prefix_tokens)
    context_token_ids += assistant_end_tokens + tool_obs_tokens + turn3_user_tokens + assistant_prefix_tokens
    params = _prepare_sampling_params(sampling_params, len(context_token_ids), ctx_len)
    if params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    output, turn3_text, turn3_tokens, turn3_logprobs = await _generate_one(url, state, context_token_ids, params)
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        sample.status = Sample.Status.ABORTED
        return sample

    context_token_ids += turn3_tokens
    response_text += turn3_text
    response_token_ids += turn3_tokens
    response_logprob += turn3_logprobs
    loss_masks += [1] * len(turn3_tokens)
    debug[2] = {"assistant": turn3_text, "finish_reason": output["meta_info"]["finish_reason"]["type"]}
    logger.info(
        "tir turn3 finished: index=%s response_tokens=%d finish_reason=%s",
        sample_index,
        len(turn3_tokens),
        output["meta_info"]["finish_reason"]["type"],
    )
    _log_wandb(
        {
            "debug/tir_turn3_tokens": len(turn3_tokens),
            "debug/tir_total_response_tokens": len(response_token_ids),
            "debug/tir_total_context_tokens": len(context_token_ids),
        }
    )

    # Populate sample
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.rollout_log_probs = response_logprob
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_masks = loss_masks
    sample.response_actions = {"analysis": 1, "code": 1, "final": 1}
    sample.last_response = turn3_text
    sample.valid_tool_calls = tool_call_count
    sample.tool_call_count = tool_call_count
    sample.turn_finished = 3

    sample.payload_text = initial_prompt + response_text
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    sample.debug_dict = debug
    return sample


def compute_score(
    solution_str: str,
    ground_truth: str,
    valid_tool_call: int = 0,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict[str, Any]:
    ground_truth = int(ground_truth)
    try:
        pred = int(remove_boxed(last_boxed_only_string(solution_str)))
    except Exception:
        pred = None
    correct = pred == ground_truth
    reward = 1.0 if correct else -1.0
    reward += valid_tool_call * 0.1
    return {"score": reward, "pred": pred, "gt": ground_truth}


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    ground_truth = sample.label if sample.label is not None else ""
    response = sample.last_response if hasattr(sample, "last_response") else sample.response
    # Only score based on model response (exclude prompt/instructions)
    valid_tool_call = getattr(sample, "valid_tool_calls", 0)
    result = compute_score(response, ground_truth, valid_tool_call=valid_tool_call, strict_box_verify=True)

    logger.info(
        f"zzzzlog grading {result=}, on {valid_tool_call=}, {response[-100:]=} against {ground_truth=}"
    )

    if sample.index % 100 == 0:
        logger.info(f"zzzzlog sample print: {result=}, {sample.prompt=}, {sample.response=}, {sample.index=}")
    debug_dict = sample.debug_dict
    if debug_dict is not None:
        debug_dict["score_result"] = result
        debug_dict["group_index"] = sample.group_index

    return result
