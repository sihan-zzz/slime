"""
Multi-turn generation with Qwen3 native tool calling format.

Uses tokenizer.apply_chat_template(tools=...) for prompt construction and
<tool_response> tags for tool outputs (under user role), matching Qwen3's
built-in tool calling conventions.

Changes from generate_with_code_execute.py:
- Native format: apply_chat_template(tools=...) instead of custom Jinja2
- Tool responses: <tool_response> under <|im_start|>user instead of
  <interpreter> under <|im_start|>tools
- Simplified inter-turn: no verbose mid-turn instructions, only
  "give final answer" on the last allowed turn
- Fixed loss_mask attribute name (singular, matching rollout system)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

logger: logging.Logger = logging.getLogger(__name__)

# System prompt for the verifier
SYSTEM_PROMPT = "You are an expert verification assistant."

# Final-turn instruction when the model must stop using tools
FINAL_TURN_INSTRUCTION = (
    "You have used the maximum number of tool calls. "
    "Based on all information gathered, give your final answer now.\n"
    "Output \\boxed{1} if the solution is correct, or \\boxed{0} if incorrect."
)


def build_initial_prompt(tokenizer, user_prompt: str, use_tools: bool, system_prompt: str = None) -> str:
    """Build the initial prompt using Qwen3's native apply_chat_template."""
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    tools = tool_registry.get_tool_specs() if use_tools else None
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )


def _try_parse_tool_call(json_str: str) -> dict | None:
    """Try to parse a tool call JSON string, with fallback for markdown fences and escape issues.

    Instruct models sometimes wrap code in ```py...``` fences or produce escaping issues
    inside the JSON code field. This function tries standard JSON parse first, then falls back
    to regex extraction of the code field.
    """
    # Attempt 1: standard JSON parse (handles well-formed tool calls)
    try:
        clean = json_str.replace("\n", "\\n")
        data = json.loads(clean)
        if data.get("name") == "code_interpreter":
            code = data.get("arguments", {}).get("code", "").strip()
            if code:
                # Strip markdown fences if present in parsed code
                code = re.sub(r"^```(?:py|python)?\n?", "", code)
                code = re.sub(r"\n?```$", "", code)
                return {
                    "code": code.strip(),
                    "stdin": data["arguments"].get("stdin", data["arguments"].get("input", None)),
                }
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # Attempt 2: regex extraction when JSON parsing fails (markdown fences, escape issues)
    # Extract code value directly from the raw string
    code_match = re.search(
        r'"code"\s*:\s*"(```(?:py|python)?\s*\n)?(.*?)(\n?```)?"\s*[,}]',
        json_str,
        re.DOTALL,
    )
    if code_match:
        code = code_match.group(2).strip()
        if code:
            # Unescape common JSON escape sequences
            code = code.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
            return {"code": code, "stdin": None}

    logger.error(f"Tool call parse error (all attempts failed), json_str={json_str[:200]!r}...")
    return None


def parse_response(prediction: str) -> tuple[str, Any]:
    """Parse model response for tool calls or boxed answers.

    Returns:
        (action, content) where action is one of:
        - "answer": content is the boxed answer string
        - "tool_calls": content is list of (code, stdin) dicts
        - "invalid": content is error message
        - "no_action": content is empty string
    """
    # Check for \boxed{...} answer
    answer_match = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", prediction, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    # Check for <tool_call> blocks
    tool_call_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)
    if not tool_call_matches:
        return "no_action", ""

    results = []
    for json_str in tool_call_matches:
        parsed = _try_parse_tool_call(json_str)
        if parsed:
            results.append(parsed)

    if results:
        return "tool_calls", results
    return "invalid", "Failed to parse any valid tool calls"


async def execute_tool_calls(tool_calls: list[dict], max_per_turn: int = 4) -> tuple[str, int]:
    """Execute tool calls and return formatted output with count of successful executions.

    Returns:
        (formatted_output, num_executed)
    """
    results = []
    executed = 0
    for call in tool_calls[:max_per_turn]:
        async with SEMAPHORE:
            result = await tool_registry.execute_tool("code_interpreter", call)
        results.append(result)
        executed += 1

    output = "\n\n".join(results)
    return output, executed


def format_tool_response(output: str) -> str:
    """Format tool output as a native Qwen3 tool response turn.

    Produces: <|im_end|>\n<|im_start|>user\n<tool_response>\n{output}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n
    """
    return f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{output}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn generation with native Qwen3 tool calling."""
    assert not args.partial_rollout, "Partial rollout is not supported."
    assert isinstance(sample, Sample), "sample must be a Sample instance"

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    use_tools = not args.disable_tool_use

    # Build initial prompt with native tool format
    prompt = build_initial_prompt(state.tokenizer, sample.prompt, use_tools)
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    # Response tracking
    response_token_ids = []
    response_logprobs = []
    loss_mask = []
    response_text = ""
    tool_call_rounds = 0
    action_counts = {}
    debug = {"prompt": prompt, "index": sample.index}

    max_turns = TOOL_CONFIGS["max_turns"]
    ctx_len = 240959  # hardcoded to avoid sglang health-check errors

    for turn in range(max_turns):
        debug[turn] = {}

        # Calculate remaining token budget
        current_ids = prompt_token_ids + response_token_ids
        allowed_new = max(0, ctx_len - len(current_ids))
        turn_params = sampling_params.copy()
        max_new = turn_params.get("max_new_tokens", None)
        turn_params["max_new_tokens"] = min(max_new, allowed_new) if max_new else allowed_new

        if turn_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            logger.info(f"Context length exceeded at turn {turn}, index={sample.index}")
            break

        # Generate
        payload = {
            "input_ids": current_ids,
            "sampling_params": turn_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        # Handle abort
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Extract tokens and logprobs
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        is_truncated = finish_reason == "length"
        cur_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False)

        debug[turn]["response"] = cur_text
        debug[turn]["response_tokens"] = len(new_tokens)
        debug[turn]["finish_reason"] = finish_reason

        # Append model-generated tokens (loss_mask=1, or 0 if truncated)
        response_token_ids += new_tokens
        response_logprobs += new_logprobs
        response_text += cur_text
        if args.no_loss_on_truncated and is_truncated:
            loss_mask += [0] * len(new_tokens)
        else:
            loss_mask += [1] * len(new_tokens)

        sample.last_round_response = cur_text

        # No tools mode: single turn
        if not use_tools:
            break

        # Truncated by length: stop
        if is_truncated:
            break

        # Parse response
        action, content = parse_response(cur_text)
        action_counts[action] = action_counts.get(action, 0) + 1
        debug[turn]["action"] = action

        if action == "answer":
            break

        if action == "tool_calls":
            # Execute tool calls
            tool_output, num_executed = await execute_tool_calls(
                content, max_per_turn=TOOL_CONFIGS["max_tool_calls_per_turn"]
            )
            tool_call_rounds += 1
            debug[turn]["tool_calls"] = content
            debug[turn]["tool_output_len"] = len(tool_output)

            # Truncate long outputs
            if len(tool_output) > 5000:
                tool_output = tool_output[:5000] + "[Truncated]"

            # Check if this is the second-to-last turn
            if turn == max_turns - 2:
                # Format tool response with final-turn instruction
                obs = format_tool_response(tool_output)
                # Inject final turn instruction before the assistant generation
                obs = obs.rstrip("\n") + "\n" + FINAL_TURN_INSTRUCTION + "\n"
            else:
                obs = format_tool_response(tool_output)
        elif turn == max_turns - 1:
            # Last turn: no more tool calls possible
            break
        else:
            # No valid tool call or answer — append error feedback
            if action == "invalid":
                error_text = f"\n{content}\n"
            else:
                error_text = "\nNo tool call or answer found. Please make a tool call or provide your answer as \\boxed{{0}} or \\boxed{{1}}.\n"
            obs = format_tool_response(error_text)

        # Tokenize observation (tool response turn) — these are environment tokens
        obs_token_ids = state.tokenizer(obs, add_special_tokens=False)["input_ids"]
        response_token_ids += obs_token_ids
        response_logprobs += [0.0] * len(obs_token_ids)  # no logprob for env tokens
        loss_mask += [0] * len(obs_token_ids)  # don't train on env tokens
        response_text += obs
        debug[turn]["obs_tokens"] = len(obs_token_ids)

    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.rollout_log_probs = response_logprobs
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_mask  # singular — matches rollout system
    sample.response_actions = action_counts
    sample.valid_tool_calls = action_counts.get("tool_calls", 0)
    sample.tool_call_count = tool_call_rounds
    sample.turn_finished = turn + 1

    sample.payload_text = prompt + response_text
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text

    # Log metrics
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({
                "debug/native_format": 1,
                "debug/prompt_len": len(prompt_token_ids),
                "debug/total_response_len": len(response_token_ids),
                "debug/valid_response_len": sum(loss_mask),
                "debug/tools_used": tool_call_rounds,
                "debug/turns": sample.turn_finished,
            })
    except ImportError:
        pass

    if output is None:
        return sample

    # Set final status
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
) -> dict[str, Any]:
    """Compute reward score: +1 correct, -1 incorrect, +0.1 per valid tool call."""
    ground_truth = int(ground_truth)
    try:
        pred = int(remove_boxed(last_boxed_only_string(solution_str)))
    except Exception:
        pred = None
    correct = pred == ground_truth
    reward = (1.0 if correct else -1.0) + valid_tool_call * 0.1
    return {"score": reward, "pred": pred, "gt": ground_truth}


async def reward_func(args, sample, **kwargs):
    """Reward function: extract boxed answer from last response, compare to label."""
    if not isinstance(sample, Sample):
        raise TypeError("sample must be a Sample instance")

    last_round_response = getattr(sample, "last_round_response", "")
    ground_truth = sample.label if sample.label is not None else ""
    result = compute_score(last_round_response, ground_truth, valid_tool_call=sample.valid_tool_calls)

    logger.info(
        f"zzzzlog grading {result=}, {sample.valid_tool_calls=}, "
        f"{last_round_response[-100:]=} vs {ground_truth=}"
    )

    debug_dict = sample.debug_dict
    if debug_dict is not None:
        debug_dict["score_result"] = result
        debug_dict["group_index"] = sample.group_index

    return result
