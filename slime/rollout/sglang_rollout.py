import asyncio
import copy
import inspect
import logging
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils import logging_utils
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def _compute_rollout_temperature(args: Namespace, rollout_id: int) -> float:
    style = getattr(args, "rollout_temperature_anneal_style", None)
    end = getattr(args, "rollout_temperature_anneal_end", None)
    steps = getattr(args, "rollout_temperature_anneal_steps", None)
    if style is None or end is None or steps is None:
        if any(v is not None for v in (style, end, steps)):
            warned_key = "_warned_incomplete_anneal_args"
            if not getattr(_compute_rollout_temperature, warned_key, False):
                logger.warning(
                    "Temperature annealing is partially configured (style/end/steps). "
                    "Ignoring annealing and using --rollout-temperature."
                )
                setattr(_compute_rollout_temperature, warned_key, True)
        return args.rollout_temperature

    start = getattr(args, "rollout_temperature_anneal_start", None)
    if start is None:
        start = args.rollout_temperature

    if steps <= 0:
        return end

    ratio = min(max(rollout_id, 0), steps) / float(steps)
    if style == "linear":
        temp = start + (end - start) * ratio
    elif style == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        temp = end + (start - end) * coeff
    elif style == "exponential":
        if start <= 0 or end <= 0:
            warned_key = "_warned_exponential_invalid"
            if not getattr(_compute_rollout_temperature, warned_key, False):
                logger.warning(
                    "Exponential annealing requires positive start/end temperatures; "
                    "falling back to linear annealing."
                )
                setattr(_compute_rollout_temperature, warned_key, True)
            temp = start + (end - start) * ratio
        else:
            temp = start * ((end / start) ** ratio)
    else:
        temp = args.rollout_temperature

    if temp < 0:
        warned_key = "_warned_negative_temperature"
        if not getattr(_compute_rollout_temperature, warned_key, False):
            logger.warning("Computed rollout temperature is negative; clamping to 0.")
            setattr(_compute_rollout_temperature, warned_key, True)
        temp = 0.0

    return temp


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor:
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    output = await post(url, payload)

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Any
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)
    current_temperature = _compute_rollout_temperature(args, rollout_id)
    state.sampling_params["temperature"] = current_temperature

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size
    num_datasets = getattr(data_source, "num_datasets", 1)
    per_dataset_target = None
    if args.rollout_samples_per_dataset is not None:
        per_dataset_target = args.rollout_samples_per_dataset
        target_split_data_size = per_dataset_target // 2
        pos_data = [[] for _ in range(num_datasets)]
        neg_data = [[] for _ in range(num_datasets)]
    else:
        target_split_data_size = target_data_size / 2
        pos_data = []
        neg_data = []

    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")

    logger.info(
        f"zzzzlog Starting rollout: {args.rollout_batch_size=}, {args.n_samples_per_prompt=}, {(target_data_size * args.n_samples_per_prompt)=}",
    )
    scores = []

    def _total_collected() -> int:
        if per_dataset_target is None:
            return len(pos_data) + len(neg_data)
        return sum(len(pos_data[i]) + len(neg_data[i]) for i in range(num_datasets))

    def _remaining_per_dataset() -> list[int]:
        remaining = [
            per_dataset_target - (len(pos_data[i]) + len(neg_data[i])) for i in range(num_datasets)
        ]
        logging.info("Remaining per dataset: " + ", ".join(f"{i}: {remaining[i]}" for i in range(num_datasets)))
        return remaining

    def _allocate_dataset_counts(remaining: list[int], max_total: int) -> dict[int, int]:
        if max_total <= 0:
            return {}
        counts = [0] * len(remaining)
        allocated = 0
        dataset_idx = 0
        while allocated < max_total and any(remaining[i] > counts[i] for i in range(len(remaining))):
            if remaining[dataset_idx] > counts[dataset_idx]:
                counts[dataset_idx] += 1
                allocated += 1
            dataset_idx = (dataset_idx + 1) % len(remaining)
        return {idx: count for idx, count in enumerate(counts) if count > 0}

    def _get_group_dataset_idx(group: list[Sample]) -> int:
        if not group:
            return 0
        sample = group[0][0] if isinstance(group[0], list) else group[0]
        metadata = getattr(sample, "metadata", {}) or {}
        return int(metadata.get("rollout_dataset_idx", 0))

    while _total_collected() < target_data_size:
        newly_add_samples = 0

        # sending over-sampling requests to keep gpu busy
        while state.remaining_batch_size < args.over_sampling_batch_size:
            # get samples from the buffer and submit the generation requests.
            if per_dataset_target is None:
                samples = data_source.get_samples(args.over_sampling_batch_size)
            else:
                remaining = _remaining_per_dataset()
                if sum(remaining) == 0:
                    break
                dataset_counts = _allocate_dataset_counts(remaining, args.over_sampling_batch_size)
                samples = data_source.get_samples(sum(dataset_counts.values()), dataset_counts=dataset_counts)
            state.submit_generate_tasks(samples)
            newly_add_samples += 1
        logging.info(f"Submitted {newly_add_samples} new sample batches, total pending: {len(state.pendings)}")

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for i, task in enumerate(done):
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            scores.append(sum([sample.get_reward_value(args) for sample in group]) / args.n_samples_per_prompt)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.

            # keep positive and negative balancedly
            group_label = group[0].label
            if per_dataset_target is None:
                if group_label > 0.5:
                    if len(pos_data) < target_split_data_size:
                        pos_data.append(group)
                        pbar.update(args.n_samples_per_prompt)
                    else:
                        state.remaining_batch_size -= 1
                else:
                    if len(neg_data) < target_split_data_size:
                        neg_data.append(group)
                        pbar.update(args.n_samples_per_prompt)
                    else:
                        state.remaining_batch_size -= 1
            else:
                dataset_idx = _get_group_dataset_idx(group)
                if dataset_idx >= num_datasets:
                    dataset_idx = 0
                if group_label > 0.5:
                    if len(pos_data[dataset_idx]) < target_split_data_size:
                        pos_data[dataset_idx].append(group)
                        pbar.update(args.n_samples_per_prompt)
                    else:
                        state.remaining_batch_size -= 1
                else:
                    if len(neg_data[dataset_idx]) < target_split_data_size:
                        neg_data[dataset_idx].append(group)
                        pbar.update(args.n_samples_per_prompt)
                    else:
                        state.remaining_batch_size -= 1

    pbar.close()
    if per_dataset_target is None:
        data = pos_data + neg_data
    else:
        data = []
        for dataset_idx in range(num_datasets):
            data.extend(pos_data[dataset_idx])
            data.extend(neg_data[dataset_idx])
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )
    positive_batch = sum(group[0].label > 0.5 for group in data)
    negative_batch = sum(group[0].label < 0.5 for group in data)
    if per_dataset_target is None:
        logger.info(
            f"zzzzlog rollout_id={rollout_id} getting {len(data)} batches, each batch of {len(data[0])} responses, "
            f"{positive_batch=} positive, {negative_batch=} negative"
        )
    else:
        dataset_names = getattr(data_source, "dataset_names", None)
        dataset_logs = []
        for dataset_idx in range(num_datasets):
            name = dataset_names[dataset_idx] if dataset_names else str(dataset_idx)
            dataset_logs.append(
                f"{name}: pos={len(pos_data[dataset_idx])}, neg={len(neg_data[dataset_idx])}"
            )
        logger.info(
            f"zzzzlog rollout_id={rollout_id} getting {len(data)} batches, each batch of {len(data[0])} responses, "
            f"{positive_batch=} positive, {negative_batch=} negative, per_dataset=({'; '.join(dataset_logs)})"
        )

    response_action_counts: dict[str, int] = {}
    for group in data:
        for sample in group:
            if hasattr(sample, "response_actions"):
                for action, count in sample.response_actions.items():
                    response_action_counts[f"rollout/response_action_ratio/{action}"] = (
                        response_action_counts.get(f"rollout/response_action_ratio/{action}", 0) + count
                    )
    # normalize by total samples
    total_samples = len(data) * args.n_samples_per_prompt
    for action in response_action_counts:
        response_action_counts[action] /= total_samples
    dataset_epoch_metrics: dict[str, int] = {}
    dataset_epoch_ids = getattr(data_source, "dataset_epoch_ids", None)
    dataset_names = getattr(data_source, "dataset_names", None)
    if dataset_epoch_ids is not None and dataset_names:
        for dataset_idx, epoch_id in enumerate(dataset_epoch_ids):
            name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else str(dataset_idx)
            dataset_epoch_metrics[f"rollout/{name}_eopch_id"] = int(epoch_id)
    adhoc_metric_dict = {
        "rollout/temperature": current_temperature,
        "rollout/dynamic_filter/remain_positive_ratio": (
            positive_batch / (positive_batch + negative_batch) if (positive_batch + negative_batch) > 0 else 0.0
        ),
        "rollout/avg_tool_call_count": sum(sample.tool_call_count for group in data for sample in group)
        / (len(data) * args.n_samples_per_prompt),
        "rollout/avg_turns": sum(sample.turn_finished for group in data for sample in group)
        / (len(data) * args.n_samples_per_prompt),
        "rollout/avg_reward_before_filter": sum(scores) / len(scores),
        **response_action_counts,
        **dataset_epoch_metrics,
    }
    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # only write out completed and truncated samples
    rank = int(os.environ.get("RANK", 0))
    if args.output_sample_file:
        with open(args.output_sample_file + f"/rank_{rank}_rollout_{rollout_id}.jsonl", "a") as dump_file:
            for prompts in data:
                for sample in prompts:
                    dump_dict = sample.debug_dict if sample.debug_dict is not None else {}
                    dump_dict["rollout_id"] = rollout_id
                    dump_file.write(json.dumps(dump_dict, ensure_ascii=True) + "\n")

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics={**metric_gatherer.collect(), **adhoc_metric_dict}), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    store_raw_data = bool(getattr(args, "eval_scan_output_path", None))
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, store_raw_data)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            store_raw_data=store_raw_data,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    max_inflight = (
        args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        if args.rollout_num_gpus
        else args.sglang_server_concurrency
    )
    max_inflight = max(1, int(max_inflight))
    pending: set[asyncio.Task] = set()
    total_tasks = len(dataset.samples) * dataset_cfg.n_samples_per_eval_prompt
    completed_tasks = 0
    log_every = max(1, total_tasks // 1000)
    data = []
    do_print = True
    pbar = tqdm(total=total_tasks, desc=f"Eval {dataset_cfg.name}", disable=not do_print)

    eval_scan_enabled = bool(args.eval_scan_output_path and args.use_wandb)
    eval_scan_min_correct = getattr(args, "eval_scan_min_correct", None)
    eval_scan_reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
    eval_scan_group_size = dataset_cfg.n_samples_per_eval_prompt or 1
    eval_scan_group_count = len(dataset.samples)
    eval_scan_selected_count = 0
    eval_scan_handled_groups: set[int] = set()
    eval_scan_pending_groups: set[int] = set()
    eval_scan_group_stats: dict[int, dict[str, Any]] = {}
    eval_scan_initialized = False
    eval_scan_output_path = getattr(args, "eval_scan_output_path", None)
    eval_scan_raw_data_key = "_raw_data"

    if eval_scan_enabled and eval_scan_min_correct is None:
        raise ValueError("--eval-scan-min-correct must be set when using eval scan output.")

    if eval_scan_enabled and eval_scan_output_path:
        output_dir = os.path.dirname(eval_scan_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _eval_scan_get_reward_value(sample: Sample, reward_key: str | None) -> Any:
        reward = sample.reward
        if isinstance(reward, dict):
            if reward_key and reward_key in reward:
                return reward[reward_key]
            if "score" in reward:
                return reward["score"]
            if "pred" in reward and "gt" in reward:
                return reward["pred"] == reward["gt"]
            return None
        return reward

    def _eval_scan_is_correct(sample: Sample, reward_key: str | None) -> bool:
        if sample.status in {Sample.Status.TRUNCATED, Sample.Status.ABORTED, Sample.Status.FAILED}:
            return False
        value = _eval_scan_get_reward_value(sample, reward_key)
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return value > 0

    def _eval_scan_extract_raw_sample(sample: Sample) -> dict[str, Any] | None:
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict) and eval_scan_raw_data_key in metadata:
            raw = metadata[eval_scan_raw_data_key]
            if isinstance(raw, dict):
                return raw
        return None

    def _eval_scan_get_group_idx(sample: Sample) -> int:
        group_idx = getattr(sample, "group_index", None)
        if group_idx is not None:
            return int(group_idx)
        if sample.index is not None:
            return int(sample.index // eval_scan_group_size)
        # Fallback: treat each sample as its own group.
        return int(len(eval_scan_group_stats))

    def _eval_scan_update_group_stats(sample: Sample) -> None:
        if not eval_scan_enabled:
            return
        group_idx = _eval_scan_get_group_idx(sample)
        stats = eval_scan_group_stats.setdefault(
            group_idx,
            {
                "completed_count": 0,
                "correct_count": 0,
                "total_count": eval_scan_group_size,
                "raw_sample": None,
                "sample": sample,
            },
        )
        stats["completed_count"] += 1
        if _eval_scan_is_correct(sample, eval_scan_reward_key):
            stats["correct_count"] += 1
        raw_sample = _eval_scan_extract_raw_sample(sample)
        if raw_sample is not None and stats["raw_sample"] is None:
            stats["raw_sample"] = raw_sample
        if stats["completed_count"] >= stats["total_count"]:
            eval_scan_pending_groups.add(group_idx)

    def _eval_scan_flush_pending(force: bool = False) -> None:
        nonlocal eval_scan_initialized, eval_scan_selected_count
        if not eval_scan_enabled or not eval_scan_output_path:
            return
        if not force and not eval_scan_pending_groups:
            return

        selected: list[dict[str, Any]] = []
        to_handle = list(eval_scan_pending_groups)
        for group_idx in to_handle:
            stats = eval_scan_group_stats.get(group_idx)
            if stats is None:
                eval_scan_pending_groups.discard(group_idx)
                continue
            if stats["completed_count"] < stats["total_count"] and not force:
                continue

            eval_scan_pending_groups.discard(group_idx)
            eval_scan_handled_groups.add(group_idx)

            correct_count = int(stats["correct_count"])
            if correct_count >= int(eval_scan_min_correct):
                continue

            raw_sample = stats["raw_sample"]
            if raw_sample is None:
                sample = stats["sample"]
                raw_sample = {
                    "prompt": sample.prompt,
                    "label": sample.label,
                    "metadata": getattr(sample, "metadata", None),
                }
            if isinstance(raw_sample, dict):
                raw_sample = dict(raw_sample)
                raw_sample["correct_count"] = correct_count
                raw_sample["total_count"] = int(stats["total_count"])
                selected.append(raw_sample)

        if selected:
            mode = "a"
            if not eval_scan_initialized and rollout_id == 0 and getattr(args, "eval_scan_overwrite", False):
                mode = "w"
            with open(eval_scan_output_path, mode) as f:
                for item in selected:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")
            eval_scan_initialized = True
            eval_scan_selected_count += len(selected)

    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample.group_index = _i
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            while len(pending) >= max_inflight:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = task.result()
                    if isinstance(result, list):
                        data.extend(result)
                        completed = len(result)
                        for sample in result:
                            _eval_scan_update_group_stats(sample)
                    else:
                        data.append(result)
                        completed = 1
                        _eval_scan_update_group_stats(result)
                    if do_print:
                        sample_preview = result[0] if isinstance(result, list) else result
                        logger.info(
                            "eval_rollout_single_dataset example data: "
                            f"{[str(sample_preview.prompt) + sample_preview.response]} "
                            f"reward={sample_preview.reward}"
                        )
                        do_print = False
                    completed_tasks += completed
                    pbar.update(completed)
                    if args.eval_scan_output_path and args.use_wandb and completed_tasks % log_every == 0:
                        _eval_scan_flush_pending(force=False)
                        progress = completed_tasks / total_tasks if total_tasks else 1.0
                        handled_groups = len(eval_scan_handled_groups)
                        selected_so_far = eval_scan_selected_count
                        not_selected_so_far = max(handled_groups - selected_so_far, 0)
                        selected_ratio_so_far = (selected_so_far / handled_groups) if handled_groups else 0.0
                        logging_utils.log(
                            args,
                            {
                                f"eval_scan_{dataset_cfg.name}/progress": progress,
                                f"eval_scan_{dataset_cfg.name}/completed": completed_tasks,
                                f"eval_scan_{dataset_cfg.name}/total": total_tasks,
                                f"eval_scan_{dataset_cfg.name}/selected": selected_so_far,
                                f"eval_scan_{dataset_cfg.name}/not_selected": not_selected_so_far,
                                f"eval_scan_{dataset_cfg.name}/selected_ratio": selected_ratio_so_far,
                                f"eval_scan_{dataset_cfg.name}/groups_handled": handled_groups,
                                f"eval_scan_{dataset_cfg.name}/groups_total": eval_scan_group_count,
                                f"eval_scan_{dataset_cfg.name}/progress_step": completed_tasks,
                            },
                            step_key=f"eval_scan_{dataset_cfg.name}/progress_step",
                        )

            pending.add(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            if isinstance(result, list):
                data.extend(result)
                completed = len(result)
                for sample in result:
                    _eval_scan_update_group_stats(sample)
            else:
                data.append(result)
                completed = 1
                _eval_scan_update_group_stats(result)
            if do_print:
                sample_preview = result[0] if isinstance(result, list) else result
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample_preview.prompt) + sample_preview.response]} "
                    f"reward={sample_preview.reward}"
                )
                do_print = False
            completed_tasks += completed
            pbar.update(completed)
            _eval_scan_flush_pending(force=(completed_tasks == total_tasks))
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key

    # TODO (Bob): needs to add a utility function or organize this function better. only temp for now
    tp = sum(
        1
        for sample in data
        if "pred" in sample.reward and "gt" in sample.reward
        and sample.reward["pred"] == 1 and sample.reward["gt"] == 1 and not sample.status == Sample.Status.TRUNCATED
    )
    tn = sum(
        1
        for sample in data
        if "pred" in sample.reward and "gt" in sample.reward
        and sample.reward["pred"] == 0 and sample.reward["gt"] == 0 and not sample.status == Sample.Status.TRUNCATED
    )
    fp = sum(
        1
        for sample in data
        if "pred" in sample.reward and "gt" in sample.reward
        and sample.reward["pred"] == 1 and sample.reward["gt"] == 0 and not sample.status == Sample.Status.TRUNCATED
    )
    fn = sum(
        1
        for sample in data
        if "pred" in sample.reward and "gt" in sample.reward
        and sample.reward["pred"] == 0 and sample.reward["gt"] == 1 and not sample.status == Sample.Status.TRUNCATED
    )

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    accuracy_all = (tp + tn) / len(data) if len(data) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    num_none = sum(1 for sample in data if sample.reward["pred"] is None)
    truncated = sum(1 for sample in data if sample.status == Sample.Status.TRUNCATED)
    average_response_length = sum(sample.response_length for sample in data if not sample.status == Sample.Status.TRUNCATED) / len(data)
    average_tool_call_count = sum(sample.tool_call_count for sample in data if not sample.status == Sample.Status.TRUNCATED) / len(data)
    average_turn_finished = sum(sample.turn_finished for sample in data if not sample.status == Sample.Status.TRUNCATED) / len(data)
    logger.info(
        f"Eval {rollout_id=}, {tp=}, {tn=}, {fp=}, {fn=}, {accuracy=}, {accuracy_all=}, {recall=}, {precision=}"
        f"{tnr=}, {f1=}, {num_none=}, {truncated=}"
    )
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
            "accuracy": accuracy,
            "accuracy_all": accuracy_all,
            "recall": recall,
            "precision": precision,
            "tnr": tnr,
            "f1": f1,
            "ratio_none": num_none / len(data),
            "average_response_length": average_response_length,
            "average_tool_call_count": average_tool_call_count,
            "average_turn_finished": average_turn_finished,
        }
    }

# BOB: this is the rollout function as default
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source))
    data_source.add_samples(aborted_samples)
    return output
