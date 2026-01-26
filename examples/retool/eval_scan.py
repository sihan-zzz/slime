import json
import logging
import os
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_RAW_DATA_KEY = "_raw_data"


def _get_reward_value(sample: Sample, reward_key: str | None) -> Any:
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


def _is_correct(sample: Sample, reward_key: str | None) -> bool:
    if sample.status in {Sample.Status.TRUNCATED, Sample.Status.ABORTED, Sample.Status.FAILED}:
        return False
    value = _get_reward_value(sample, reward_key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value > 0


def _extract_raw_sample(samples: list[Sample]) -> dict[str, Any] | None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict) and _RAW_DATA_KEY in metadata:
            return metadata[_RAW_DATA_KEY]
    return None

from time import sleep

def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None) -> bool:
    output_path = getattr(args, "eval_scan_output_path", None)
    if not output_path:
        return False

    min_correct = getattr(args, "eval_scan_min_correct", None)
    if min_correct is None:
        raise ValueError("--eval-scan-min-correct must be set when using eval scan output.")

    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
    group_size = getattr(args, "n_samples_per_eval_prompt", 1) or 1

    selected = []
    total_groups = 0
    selected_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for dataset_name, payload in data.items():
        samples = payload.get("samples") or []
        groups: dict[int, list[Sample]] = {}
        for sample in samples:
            group_idx = getattr(sample, "group_index", None)
            if group_idx is None:
                if sample.index is not None:
                    group_idx = sample.index // group_size
                else:
                    group_idx = len(groups)
            groups.setdefault(group_idx, []).append(sample)

        group_counts[dataset_name] = len(groups)
        selected_counts[dataset_name] = 0
        for group_idx, group_samples in groups.items():
            total_groups += 1
            correct_count = sum(_is_correct(sample, reward_key) for sample in group_samples)
            if correct_count < min_correct:
                raw_sample = _extract_raw_sample(group_samples)
                if raw_sample is None and group_samples:
                    raw_sample = {
                        "prompt": group_samples[0].prompt,
                        "label": group_samples[0].label,
                        "metadata": getattr(group_samples[0], "metadata", None),
                        "correct_count": correct_count,
                        "total_count": len(group_samples),
                    }
                if raw_sample is not None:
                    selected.append(raw_sample)
                    selected_counts[dataset_name] += 1
        logger.info(
            "eval_scan dataset=%s groups=%d selected=%d min_correct=%d",
            dataset_name,
            len(groups),
            selected_counts[dataset_name],
            min_correct,
        )

    log_dict = extra_metrics or {}
    for dataset_name, payload in data.items():
        selected_count = selected_counts.get(dataset_name, 0)
        group_count = group_counts.get(dataset_name, 0)
        not_selected = max(group_count - selected_count, 0)
        selected_ratio = (selected_count / group_count) if group_count else 0.0
        log_dict[f"eval_scan/{dataset_name}/selected"] = selected_count
        log_dict[f"eval_scan/{dataset_name}/not_selected"] = not_selected
        log_dict[f"eval_scan/{dataset_name}/total"] = group_count
        log_dict[f"eval_scan/{dataset_name}/selected_ratio"] = selected_ratio

        if "accuracy" in payload:
            log_dict[f"eval/{dataset_name}-acc"] = payload["accuracy"]
        if "precision" in payload:
            log_dict[f"eval/{dataset_name}-precision"] = payload["precision"]
        if "recall" in payload:
            log_dict[f"eval/{dataset_name}-recall"] = payload["recall"]
        if "tnr" in payload:
            log_dict[f"eval/{dataset_name}-tnr"] = payload["tnr"]
        if "f1" in payload:
            log_dict[f"eval/{dataset_name}-f1"] = payload["f1"]
        if "average_response_length" in payload:
            log_dict[f"eval/{dataset_name}-average_response_length"] = payload["average_response_length"]
        if "average_tool_call_count" in payload:
            log_dict[f"eval/{dataset_name}-average_tool_call_count"] = payload["average_tool_call_count"]
        if "average_turn_finished" in payload:
            log_dict[f"eval/{dataset_name}-average_turn_finished"] = payload["average_turn_finished"]
        if "truncated" in payload:
            truncated = payload["truncated"]
            log_dict[f"eval/{dataset_name}-truncated_ratio"] = (
                sum(truncated) / len(truncated) if truncated else 0.0
            )

    total_selected = len(selected)
    total_not_selected = max(total_groups - total_selected, 0)
    log_dict["eval_scan/selected"] = total_selected
    log_dict["eval_scan/not_selected"] = total_not_selected
    log_dict["eval_scan/total"] = total_groups
    log_dict["eval_scan/selected_ratio"] = (total_selected / total_groups) if total_groups else 0.0

    if not selected:
        logger.info("eval_scan: no prompts selected for output.")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = "a"
        if rollout_id == 0 and getattr(args, "eval_scan_overwrite", False):
            mode = "w"
        with open(output_path, mode) as f:
            for item in selected:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        logger.info(
            "eval_scan wrote %d/%d prompts to %s (rollout_id=%s, overwrite=%s)",
            len(selected),
            total_groups,
            output_path,
            rollout_id,
            bool(getattr(args, "eval_scan_overwrite", False)),
        )

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    sleep(30)
    return True
