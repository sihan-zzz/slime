import json
import logging
import os
from typing import Any

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
                    }
                if raw_sample is not None:
                    selected.append(raw_sample)
        logger.info(
            "eval_scan dataset=%s groups=%d selected=%d min_correct=%d",
            dataset_name,
            len(groups),
            len(selected),
            min_correct,
        )

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

    return True
