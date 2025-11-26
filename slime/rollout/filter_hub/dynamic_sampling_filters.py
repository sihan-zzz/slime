import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample
import logging

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["check_reward_nonzero_std", "filter_truncated_samples"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    logger.info(f"zzzzlog check reward keep_samples={keep}, {rewards=}")
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def filter_truncated_samples(args, samples: list[Sample], **kwargs):
    """Filter out groups that contain any truncated samples."""
    # Check if any sample in the group is truncated
    truncated = [sample.status == Sample.Status.TRUNCATED for sample in samples]
    has_truncated = any(sample.status == Sample.Status.TRUNCATED for sample in samples)
    
    if has_truncated:
        logger.info(f"zzzzlog reject truncated samples {truncated=}")
        return DynamicFilterOutput(keep=False, reason="contains_truncated")
    else:
        return check_reward_nonzero_std(args, samples, **kwargs)
