import os
from typing import TextIO
import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking
import logging

# Configure logging with file and line information
def get_rank() -> int:
    """Get current process rank, defaults to 0 if not in distributed setting."""
    return int(os.environ.get("RANK", 0))


class RankFormatter(logging.Formatter):
    """Custom formatter that includes the trainer rank in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        # Add rank information to the record
        record.rank = get_rank()  # pyre-ignore[16]: LogRecord has no attribute rank
        return super().format(record)


# Set up logging with rank information
rank_formatter: RankFormatter = RankFormatter(
    fmt="%(asctime)s - [Rank %(rank)d] %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure root logger
root_logger: logging.Logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplication
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add console handler with rank formatter
console_handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
console_handler.setFormatter(rank_formatter)
root_logger.addHandler(console_handler)

logger: logging.Logger = logging.getLogger(__name__)

def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    logger.info("zzzzlog created rollout manager")

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    logger.info("zzzzlog created training models")

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    if args.offload_train and not args.enable_weights_backuper:
        actor_model.onload()
    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    if args.offload_train and not args.enable_weights_backuper:
        actor_model.offload()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # # special case for eval-only
    # if args.num_rollout == 0 and args.eval_interval is not None:
    #     ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):

        logger.info(f"zzzzlog starting {rollout_id=}")
        # TODO extract the duplicated eval logic
        # if args.eval_interval is not None and rollout_id == 0:
        #     ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        logger.info(f"zzzzlog {rollout_id=}, rollout data done")

        # data = ray.get(rollout_data_ref)

        # logger.info(f"[debug] rollout_id={rollout_id}, "
        #             f"type={type(data)}, keys={getattr(data, 'keys', lambda: None)()}")

        # # print shapes/lengths instead of full tensors
        # if isinstance(data, dict):
        #     for k, v in data.items():
        #         try:
        #             if hasattr(v, "shape"):
        #                 logger.info(f"[debug] key={k}, shape={v.shape}")
        #             elif hasattr(v, "__len__"):
        #                 logger.info(f"[debug] key={k}, len={len(v)}")
        #             else:
        #                 logger.info(f"[debug] key={k}, type={type(v)}")
        #         except Exception as e:
        #             logger.warn(f"[debug] error inspecting key={k}: {e}")

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        logger.info(f"zzzzlog {rollout_id=}, async training done")

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            logger.info(f"zzzzlog {rollout_id=}, cp caving done with freq {args.save_interval=}")

        if args.enable_weights_backuper:
            offload_train()
            onload_rollout()
            actor_model.update_weights()
        else:
            actor_model.clear_memory()
            onload_rollout()
            actor_model.update_weights()
            offload_train()

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))
            logger.info(f"zzzzlog {rollout_id=}, eval done with freq {args.eval_interval=}")

    ray.get(rollout_manager.dispose.remote())

def add_custom_args(parser):
    parser.add_argument(
        "--output_sample_file",
        type=str,
        default="",
        help="Path to local file to store generations",
    )
    return parser

if __name__ == "__main__":
    args = parse_args(add_custom_args)
    train(args)
