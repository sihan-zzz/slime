import importlib
import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")


from .arguments import parse_args, set_default_megatron_args, validate_args
from .checkpoint import load_checkpoint, save_checkpoint

logging.getLogger().setLevel(logging.WARNING)


__all__ = [
    "parse_args",
    "validate_args",
    "load_checkpoint",
    "save_checkpoint",
    "set_default_megatron_args",
    "MegatronTrainRayActor",
    "init",
    "initialize_model_and_optimizer",
]

_LAZY_ATTRS = {
    "init": ".initialize",
    "initialize_model_and_optimizer": ".model",
    "MegatronTrainRayActor": ".actor",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
