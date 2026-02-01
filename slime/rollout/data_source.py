import abc
import copy
import logging
import os
from pathlib import Path

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int, **kwargs) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
        self.datasets = None
        self.dataset_names = None
        self.dataset_sample_offsets = None
        self.dataset_epoch_ids = None
        self.num_datasets = 1

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            if getattr(args, "rollout_datasets", None):
                self.dataset_names = [cfg["name"] for cfg in args.rollout_datasets]
                self.datasets = [
                    Dataset(
                        cfg["path"],
                        tokenizer=tokenizer,
                        processor=processor,
                        max_length=args.rollout_max_prompt_len,
                        prompt_key=args.input_key,
                        multimodal_keys=args.multimodal_keys,
                        label_key=args.label_key,
                        metadata_key=args.metadata_key,
                        tool_key=args.tool_key,
                        apply_chat_template=args.apply_chat_template,
                        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                        disable_tool_use=getattr(args, "disable_tool_use", False),
                        seed=args.rollout_seed,
                    )
                    for cfg in args.rollout_datasets
                ]
                self.dataset_sample_offsets = [0] * len(self.datasets)
                self.dataset_epoch_ids = [0] * len(self.datasets)
                self.num_datasets = len(self.datasets)
                if self.args.rollout_shuffle:
                    for i, dataset in enumerate(self.datasets):
                        dataset.shuffle(self.dataset_epoch_ids[i])

                class _DatasetView:
                    def __init__(self, datasets):
                        self._datasets = datasets

                    def __len__(self):
                        return sum(len(dataset) for dataset in self._datasets)

                self.dataset = _DatasetView(self.datasets)
            else:
                self.dataset = Dataset(
                    args.prompt_data,
                    tokenizer=tokenizer,
                    processor=processor,
                    max_length=args.rollout_max_prompt_len,
                    prompt_key=args.input_key,
                    multimodal_keys=args.multimodal_keys,
                    label_key=args.label_key,
                    metadata_key=args.metadata_key,
                    tool_key=args.tool_key,
                    apply_chat_template=args.apply_chat_template,
                    apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                    disable_tool_use=getattr(args, "disable_tool_use", False),
                    seed=args.rollout_seed,
                )
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples, dataset_counts: dict[int, int] | None = None):
        # TODO further improve code
        if self.dataset is not None:
            if self.datasets is not None:
                prompt_samples = self._get_prompt_samples_multi_dataset(num_samples, dataset_counts)
            else:
                prompt_samples = self._get_prompt_samples_single_dataset(num_samples)
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                if isinstance(prompt_sample, tuple):
                    dataset_idx, prompt_sample_value = prompt_sample
                else:
                    dataset_idx, prompt_sample_value = None, prompt_sample
                sample = copy.deepcopy(prompt_sample_value)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                if dataset_idx is not None:
                    if sample.metadata is None:
                        sample.metadata = {}
                    sample.metadata["rollout_dataset_idx"] = dataset_idx
                    sample.metadata["rollout_dataset_name"] = self.dataset_names[dataset_idx]
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def _get_prompt_samples_single_dataset(self, num_samples: int):
        if self.sample_offset + num_samples <= len(self.dataset):
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.dataset.samples[self.sample_offset :]
            num_samples -= len(prompt_samples)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset.samples[:num_samples]
            self.sample_offset = num_samples
        return prompt_samples

    def _get_prompt_samples_multi_dataset(self, num_samples: int, dataset_counts: dict[int, int] | None):
        if dataset_counts is None:
            dataset_counts = {i: 0 for i in range(self.num_datasets)}
            for i in range(num_samples):
                dataset_counts[i % self.num_datasets] += 1
        else:
            dataset_counts = {i: count for i, count in dataset_counts.items() if count > 0}
            if num_samples != sum(dataset_counts.values()):
                num_samples = sum(dataset_counts.values())

        prompt_samples = []
        for dataset_idx, count in dataset_counts.items():
            if count <= 0:
                continue
            if dataset_idx >= len(self.datasets):
                continue
            dataset = self.datasets[dataset_idx]
            offset = self.dataset_sample_offsets[dataset_idx]
            if offset + count <= len(dataset):
                dataset_samples = dataset.samples[offset : offset + count]
                offset += count
            else:
                dataset_samples = dataset.samples[offset:]
                count -= len(dataset_samples)
                self.dataset_epoch_ids[dataset_idx] += 1
                if self.args.rollout_shuffle:
                    dataset.shuffle(self.dataset_epoch_ids[dataset_idx])
                dataset_samples += dataset.samples[:count]
                offset = count
            self.dataset_sample_offsets[dataset_idx] = offset
            prompt_samples.extend([(dataset_idx, sample) for sample in dataset_samples])

        return prompt_samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        if self.datasets is not None:
            state_dict["dataset_sample_offsets"] = self.dataset_sample_offsets
            state_dict["dataset_epoch_ids"] = self.dataset_epoch_ids
            state_dict["dataset_names"] = self.dataset_names
        else:
            state_dict["sample_offset"] = self.sample_offset
            state_dict["epoch_id"] = self.epoch_id
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.datasets is not None:
            self.dataset_sample_offsets = state_dict.get("dataset_sample_offsets", self.dataset_sample_offsets)
            self.dataset_epoch_ids = state_dict.get("dataset_epoch_ids", self.dataset_epoch_ids)
            if self.args.rollout_global_dataset and self.args.rollout_shuffle:
                for idx, dataset in enumerate(self.datasets):
                    dataset.shuffle(self.dataset_epoch_ids[idx])
        else:
            self.sample_offset = state_dict.get("sample_offset", 0)
            self.epoch_id = state_dict.get("epoch_id", 0)
            if self.args.rollout_global_dataset and self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        if self.num_datasets > 1:
            self.buffer = [[] for _ in range(self.num_datasets)]
        else:
            self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)
            if self.num_datasets > 1:
                logger.warning("buffer_filter_path is ignored for multi-dataset rollout buffers.")

    def get_samples(self, num_samples: int, dataset_counts: dict[int, int] | None = None) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples, dataset_counts=dataset_counts)
        num_samples -= len(samples)

        if num_samples == 0 and (dataset_counts is None or sum(dataset_counts.values()) == 0):
            return samples

        remaining_counts = None
        if dataset_counts is not None:
            remaining_counts = dict(dataset_counts)
            for group in samples:
                dataset_idx = _get_group_dataset_idx(group)
                if dataset_idx in remaining_counts:
                    remaining_counts[dataset_idx] -= 1
                    if remaining_counts[dataset_idx] <= 0:
                        del remaining_counts[dataset_idx]
            if not remaining_counts:
                return samples
            num_samples = sum(remaining_counts.values())

        samples += super().get_samples(num_samples=num_samples, dataset_counts=remaining_counts)
        return samples

    def _get_samples_from_buffer(
        self, num_samples: int, dataset_counts: dict[int, int] | None = None
    ) -> list[list[Sample]]:
        if num_samples == 0:
            return []
        if self.num_datasets > 1:
            return _pop_from_multibuffer(self.buffer, num_samples, dataset_counts)
        if len(self.buffer) == 0:
            return []
        if dataset_counts is not None:
            num_samples = min(num_samples, dataset_counts.get(0, 0))
            if num_samples == 0:
                return []
        return self.buffer_filter(self.args, None, self.buffer, num_samples)

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            if self.num_datasets > 1:
                dataset_idx = _get_group_dataset_idx(group)
                if dataset_idx >= self.num_datasets:
                    dataset_idx = 0
                self.buffer[dataset_idx].append(group)
            else:
                self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        if self.num_datasets > 1:
            return sum(len(buf) for buf in self.buffer)
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


def _get_group_dataset_idx(group: list[Sample]) -> int:
    if not group:
        return 0
    sample = group[0][0] if isinstance(group[0], list) else group[0]
    metadata = getattr(sample, "metadata", {}) or {}
    return int(metadata.get("rollout_dataset_idx", 0))


def _pop_from_multibuffer(
    buffers: list[list[list[Sample]]], num_samples: int, dataset_counts: dict[int, int] | None
) -> list[list[Sample]]:
    samples: list[list[Sample]] = []
    num_datasets = len(buffers)
    if dataset_counts is None:
        dataset_idx = 0
        while len(samples) < num_samples and any(buffers):
            if buffers[dataset_idx]:
                samples.append(buffers[dataset_idx].pop(0))
            dataset_idx = (dataset_idx + 1) % num_datasets
        return samples

    for dataset_idx, count in dataset_counts.items():
        if dataset_idx >= num_datasets or count <= 0:
            continue
        buffer = buffers[dataset_idx]
        take = min(len(buffer), count)
        if take > 0:
            samples.extend(buffer[:take])
            del buffer[:take]
    return samples
