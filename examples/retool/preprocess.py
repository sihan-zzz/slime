#!/usr/bin/env python3
"""Split JSONL files into train/test/validation sets based on their `split` column."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import IO


PROMPT_TEMPLATE = (
    "Here is a programming problem and a candidate python solution. Think step by step and verify if the "
    "solution is correct for all valid inputs described by the problem. Do not provide fixes.\n\n"
    "The last line of your response should be of the form \n"
    "Answer: \\\\boxed{{$Answer}}\n"
    "where $Answer is 1 if the solution is correct and 0 if it is incorrect.\n\n"
    "**Problem**\n"
    "{question}\n\n"
    "**Solution**\n"
    "{program}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split every JSONL file under the source directory into train/test/validation files. "
            "Each output file contains the rows corresponding to that split."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/slime/data/new_code"),
        help="Directory that contains the new_code files (default: %(default)s)",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Destination directory for the train split files (default: <source>/train)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Destination directory for the test split files (default: <source>/test)",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Destination directory for the validation split files (default: <source>/validation)",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return
    else:
        path.mkdir(parents=True, exist_ok=True)


def build_split_name(filename: str, split_label: str) -> str:
    if "." in filename:
        stem, suffix = filename.rsplit(".", 1)
        return f"{stem}_{split_label}.{suffix}"
    return f"{filename}_{split_label}"


def normalize_split_name(name: str | None) -> str:
    if not name:
        raise ValueError("Missing 'split' value in record.")
    normalized = name.strip().lower()
    if normalized in {"validation", "valid", "val", "eval"}:
        return "validation"
    if normalized in {"train", "test"}:
        return normalized
    raise ValueError(f"Unsupported split '{name}'.")


def transform_record(row: dict) -> dict:
    program = row.get("program")
    if program is None:
        raise ValueError("Row missing 'program' field.")
    original_prompt = row.pop("prompt", None)
    if original_prompt is None:
        raise ValueError("Row missing 'prompt' field.")

    row["question"] = original_prompt
    row["prompt"] = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                question=row["question"],
                program=program,
            ),
        }
    ]
    return row


def split_file(
    src: Path,
    split_targets: dict[str, tuple[Path, str]],
) -> dict[str, int]:
    handles: dict[str, IO[str]] = {}
    counts: dict[str, int] = {split: 0 for split in split_targets}
    try:
        with src.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                split_name = normalize_split_name(row.get("split"))
                if split_name not in split_targets:
                    raise ValueError(f"Split '{split_name}' from file '{src}' has no configured target directory.")
                transformed = transform_record(row)
                target_dir, label = split_targets[split_name]

                writer = handles.get(split_name)
                if writer is None:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    out_path = target_dir / build_split_name(src.name, label)
                    writer = out_path.open("w", encoding="utf-8")
                    handles[split_name] = writer

                writer.write(json.dumps(transformed, ensure_ascii=False))
                writer.write("\n")
                counts[split_name] += 1
    finally:
        for writer in handles.values():
            writer.close()

    return counts


def main() -> None:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    train_dir = args.train_dir.expanduser().resolve() if args.train_dir else (source_dir / "train").resolve()
    test_dir = args.test_dir.expanduser().resolve() if args.test_dir else (source_dir / "test").resolve()
    validation_dir = (
        args.validation_dir.expanduser().resolve() if args.validation_dir else (source_dir / "validation").resolve()
    )

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    split_targets = {
        "train": (train_dir, "train"),
        "test": (test_dir, "test"),
        "validation": (validation_dir, "validation"),
    }

    for target_dir in {train_dir, test_dir, validation_dir}:
        ensure_empty_dir(target_dir)

    aggregate_counts = {split: 0 for split in split_targets}
    for path in files:
        counts = split_file(path, split_targets)
        for split_name, value in counts.items():
            aggregate_counts[split_name] += value

    for split_name, (target_dir, _) in split_targets.items():
        print(f"Wrote {aggregate_counts[split_name]} rows into directory {target_dir}")


if __name__ == "__main__":
    main()
