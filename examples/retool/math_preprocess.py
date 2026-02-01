#!/usr/bin/env python3
"""Convert math JSONL rows into prompt-format chat messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MATH_PROMPT_TEMPLATE = (
    "Here is a math problem and a solution. Think step by step and verify if the "
    "solution is correct to the problem. Use the provided tool if necessary.\n\n"
    "The last line of your response should be of the form \n"
    "Answer: \\boxed{{$Answer}}\n"
    "where $Answer is 1 if the solution is correct and 0 if it is incorrect.\n\n"
    "**Problem**\n"
    "{prompt}\n\n"
    "**Solution**\n"
    "{answer}"
)

NEW_MATH_PROMPT_TEMPLATE = """
    You are an expert in mathematical verification. You will be given a problem and a candidate solution. Please carefully analyze and determine whether the solution is correct.\nPlease analyze the logical reasoning at each step in natural language carefully, use Python interpreter to verify the correctness of each computation, and synthesize your findings to reach a conclusion.
    **Problem**\n
    \n{prompt}\n
    **Solution**\n{candidate_solution}\n
    Please output your final answer in \\boxed{{}} as either 1 for correct solution or 0 for incorrect solution, e.g., \\boxed{{1}}.
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert math JSONL rows by moving 'prompt' into 'question' and "
            "formatting a chat prompt using MATH_PROMPT_TEMPLATE."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/sihanzeng_meta_com/uv/5_mathematical_processed_data.jsonl"),
        help="Input JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: <input>_prompted.jsonl)",
    )
    return parser.parse_args()


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_prompted{input_path.suffix}")


def transform_record(row: dict) -> dict:
    # original_prompt = row.get("prompt")
    # if original_prompt is None:
    #     raise ValueError("Row missing 'prompt' field.")
    answer = row.get("answer")
    if answer is None:
        raise ValueError("Row missing 'answer' field.")

    # row["question"] = original_prompt
    # print(row["question"])
    row["prompt"] = [
        {
            "role": "user",
            "content": NEW_MATH_PROMPT_TEMPLATE.format(
                prompt=row["question"],
                candidate_solution=row["answer"],
            ),
        }
    ]
    # print(row["prompt"])
    # 1 / 0
    return row


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else build_default_output_path(input_path)
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    count = 0
    with input_path.open("r", encoding="utf-8") as handle, output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            transformed = transform_record(row)
            # print(transformed)
            writer.write(json.dumps(transformed, ensure_ascii=False))
            writer.write("\n")
            count += 1
            # break

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
