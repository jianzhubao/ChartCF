#!/usr/bin/env python3
"""Select low-similarity counterfactual pairs for DPO training."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError(f"Sample has no assistant message: {messages}")


def validate_dpo_pair(first: dict[str, Any], second: dict[str, Any], pair_index: int) -> float:
    for key in ("messages", "images", "question_type", "similarity_score"):
        if first.get(key) != second.get(key):
            raise ValueError(f"Pair {pair_index} has mismatched {key}")

    if "similarity_score" not in first:
        raise ValueError(f"Pair {pair_index} is missing similarity_score")

    assistant_content = get_assistant_content(first["messages"])
    if first.get("rejected_response") == assistant_content:
        raise ValueError(f"Pair {pair_index} text DPO rejected_response equals chosen response")
    if first.get("rejected_images") != first.get("images"):
        raise ValueError(f"Pair {pair_index} text DPO rejected_images must equal images")
    if second.get("rejected_response") != assistant_content:
        raise ValueError(f"Pair {pair_index} image DPO rejected_response must equal chosen response")
    if second.get("rejected_images") == second.get("images"):
        raise ValueError(f"Pair {pair_index} image DPO rejected_images must differ from images")

    score = first["similarity_score"]
    if not isinstance(score, int | float):
        raise ValueError(f"Pair {pair_index} has non-numeric similarity_score: {score!r}")
    return float(score)


def group_pairs(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(data) % 2 != 0:
        raise ValueError(f"DPO data must contain an even number of samples, got {len(data)}")

    pairs = []
    for index in range(0, len(data), 2):
        pair_index = index // 2
        first = data[index]
        second = data[index + 1]
        score = validate_dpo_pair(first, second, pair_index)
        pairs.append(
            {
                "index": pair_index,
                "score": score,
                "question_type": first.get("question_type", "unknown"),
                "samples": [first, second],
            }
        )
    return pairs


def select_low_similarity_pairs(
    pairs: list[dict[str, Any]],
    keep_percent: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remove_percent = 100 - keep_percent
    remove_count = int(len(pairs) * remove_percent / 100)
    keep_count = len(pairs) - remove_count

    sorted_pairs = sorted(pairs, key=lambda pair: (pair["score"], pair["index"]))
    selected_pairs = sorted_pairs[:keep_count]
    removed_pairs = sorted_pairs[keep_count:]
    return selected_pairs, removed_pairs


def flatten_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for pair in pairs:
        flattened.extend(pair["samples"])
    return flattened


def format_percent(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def default_output_file(input_file: Path, output_dir: Path, keep_percent: float) -> Path:
    return output_dir / f"{input_file.stem}_keep{format_percent(keep_percent)}.json"


def print_stats(label: str, pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        print(f"{label}: 0 pairs")
        return

    scores = [pair["score"] for pair in pairs]
    question_types = Counter(pair["question_type"] for pair in pairs)
    qtype_summary = ", ".join(f"{name}={count}" for name, count in sorted(question_types.items()))

    print(f"{label}: {len(pairs)} pairs / {len(pairs) * 2} DPO samples")
    print(f"  score range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"  mean score: {sum(scores) / len(scores):.2f}")
    print(f"  question types: {qtype_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the lowest-similarity counterfactual pairs. This implements the "
            "ChartCF data selection strategy: rank pairs by image similarity and keep "
            "the rho percent with the lowest scores."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/ECD/syn_data/dpo_data.json"),
        help="Input paired DPO JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ECD/syn_data"),
        help="Output directory used when --output-file is not set.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <output-dir>/<input-stem>_keep<percent>.json.",
    )
    parser.add_argument(
        "--keep-percent",
        type=float,
        default=40,
        help="Percent of lowest-similarity pairs to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.keep_percent < 100:
        raise ValueError("--keep-percent must be between 0 and 100")
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print(f"Reading DPO data from {args.input_file}")
    data = read_json(args.input_file)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON list")

    pairs = group_pairs(data)
    selected_pairs, removed_pairs = select_low_similarity_pairs(pairs, args.keep_percent)
    selected_data = flatten_pairs(selected_pairs)

    output_file = args.output_file or default_output_file(
        args.input_file,
        args.output_dir,
        args.keep_percent,
    )
    write_json(output_file, selected_data)

    print(f"Total: {len(pairs)} pairs / {len(data)} DPO samples")
    print_stats(f"Selected lowest {args.keep_percent:g}%", selected_pairs)
    print_stats(f"Filtered out highest {100 - args.keep_percent:g}%", removed_pairs)
    print(f"Saved selected data to {output_file}")


if __name__ == "__main__":
    main()
