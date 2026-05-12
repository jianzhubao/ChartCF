#!/usr/bin/env python3
"""Merge generated counterfactual data into paired DPO samples."""

import argparse
import json
import re
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


def extract_base_image_id(image_path: str) -> str:
    filename = Path(image_path).name
    match = re.match(r"(\d{6})", filename)
    if match:
        return match.group(1)
    return Path(filename).stem.split("_")[0]


def get_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError(f"Sample has no assistant message: {messages}")


def load_similarity_score(similarity_dir: Path, item_id: str) -> int | float | None:
    similarity_file = similarity_dir / f"{item_id}_similarity.json"
    if not similarity_file.exists():
        return None

    similarity_data = read_json(similarity_file)
    return similarity_data.get("score")


def build_messages(qa_data: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"<image>{qa_data['original_question']}",
        },
        {
            "role": "assistant",
            "content": qa_data["original_answer"],
        },
    ]


def build_dpo_pair(
    qa_data: dict[str, Any],
    original_image_path: Path,
    rejected_image_path: Path,
    question_type: str,
    similarity_score: int | float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = build_messages(qa_data)
    original_image = str(original_image_path)
    rejected_image = str(rejected_image_path)
    assistant_content = qa_data["original_answer"]

    shared_fields: dict[str, Any] = {
        "messages": messages,
        "images": [original_image],
        "question_type": question_type,
    }
    if similarity_score is not None:
        shared_fields["similarity_score"] = similarity_score

    text_dpo_entry = {
        **shared_fields,
        "rejected_response": qa_data["new_answer"],
        "rejected_images": [original_image],
    }
    image_dpo_entry = {
        **shared_fields,
        "rejected_response": assistant_content,
        "rejected_images": [rejected_image],
    }
    return text_dpo_entry, image_dpo_entry


def load_split_pairs(
    split_dir: Path,
    original_images_dir: Path,
    question_type: str,
) -> tuple[list[tuple[str, str, dict[str, Any], dict[str, Any]]], Counter[str]]:
    qa_dir = split_dir / "qa_pairs"
    modified_images_dir = split_dir / "images"
    similarity_dir = split_dir / "similarity_results"
    skipped: Counter[str] = Counter()
    pairs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

    if not qa_dir.exists():
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")
    if not modified_images_dir.exists():
        raise FileNotFoundError(f"Modified image directory not found: {modified_images_dir}")

    qa_files = sorted(qa_dir.glob("*.json"))
    for qa_file in qa_files:
        if qa_file.name.startswith("failed_"):
            skipped["failed_qa_file"] += 1
            continue

        qa_data = read_json(qa_file)
        item_id = str(qa_data.get("item_id") or qa_file.stem)
        image_id = str(qa_data["image_id"])

        if qa_data.get("new_answer") is None:
            skipped["missing_new_answer"] += 1
            continue

        original_image_path = original_images_dir / f"{image_id}.png"
        rejected_image_path = modified_images_dir / f"{item_id}.png"

        if not original_image_path.exists():
            skipped["missing_original_image"] += 1
            continue
        if not rejected_image_path.exists():
            skipped["missing_rejected_image"] += 1
            continue

        similarity_score = load_similarity_score(similarity_dir, item_id)
        text_entry, image_entry = build_dpo_pair(
            qa_data=qa_data,
            original_image_path=original_image_path,
            rejected_image_path=rejected_image_path,
            question_type=question_type,
            similarity_score=similarity_score,
        )
        pairs.append((image_id, item_id, text_entry, image_entry))

    return pairs, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ChartCF generated QA, rendered images, and similarity scores into DPO data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ECD"),
        help="Root directory for ECD data.",
    )
    parser.add_argument(
        "--syn-dir",
        type=Path,
        default=Path("syn_data"),
        help="Synthetic data directory relative to --data-root.",
    )
    parser.add_argument(
        "--original-images-dir",
        type=Path,
        default=Path("rendered_images_with_seed"),
        help="Original image directory relative to --data-root.",
    )
    parser.add_argument(
        "--split-types",
        nargs="+",
        default=["descriptive", "reasoning"],
        help="Synthetic split names to merge.",
    )
    parser.add_argument(
        "--post-suffix",
        default="_post",
        help="Suffix appended to each split directory after post processing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ECD/syn_data/dpo_data.json"),
        help="Output DPO JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    syn_root = data_root / args.syn_dir
    original_images_dir = data_root / args.original_images_dir

    if not original_images_dir.exists():
        raise FileNotFoundError(f"Original image directory not found: {original_images_dir}")

    all_pairs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    total_skipped: Counter[str] = Counter()

    for split_type in args.split_types:
        split_dir = syn_root / f"{split_type}{args.post_suffix}"
        print(f"Loading {split_type} data from {split_dir}")
        pairs, skipped = load_split_pairs(
            split_dir=split_dir,
            original_images_dir=original_images_dir,
            question_type=split_type,
        )
        all_pairs.extend(pairs)
        total_skipped.update(skipped)
        print(f"  Loaded {len(pairs)} QA pairs")
        if skipped:
            skipped_msg = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
            print(f"  Skipped {sum(skipped.values())}: {skipped_msg}")

    all_pairs.sort(key=lambda item: (extract_base_image_id(item[2]["images"][0]), item[1]))

    dpo_data: list[dict[str, Any]] = []
    for _, _, text_entry, image_entry in all_pairs:
        dpo_data.append(text_entry)
        dpo_data.append(image_entry)

    if not dpo_data:
        raise ValueError("No DPO samples were generated.")

    write_json(args.output, dpo_data)

    text_count = len(dpo_data) // 2
    print(f"Saved {len(dpo_data)} DPO samples ({text_count} text/image pairs) to {args.output}")
    if total_skipped:
        skipped_msg = ", ".join(f"{key}={value}" for key, value in sorted(total_skipped.items()))
        print(f"Total skipped {sum(total_skipped.values())}: {skipped_msg}")


if __name__ == "__main__":
    main()
