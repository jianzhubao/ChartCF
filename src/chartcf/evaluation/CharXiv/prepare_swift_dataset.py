"""
Prepare CharXiv dataset in SWIFT jsonl format for inference

This script converts CharXiv data to SWIFT's expected format:
{
    "messages": [
        {"role": "user", "content": [{"type": "image", "image": "<image_path>"}, {"type": "text", "text": "<question>"}]},
        {"role": "assistant", "content": "<answer>"}  # Optional, for evaluation
    ]
}
"""

import os
import json
import argparse
from eval_utils.descriptive_utils import build_descriptive_quries
from eval_utils.reasoning_utils import build_reasoning_queries


def convert_to_swift_format(queries, mode):
    """
    Convert queries to SWIFT format (matching training data format)

    Training data format:
    {
        "messages": [
            {"role": "user", "content": "<image>\\n<question>"},
            {"role": "assistant", "content": "<answer>"}  # Not needed for inference
        ],
        "images": ["<image_path>"],
        "id": "<id>"
    }

    Args:
        queries: Dict with query data (keys, questions, image paths, etc.)
        mode: 'descriptive' or 'reasoning'

    Returns:
        List of dicts in SWIFT format
    """
    swift_data = []

    for key, value in queries.items():
        # Create message in SWIFT format (same as training data)
        # Use <image> placeholder in content, and provide actual path in images field
        messages = [
            {
                "role": "user",
                "content": f"<image>{value['question']}"
            }
        ]

        # Create data item with metadata for later matching
        data_item = {
            "messages": messages,
            "images": [value['figure_path']],  # Images as a separate field
            "id": key,  # Use key as id
            # Keep metadata for result matching
            "_figure_id": value['figure_id'],
        }

        # Add mode-specific metadata
        if mode == 'descriptive':
            data_item.update({
                '_subq_idx': value['subq_idx'],
                '_qid': value['qid'],
            })
        else:  # reasoning mode
            data_item.update({
                '_inst_category': value['inst_category'],
                '_raw_question': value['raw_question'],
            })

        swift_data.append(data_item)

    return swift_data


def main():
    parser = argparse.ArgumentParser(description='Prepare CharXiv dataset for SWIFT inference')
    parser.add_argument('--split', type=str, required=True, choices=['val', 'test'],
                        help='Dataset split')
    parser.add_argument('--mode', type=str, required=True, choices=['descriptive', 'reasoning'],
                        help='Evaluation mode')
    parser.add_argument('--output_dir', type=str, default='data/public_benchmarks/CharXiv/swift_data',
                        help='Output directory for SWIFT format data')
    parser.add_argument('--data_dir', type=str, default='data/public_benchmarks/CharXiv/data',
                        help='Directory containing CharXiv annotation JSON files')
    parser.add_argument('--image_dir', type=str, default='data/public_benchmarks/CharXiv/images',
                        help='Directory containing CharXiv images')

    args = parser.parse_args()

    # Paths
    image_dir = args.image_dir
    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.mode}_{args.split}.json")

    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # Build queries using existing utilities
    if args.mode == 'descriptive':
        queries = build_descriptive_quries(data, image_dir)
    elif args.mode == 'reasoning':
        queries = build_reasoning_queries(data, image_dir)
    else:
        raise ValueError("Mode not supported")

    print(f"Total samples: {len(queries)}")

    # Convert to SWIFT format
    swift_data = convert_to_swift_format(queries, args.mode)

    # Write to JSON file (as a list)
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(swift_data, f, ensure_ascii=False, indent=2)

    print(f"Done! Data saved to {output_file}")
    print(f"Total samples written: {len(swift_data)}")


if __name__ == "__main__":
    main()
