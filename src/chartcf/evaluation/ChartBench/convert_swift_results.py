"""
Convert SWIFT inference results back to ChartBench evaluation format

SWIFT output format (jsonl):
{
    "messages": [...],
    "response": "<model_response>",
    ...other fields from input
}

ChartBench expected format (json list):
[
    {
        "id": ...,
        "type": {...},
        "question": "...",
        "gt_answer": "...",
        "pred_answer": "...",
        "image": "..."
    }
]
"""

import json
import argparse


def convert_swift_to_chartbench(swift_output_file, dataset_file):
    """
    Convert SWIFT inference results to ChartBench format

    Args:
        swift_output_file: Path to SWIFT output file (one JSON object per line)
        dataset_file: Path to original dataset JSON file (contains metadata)

    Returns:
        List of dicts in ChartBench evaluation format
    """
    # Read SWIFT output (one JSON object per line)
    swift_results = []
    with open(swift_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                swift_results.append(json.loads(line))

    # Read original dataset to get metadata (JSON list)
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Verify lengths match
    if len(swift_results) != len(dataset):
        print(f"WARNING: Result count ({len(swift_results)}) != Dataset count ({len(dataset)})")

    # Convert to ChartBench evaluation format
    chartbench_results = []

    for swift_result, data_item in zip(swift_results, dataset):
        # Extract response and metadata
        response = swift_result.get('response', '')

        # Get question from messages
        question = data_item['messages'][0]['content']
        # Remove <image> prefix if present
        if question.startswith('<image>'):
            question = question[len('<image>'):]

        result = {
            'id': data_item['_original_id'],
            'type': data_item['_type'],
            'question': question,
            'gt_answer': data_item['_label'],
            'pred_answer': response,
            'image': data_item['_image'],
        }

        chartbench_results.append(result)

    return chartbench_results


def main():
    parser = argparse.ArgumentParser(description='Convert SWIFT results to ChartBench format')
    parser.add_argument('--swift_output', type=str, required=True,
                        help='Path to SWIFT output JSONL file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to original dataset JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')

    args = parser.parse_args()

    print(f"Reading SWIFT output from: {args.swift_output}")
    print(f"Reading dataset from: {args.dataset}")

    # Convert results
    chartbench_results = convert_swift_to_chartbench(
        args.swift_output,
        args.dataset
    )

    print(f"Total results: {len(chartbench_results)}")

    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chartbench_results, f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
