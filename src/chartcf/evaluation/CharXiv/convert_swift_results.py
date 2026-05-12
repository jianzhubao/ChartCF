"""
Convert SWIFT inference results back to CharXiv evaluation format

SWIFT output format (jsonl):
{
    "messages": [...],
    "response": "<model_response>",
    ...other fields from input
}

CharXiv expected format (json):
{
    "key": {
        "figure_id": ...,
        "response": ...,
        "subq_idx": ...,  # for descriptive
        "qid": ...,       # for descriptive
        "inst_category": ...,  # for reasoning
        "raw_question": ...    # for reasoning
    }
}
"""

import json
import argparse


def get_user_content(item):
    messages = item.get('messages', [])
    for message in messages:
        if message.get('role') == 'user':
            return message.get('content', '')
    return ''


def get_image_path(item):
    images = item.get('images', [])
    if not images:
        return ''
    image = images[0]
    if isinstance(image, dict):
        return image.get('path', '')
    return image


def build_dataset_lookup(dataset):
    lookup = {}
    for item in dataset:
        lookup[(get_image_path(item), get_user_content(item))] = item
    return lookup


def convert_swift_to_charxiv(swift_output_file, dataset_file, mode):
    """
    Convert SWIFT inference results to CharXiv format

    Args:
        swift_output_file: Path to SWIFT output file (one JSON object per line)
        dataset_file: Path to original dataset JSON file (contains metadata)
        mode: 'descriptive' or 'reasoning'

    Returns:
        Dict in CharXiv format
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

    dataset_lookup = build_dataset_lookup(dataset)

    # Verify lengths match
    if len(swift_results) != len(dataset):
        print(f"WARNING: Result count ({len(swift_results)}) != Dataset count ({len(dataset)})")

    # Convert to CharXiv format
    charxiv_results = {}

    for idx, swift_result in enumerate(swift_results):
        lookup_key = (get_image_path(swift_result), get_user_content(swift_result))
        data_item = dataset_lookup.get(lookup_key)
        if data_item is None:
            if idx >= len(dataset):
                raise ValueError(f"Cannot match SWIFT result at index {idx}: {lookup_key}")
            data_item = dataset[idx]
            print(f"WARNING: Falling back to positional metadata for result {idx}")

        # Extract key and response (metadata fields start with underscore)
        key = data_item['id']
        response = swift_result.get('response', '')

        # Build result entry
        result = {
            'figure_id': data_item['_figure_id'],
            'response': response,
        }

        # Add mode-specific fields
        if mode == 'descriptive':
            result.update({
                'subq_idx': data_item['_subq_idx'],
                'qid': data_item['_qid'],
            })
        else:  # reasoning mode
            result.update({
                'inst_category': data_item['_inst_category'],
                'raw_question': data_item['_raw_question'],
            })

        charxiv_results[key] = result

    return charxiv_results


def main():
    parser = argparse.ArgumentParser(description='Convert SWIFT results to CharXiv format')
    parser.add_argument('--swift_output', type=str, required=True,
                        help='Path to SWIFT output JSON file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to original dataset JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--mode', type=str, required=True, choices=['descriptive', 'reasoning'],
                        help='Evaluation mode')

    args = parser.parse_args()

    print(f"Reading SWIFT output from: {args.swift_output}")
    print(f"Reading dataset from: {args.dataset}")

    # Convert results
    charxiv_results = convert_swift_to_charxiv(
        args.swift_output,
        args.dataset,
        args.mode
    )

    print(f"Total results: {len(charxiv_results)}")

    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(charxiv_results, f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
