"""
Prepare ChartX dataset in SWIFT json format for inference

This script converts ChartX data to SWIFT's expected format:
{
    "messages": [
        {"role": "user", "content": "<image><question>"}
    ],
    "images": ["<image_path>"],
    "id": "<id>"
}
"""

import os
import json
import argparse


def convert_to_swift_format(data, image_dir):
    """
    Convert ChartX data to SWIFT format

    Args:
        data: List of ChartX data items
        image_dir: Base directory for images

    Returns:
        List of dicts in SWIFT format
    """
    swift_data = []

    for idx, item in enumerate(data):
        # img field is like "./bar_chart/png/bar_87.png", remove leading "./"
        img_path = item['img']
        if img_path.startswith('./'):
            img_path = img_path[2:]
        image_path = os.path.join(image_dir, img_path)

        # Create message in SWIFT format
        messages = [
            {
                "role": "user",
                "content": f"<image>{item['QA']['input']}"
            }
        ]

        # Create data item with metadata for later matching
        data_item = {
            "messages": messages,
            "images": [image_path],
            "id": str(idx),
            # Keep metadata for result matching and evaluation
            "_imgname": item['imgname'],
            "_img": item['img'],
            "_label": item['QA']['output'],
            "_chart_type": item['chart_type'],
            "_topic": item.get('topic', ''),
            "_title": item.get('title', ''),
        }

        swift_data.append(data_item)

    return swift_data


def main():
    parser = argparse.ArgumentParser(description='Prepare ChartX dataset for SWIFT inference')
    parser.add_argument('--input_file', type=str,
                        default='data/public_benchmarks/ChartX/ChartX_annotation_test.json',
                        help='Input data file')
    parser.add_argument('--image_dir', type=str,
                        default='data/public_benchmarks/ChartX/ChartX_png',
                        help='Base directory for images')
    parser.add_argument('--output_dir', type=str,
                        default='data/public_benchmarks/ChartX/swift_data',
                        help='Output directory for SWIFT format data')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'test.json')

    print(f"Reading {args.input_file}...")
    with open(args.input_file) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Convert to SWIFT format
    swift_data = convert_to_swift_format(data, args.image_dir)

    # Write to JSON file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(swift_data, f, ensure_ascii=False, indent=2)

    print(f"Done! Data saved to {output_file}")
    print(f"Total samples written: {len(swift_data)}")


if __name__ == "__main__":
    main()
