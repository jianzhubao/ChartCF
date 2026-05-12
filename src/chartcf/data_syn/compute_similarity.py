"""
Image Similarity Computation - Compare original and modified chart images using GPT-4o-mini.

This script compares pairs of chart images (original vs modified) using OpenAI's GPT-4o-mini model
to assess their similarity based on various criteria like chart types, layout, text content, data,
style, and overall visual coherence.
"""

import os
import json
import argparse
import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


# Lock for thread-safe file operations
file_lock = Lock()


# Error type constants
ERROR_TYPE_API = "API_ERROR"
ERROR_TYPE_PARSE = "PARSE_ERROR"
ERROR_TYPE_NOT_FOUND = "NOT_FOUND"
ERROR_TYPE_UNKNOWN = "UNKNOWN_ERROR"


def extract_image_id(filename: str) -> str:
    """
    Extract the base ID from a modified image filename.

    Example: '000001_000000.png' -> '000001'

    Args:
        filename: The filename of the modified image

    Returns:
        The extracted base ID
    """
    # Remove extension and split by underscore
    base_name = Path(filename).stem
    image_id = base_name.split('_')[0]
    return image_id


def find_original_image(image_id: str, original_dir: Path) -> Optional[Path]:
    """
    Find the original image corresponding to the given ID.

    Args:
        image_id: The ID to search for (e.g., '000001')
        original_dir: Directory containing original images

    Returns:
        Path to the original image if found, None otherwise
    """
    original_path = original_dir / f"{image_id}.png"
    if original_path.exists():
        return original_path
    return None


def load_prompt_template(template_path: Path) -> str:
    """Load the evaluation prompt template."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt_api(
    client: OpenAI,
    prompt: str,
    original_image_path: Path,
    modified_image_path: Path,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Call OpenAI GPT API to evaluate image similarity.

    Args:
        client: OpenAI client instance
        prompt: Evaluation prompt
        original_image_path: Path to the original image
        modified_image_path: Path to the modified image
        model: GPT model to use

    Returns:
        GPT response text
    """
    try:
        # Encode both images
        original_base64 = encode_image_to_base64(original_image_path)
        modified_base64 = encode_image_to_base64(modified_image_path)

        # Prepare message content with both images
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{modified_base64}"
                }
            }
        ]

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Error calling GPT API: {str(e)}") from e


def parse_similarity_score(response_text: str) -> Tuple[Optional[int], Dict[str, str]]:
    """
    Parse the similarity score and comments from GPT response.

    Args:
        response_text: Raw response from GPT

    Returns:
        Tuple of (score, comments_dict) where score is the final score (0-100)
        and comments_dict contains detailed comments for each criterion
    """
    # Extract final score (avoid matching "Subscore:")
    # Look for "Score: X/100" or "Score: X" at the end of the response
    score_match = re.search(r'(?<!\w)Score:\s*(\d+)(?:/100)?', response_text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    # Extract individual comments
    # Updated to match the actual 5 criteria from prompt_v1.md
    comments = {}
    criteria = [
        "Chart Types",
        "Layout",
        "Text Content",
        "Data",
        "Style"
    ]

    for criterion in criteria:
        # Match pattern: "- Criterion: comment text. Subscore: X/Y"
        # Using a more flexible pattern that handles multi-line comments
        pattern = rf'-\s*{re.escape(criterion)}:\s*(.+?)(?=\n-\s*[A-Z]|\nScore:|\Z)'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            comment = match.group(1).strip()
            # Clean up any extra whitespace or newlines
            comment = re.sub(r'\s+', ' ', comment)
            comments[criterion] = comment

    return score, comments


def check_if_processed(modified_image_name: str, output_dir: Path) -> bool:
    """
    Check if an image pair has already been processed.

    Args:
        modified_image_name: Name of the modified image
        output_dir: Directory where results are stored

    Returns:
        True if already processed, False otherwise
    """
    result_filename = Path(modified_image_name).stem + "_similarity.json"
    result_path = output_dir / result_filename
    return result_path.exists()


def save_result(
    modified_image_name: str,
    original_image_name: str,
    score: Optional[int],
    comments: Dict[str, str],
    raw_response: str,
    output_dir: Path,
    error: Optional[str] = None,
    error_type: Optional[str] = None
):
    """
    Save the similarity evaluation result to a JSON file.

    Args:
        modified_image_name: Name of the modified image
        original_image_name: Name of the original image
        score: Similarity score (0-100)
        comments: Detailed comments for each criterion
        raw_response: Raw GPT response
        output_dir: Directory to save the result
        error: Error message if any
        error_type: Type of error if any (API_ERROR, PARSE_ERROR, NOT_FOUND, UNKNOWN_ERROR)
    """
    result = {
        "modified_image": modified_image_name,
        "original_image": original_image_name,
        "score": score,
        "comments": comments,
        "raw_response": raw_response,
        "error": error,
        "error_type": error_type
    }

    # Determine filename based on whether there's an error
    stem = Path(modified_image_name).stem
    if error:
        result_filename = f"{stem}_similarity_failed.json"
    else:
        result_filename = f"{stem}_similarity.json"

    result_path = output_dir / result_filename

    # Thread-safe file write
    with file_lock:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # If saving a successful result, remove any existing failed result
        if not error:
            failed_path = output_dir / f"{stem}_similarity_failed.json"
            if failed_path.exists():
                failed_path.unlink()


def find_failed_samples(output_dir: Path) -> List[str]:
    """
    Find all failed samples in the output directory.

    Args:
        output_dir: Directory where results are stored

    Returns:
        List of modified image names (with .png extension) that have failed
    """
    if not output_dir.exists():
        return []

    failed_files = list(output_dir.glob("*_similarity_failed.json"))
    failed_image_names = []

    for failed_file in failed_files:
        # Extract the stem (e.g., "000001_000000" from "000001_000000_similarity_failed.json")
        stem = failed_file.stem.replace("_similarity_failed", "")
        # Add .png extension to get the modified image name
        image_name = f"{stem}.png"
        failed_image_names.append(image_name)

    return failed_image_names


def process_image_pair(
    modified_image_path: Path,
    original_dir: Path,
    output_dir: Path,
    prompt: str,
    client: OpenAI,
    model: str,
    skip_existing: bool = True
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image pair and compute similarity.

    Args:
        modified_image_path: Path to the modified image
        original_dir: Directory containing original images
        output_dir: Directory to save results
        prompt: Evaluation prompt
        client: OpenAI client
        model: GPT model to use
        skip_existing: Whether to skip already processed images

    Returns:
        Tuple of (image_name, success, error_message)
    """
    modified_image_name = modified_image_path.name

    # Check if already processed
    if skip_existing and check_if_processed(modified_image_name, output_dir):
        return (modified_image_name, True, None)

    try:
        # Extract ID and find original image
        image_id = extract_image_id(modified_image_name)
        original_image_path = find_original_image(image_id, original_dir)

        if original_image_path is None:
            error_msg = f"Original image not found for ID: {image_id}"
            save_result(
                modified_image_name,
                f"{image_id}.png",
                None,
                {},
                "",
                output_dir,
                error=error_msg,
                error_type=ERROR_TYPE_NOT_FOUND
            )
            return (modified_image_name, False, error_msg)

        # Call GPT API
        response_text = call_gpt_api(
            client,
            prompt,
            original_image_path,
            modified_image_path,
            model
        )

        # Parse response
        score, comments = parse_similarity_score(response_text)

        # Check if parsing was successful
        if score is None:
            # Parsing failed, save as failed sample
            error_msg = "Failed to parse similarity score from GPT response"
            save_result(
                modified_image_name,
                original_image_path.name,
                None,
                comments,
                response_text,
                output_dir,
                error=error_msg,
                error_type=ERROR_TYPE_PARSE
            )
            return (modified_image_name, False, error_msg)

        # Parsing successful, save result
        save_result(
            modified_image_name,
            original_image_path.name,
            score,
            comments,
            response_text,
            output_dir
        )

        return (modified_image_name, True, None)

    except RuntimeError as e:
        # API call failed
        error_msg = str(e)
        save_result(
            modified_image_name,
            "",
            None,
            {},
            "",
            output_dir,
            error=error_msg,
            error_type=ERROR_TYPE_API
        )
        return (modified_image_name, False, error_msg)

    except Exception as e:
        # Other unknown errors
        error_msg = str(e)
        save_result(
            modified_image_name,
            "",
            None,
            {},
            "",
            output_dir,
            error=error_msg,
            error_type=ERROR_TYPE_UNKNOWN
        )
        return (modified_image_name, False, error_msg)


def process_directory(
    modified_dir: Path,
    original_dir: Path,
    output_dir: Path,
    prompt_template_path: Path,
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_workers: int,
    skip_existing: bool = True,
    limit: Optional[int] = None
):
    """
    Process all images in a directory.

    Args:
        modified_dir: Directory containing modified images
        original_dir: Directory containing original images
        output_dir: Directory to save results
        prompt_template_path: Path to the prompt template
        api_key: OpenAI API key
        base_url: Optional custom API base URL
        model: GPT model to use
        max_workers: Number of parallel workers
        skip_existing: Whether to skip already processed images
        limit: Maximum number of images to process (None = process all)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt = load_prompt_template(prompt_template_path)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    # Get list of modified images
    all_modified_images = sorted(modified_dir.glob("*.png"))

    if not all_modified_images:
        print(f"No images found in {modified_dir}")
        return

    # Find failed samples that need to be retried
    failed_sample_names = find_failed_samples(output_dir)
    failed_sample_set = set(failed_sample_names)

    # Build a mapping from image name to path for quick lookup
    image_name_to_path = {img.name: img for img in all_modified_images}

    # Separate retry images and new images
    retry_images = []
    new_images = []

    for img_path in all_modified_images:
        img_name = img_path.name
        if img_name in failed_sample_set:
            # This is a failed sample that needs retry
            retry_images.append(img_path)
        elif not check_if_processed(img_name, output_dir):
            # This is a new image that hasn't been processed
            new_images.append(img_path)
        # else: already successfully processed, skip

    # Apply limit: retry images first, then new images
    if limit is not None and limit > 0:
        # Take retry images up to limit
        images_to_process = retry_images[:limit]
        remaining_limit = limit - len(images_to_process)

        # Add new images if there's remaining limit
        if remaining_limit > 0:
            images_to_process.extend(new_images[:remaining_limit])

        print(f"Found {len(retry_images)} failed samples to retry and {len(new_images)} new images")
        print(f"Processing {len(images_to_process)} images (limited to {limit}) in {modified_dir.name}")
    else:
        # No limit, process all retry and new images
        images_to_process = retry_images + new_images
        print(f"Found {len(retry_images)} failed samples to retry and {len(new_images)} new images")
        print(f"Processing {len(images_to_process)} images in total in {modified_dir.name}")

    if not images_to_process:
        print(f"No images to process in {modified_dir}")
        return

    modified_images = images_to_process

    # Process images in parallel
    success_count = 0
    error_count = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_image_pair,
                img_path,
                original_dir,
                output_dir,
                prompt,
                client,
                model,
                skip_existing
            ): img_path
            for img_path in modified_images
        }

        # Process results with progress bar
        with tqdm(total=len(modified_images), desc=f"Processing {modified_dir.name}") as pbar:
            for future in as_completed(futures):
                img_name, success, error = future.result()

                if success:
                    if skip_existing and check_if_processed(img_name, output_dir):
                        skipped_count += 1
                    else:
                        success_count += 1
                else:
                    error_count += 1
                    if error:
                        tqdm.write(f"Error processing {img_name}: {error}")

                pbar.update(1)

    print(f"\nResults for {modified_dir.name}:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarity between original and modified chart images using GPT-4o-mini"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/ECD",
        help="Root directory for data"
    )
    parser.add_argument(
        "--original-dir",
        type=str,
        default="rendered_images_with_seed",
        help="Directory name for original images (relative to data-root)"
    )
    parser.add_argument(
        "--modified-dirs",
        type=str,
        nargs="+",
        default=[
            "syn_data/descriptive_post/images",
            "syn_data/reasoning_post/images"
        ],
        help="Directory names for modified images (relative to data-root)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="similarity_results",
        help="Suffix for output directory name"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="prompts/prompt_image_similarity_eval.md",
        help="Path to the prompt template"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="GPT model to use"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not set, will read from OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL (optional)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess all images even if results already exist"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process per directory (for testing, default: process all)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or use --api-key")

    # Convert paths to Path objects
    data_root = Path(args.data_root)
    original_dir = data_root / args.original_dir
    prompt_template_path = Path(args.prompt_template)

    # Check if paths exist
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not original_dir.exists():
        raise FileNotFoundError(f"Original images directory not found: {original_dir}")
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")

    # Process each modified directory
    for modified_dir_name in args.modified_dirs:
        modified_dir = data_root / modified_dir_name

        if not modified_dir.exists():
            print(f"Warning: Modified directory not found: {modified_dir}")
            continue

        # Determine output directory (same parent as modified_dir)
        output_dir = modified_dir.parent / args.output_suffix

        print(f"\n{'='*80}")
        print(f"Processing directory: {modified_dir}")
        print(f"Original images: {original_dir}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")

        process_directory(
            modified_dir=modified_dir,
            original_dir=original_dir,
            output_dir=output_dir,
            prompt_template_path=prompt_template_path,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            max_workers=args.max_workers,
            skip_existing=not args.no_skip_existing,
            limit=args.limit
        )

    print("\n" + "="*80)
    print("All directories processed!")
    print("="*80)


if __name__ == "__main__":
    main()
