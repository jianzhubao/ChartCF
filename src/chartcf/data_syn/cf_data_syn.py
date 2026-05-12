"""
Modify chart code to produce different answers to questions.

This script reads existing chart codes and QA pairs, uses OpenAI API to modify the code
so that the answer to the question becomes different, and saves the results.
"""

import os
import json
import argparse
import base64
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from jinja2 import Template
from dotenv import load_dotenv
from tqdm import tqdm


def parse_reasoning_answer(answer: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse reasoning-type answer from "xxx Answer: xxx" format.

    Args:
        answer: Answer string in format "xxx Answer: xxx"

    Returns:
        tuple: (reasoning_process, answer) or (None, None) if parsing fails
    """
    # Find the last occurrence of "Answer:" to handle cases where it might appear in reasoning
    match = re.search(r'^(.*?)\s*Answer:\s*(.*)$', answer, re.DOTALL)
    if match:
        reasoning_process = match.group(1).strip()
        answer_text = match.group(2).strip()
        return reasoning_process, answer_text
    return None, None


def format_reasoning_answer(new_answer_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert LLM output from "Reasoning Process: xxx\\nAnswer: yyy" back to "xxx Answer: yyy" format.

    Args:
        new_answer_text: Text from LLM containing "Reasoning Process:" and "Answer:" sections

    Returns:
        tuple: (formatted_answer, error_message)
            - formatted_answer: "xxx Answer: yyy" format or None if parsing fails
            - error_message: Error description if parsing fails, None otherwise
    """
    # Extract "Reasoning Process:" and "Answer:" sections
    reasoning_match = re.search(r'Reasoning Process:\s*(.*?)(?=\nAnswer:|\Z)', new_answer_text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'Answer:\s*(.*?)(?=\n\*\*|\Z)', new_answer_text, re.DOTALL | re.IGNORECASE)

    if not reasoning_match or not answer_match:
        return None, "Failed to parse reasoning answer: missing 'Reasoning Process:' or 'Answer:' section"

    reasoning_process = reasoning_match.group(1).strip()
    answer_text = answer_match.group(1).strip()

    # Remove common LLM artifacts like "None" for unfeasible cases
    if reasoning_process.lower() == "none" or answer_text.lower() == "none":
        return None, "Reasoning or answer is 'None' (task marked as infeasible)"

    # Combine back to original format: "xxx Answer: xxx"
    formatted_answer = f"{reasoning_process} Answer: {answer_text}"
    return formatted_answer, None


def clean_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting from text to produce plain text.

    This function removes common markdown syntax like **bold**, *italic*, `code`, etc.
    while being careful not to affect normal text content.

    Args:
        text: Text with potential markdown formatting

    Returns:
        Plain text without markdown formatting
    """
    if not text:
        return text

    # Remove bold: **text** or __text__ -> text
    # Use non-greedy match and word boundaries to avoid affecting expressions like 2**3
    text = re.sub(r'\*\*([^\*]+?)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+?)__', r'\1', text)

    return text


def load_prompt_template(template_path: str) -> Template:
    """Load the Jinja2 prompt template."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)


def load_qa_data(qa_path: str) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(qa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_code(code_path: str) -> str:
    """Load Python code from file."""
    with open(code_path, 'r', encoding='utf-8') as f:
        return f.read()


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_openai_api(
    client: OpenAI,
    prompt: str,
    image_path: Optional[str] = None,
    model: str = "gpt-5-2025-08-07"
) -> str:
    """Call OpenAI API with the given prompt and optional image."""
    try:
        # Prepare message content
        content = [{"type": "text", "text": prompt}]

        # Add image if provided
        if image_path:
            base64_image = encode_image_to_base64(image_path)

            # Determine MIME type based on file extension
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'
            }.get(ext, 'image/png')

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            # reasoning_effort="high",
            # reasoning={"effort": "none"},
            # temperature=0.7,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def _try_match_with_fallback(response: str, patterns: List[str], flags: int = 0):
    """
    Try multiple regex patterns in order to extract information, allowing for fallback if the primary pattern fails.
    
    Args:
        response: The LLM response string to search within.
        patterns: A list of regex patterns to try, ordered by priority.
        flags: Optional regex flags (e.g., re.DOTALL) to apply to all patterns.
        
    Returns:
        The first successful regex match object, or None if all patterns fail.
    """
    for pattern in patterns:
        match = re.search(pattern, response, flags)
        if match:
            return match
    return None


def parse_response(response: str) -> Dict[str, str]:
    """
    Parse the OpenAI response to extract:
    - Feasibility (YES/NO)
    - Rationale
    - Modified Code
    - New Answer

    Supports both bold (**Field:**) and non-bold (Field:) formats.
    """
    result = {
        "feasibility": None,
        "rationale": None,
        "modified_code": None,
        "new_answer": None
    }

    # Define fallback patterns for each field
    feasibility_patterns = [
        r'\*\*Feasibility:\*\*\s*\[?(YES|NO)\]?',  # Original: **Feasibility:**
        r'(?:^|\n)Feasibility:\s*\[?(YES|NO)\]?',  # Fallback: Feasibility:
    ]

    rationale_patterns = [
        r'\*\*Rationale of Modification:\*\*\s*\n(.*?)(?=\*\*Modified Code:\*\*|\Z)',
        r'(?:^|\n)Rationale of Modification:\s*\n(.*?)(?=(?:\*\*)?Modified Code:|\Z)',
    ]

    code_patterns = [
        r'\*\*Modified Code:\*\*\s*\n```python\s*\n(.*?)```',
        r'(?:^|\n)Modified Code:\s*\n```python\s*\n(.*?)```',
    ]

    answer_patterns = [
        r'\*\*New Answer:\*\*\s*\n(.*?)(?=\n\*\*|\Z)',
        r'(?:^|\n)New Answer:\s*\n(.*?)(?=\n\*\*|\Z)',
    ]

    # Extract Feasibility
    feasibility_match = _try_match_with_fallback(response, feasibility_patterns, re.IGNORECASE)
    if feasibility_match:
        result["feasibility"] = feasibility_match.group(1).upper()

    # Extract Rationale
    rationale_match = _try_match_with_fallback(response, rationale_patterns, re.DOTALL)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()

    # Extract Modified Code
    code_match = _try_match_with_fallback(response, code_patterns, re.DOTALL)
    if code_match:
        result["modified_code"] = code_match.group(1).strip()

    # Extract New Answer
    answer_match = _try_match_with_fallback(response, answer_patterns, re.DOTALL)
    if answer_match:
        result["new_answer"] = answer_match.group(1).strip()

    return result


def execute_code_and_render(code: str, item_id: str, timeout: int = 30) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Execute Python code in a temporary directory and capture the generated image.

    Args:
        code: Python code to execute
        item_id: Item identifier for naming
        timeout: Maximum execution time in seconds

    Returns:
        tuple: (success, image_path, error_message)
            - success: True if execution succeeded and image was found
            - image_path: Path to the generated image (in temp dir)
            - error_message: Error message if failed
    """
    # Create a temporary directory for isolated execution
    temp_dir = Path(tempfile.mkdtemp(prefix=f"chart_render_{item_id}_"))

    try:
        # Create subdirectory for rendered images (many original codes save to rendered_images/)
        rendered_images_subdir = temp_dir / "rendered_images"
        rendered_images_subdir.mkdir(parents=True, exist_ok=True)

        # Write code to a temporary Python file
        code_file = temp_dir / "chart_code.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Set up environment with PYTHONHASHSEED for reproducibility
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = '42'
        env['MPLBACKEND'] = 'Agg'
        env['DISPLAY'] = ''

        # Execute the code in the temporary directory
        result = subprocess.run(
            ['python', str(code_file)],
            cwd=str(temp_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        # Check if execution was successful
        if result.returncode != 0:
            error_msg = f"Code execution failed with return code {result.returncode}: {result.stderr}"
            return False, None, error_msg

        # Look for the generated image in rendered_images subdirectory
        # Each code generates exactly one image: rendered_images/XXXXXX.png (6 digits)
        image_files = [
            f for f in rendered_images_subdir.glob("*.png")
            if re.match(r'^\d{6}\.png$', f.name)
        ]

        if not image_files:
            error_msg = "Code executed but no image file matching XXXXXX.png found in rendered_images/"
            return False, None, error_msg

        return True, image_files[0], None

    except subprocess.TimeoutExpired:
        error_msg = f"Code execution timed out after {timeout} seconds"
        return False, None, error_msg
    except Exception as e:
        error_msg = f"Exception during code execution: {str(e)}"
        return False, None, error_msg
    finally:
        # Note: We don't delete temp_dir here because we need to copy the image first
        # The caller should handle cleanup after copying the image
        pass


def check_completion_status(output_dir: Path, item_id: str) -> tuple[bool, bool]:
    """
    Check if an item has been processed.

    Returns:
        tuple: (is_completed, is_failed)
            - is_completed: True if successfully completed (has files in codes, images, and qa_pairs)
            - is_failed: True if failed (needs retry)
    """
    codes_dir = output_dir / "codes"
    images_dir = output_dir / "images"
    qa_dir = output_dir / "qa_pairs"

    # Check for successful completion - must have files in all three directories
    qa_file = qa_dir / f"{item_id}.json"
    code_file = codes_dir / f"{item_id}.py"

    # Check for image with different extensions
    image_file = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_image = images_dir / f"{item_id}{ext}"
        if potential_image.exists():
            image_file = potential_image
            break

    # All three files must exist for successful completion
    if qa_file.exists() and code_file.exists() and image_file is not None:
        return True, False

    # Check for failed completion
    failed_file = qa_dir / f"failed_{item_id}.json"
    if failed_file.exists():
        return False, True

    # Not processed yet
    return False, False


def save_results(output_dir: Path, item_id: str, parsed_result: Dict, qa_item: Dict, error_message: Optional[str] = None, timeout: int = 30, raw_response: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Save the modified code and new QA pair.

    Args:
        raw_response: The raw response text from the LLM (for debugging)

    Returns:
        tuple: (final_success, final_error_message)
    """
    # Create output directories
    codes_dir = output_dir / "codes"
    qa_dir = output_dir / "qa_pairs"
    images_dir = output_dir / "images"
    codes_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine success status
    success = error_message is None and parsed_result and parsed_result["feasibility"] == "YES"

    # Remove old failed file if this is a retry and now successful
    if success:
        old_failed_file = qa_dir / f"failed_{item_id}.json"
        if old_failed_file.exists():
            old_failed_file.unlink()

    # Variables for rendering status
    render_success = False
    render_error = None
    rendered_image_path = None

    # Save modified code if feasible and try to render
    if success and parsed_result["modified_code"]:
        code_file = codes_dir / f"{item_id}.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(parsed_result["modified_code"])

        # Try to execute the code and render the image
        render_success, temp_image_path, render_error = execute_code_and_render(
            parsed_result["modified_code"],
            item_id,
            timeout=timeout
        )

        if render_success and temp_image_path:
            # Copy the rendered image to the output directory
            image_extension = temp_image_path.suffix
            rendered_image_path = images_dir / f"{item_id}{image_extension}"
            shutil.copy2(temp_image_path, rendered_image_path)

            # Clean up the temporary directory
            temp_dir = temp_image_path.parent
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # Clean up temp directory even if rendering failed
            if temp_image_path:
                temp_dir = temp_image_path.parent
                shutil.rmtree(temp_dir, ignore_errors=True)
            # If rendering failed, mark as failed
            success = False
            if render_error:
                error_message = f"Rendering failed: {render_error}"

    # Save QA pair with status
    qa_result = {
        "item_id": item_id,
        "image_id": qa_item["image_id"],
        "original_question": qa_item["question"],
        "original_answer": qa_item["answer"],
        "split": qa_item.get("split", "unknown"),
        "success": success,
        "error_message": error_message,
        "feasibility": parsed_result["feasibility"] if parsed_result else None,
        "rationale": parsed_result["rationale"] if parsed_result else None,
        "new_answer": parsed_result["new_answer"] if parsed_result else None,
        "render_success": render_success,
        "render_error": render_error,
        "rendered_image": str(rendered_image_path.relative_to(output_dir)) if rendered_image_path else None,
        "raw_response": raw_response  # Add raw LLM response for debugging
    }

    # Use failed_ prefix for failed items
    prefix = "failed_" if not success else ""
    qa_file = qa_dir / f"{prefix}{item_id}.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_result, f, indent=2, ensure_ascii=False)

    # Return final success status and error message
    return success, error_message


def process_single_item_wrapper(args_tuple):
    """
    Wrapper function for ThreadPoolExecutor.
    Unpacks arguments and calls process_single_item.
    """
    (client, template, qa_item, codes_dir, images_dir,
     output_dir, item_id, model, timeout, is_retry) = args_tuple

    result, error_message = process_single_item(
        client=client,
        template=template,
        qa_item=qa_item,
        codes_dir=codes_dir,
        images_dir=images_dir,
        output_dir=output_dir,
        item_id=item_id,
        model=model,
        timeout=timeout
    )

    # Return a dict with all information needed for results tracking
    return {
        "item_id": item_id,
        "image_id": qa_item["image_id"],
        "success": error_message is None and result and result["feasibility"] == "YES",
        "error_message": error_message,
        "feasibility": result["feasibility"] if result else None,
        "is_retry": is_retry,
        "result": result
    }


def process_single_item(
    client: OpenAI,
    template: Template,
    qa_item: Dict,
    codes_dir: Path,
    images_dir: Path,
    output_dir: Path,
    item_id: str,
    model: str,
    timeout: int = 30
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Process a single QA item.

    Returns:
        tuple: (parsed_result, error_message)
    """
    image_id = qa_item["image_id"]
    code_path = codes_dir / f"{image_id}.py"

    # Try to find image with different extensions
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_path = images_dir / f"{image_id}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break

    # Check if code file exists
    if not code_path.exists():
        error_msg = f"Code file not found for image_id {image_id}"
        print(f"Warning: {error_msg}")
        final_success, final_error = save_results(output_dir, item_id, None, qa_item, error_msg, timeout)
        return None, final_error

    # Check if image file exists
    if image_path is None:
        error_msg = f"Image file not found for image_id {image_id}"
        print(f"Warning: {error_msg}")
        final_success, final_error = save_results(output_dir, item_id, None, qa_item, error_msg, timeout)
        return None, final_error

    try:
        # Load the code
        python_code = load_code(str(code_path))

        # Process answer based on split type
        split_type = qa_item.get("split", "descriptive")

        if split_type == "reasoning":
            # Parse reasoning answer from "xxx Answer: xxx" format
            reasoning_process, answer_only = parse_reasoning_answer(qa_item["answer"])

            if reasoning_process is None or answer_only is None:
                error_msg = f"Failed to parse reasoning answer format (expected 'xxx Answer: xxx')"
                print(f"Warning: {error_msg} for item {item_id}")
                final_success, final_error = save_results(output_dir, item_id, None, qa_item, error_msg, timeout)
                return None, final_error

            # Render the prompt with separated reasoning process and answer
            prompt = template.render(
                python_code=python_code,
                question=qa_item["question"],
                current_reasoning_process=reasoning_process,
                current_answer=answer_only
            )
        else:
            # For descriptive type, use current_answer directly
            prompt = template.render(
                python_code=python_code,
                question=qa_item["question"],
                current_answer=qa_item["answer"]
            )

        # Call OpenAI API with image
        response = call_openai_api(client, prompt, str(image_path), model)
        if response is None:
            error_msg = "OpenAI API call failed (returned None)"
            print(f"Warning: {error_msg} for item {item_id}")
            final_success, final_error = save_results(output_dir, item_id, None, qa_item, error_msg, timeout)
            return None, final_error

        # Parse the response
        parsed_result = parse_response(response)

        # Check if parsing was successful
        if not parsed_result or parsed_result["feasibility"] is None:
            error_msg = "Failed to parse API response"
            print(f"Warning: {error_msg} for item {item_id}")
            final_success, final_error = save_results(output_dir, item_id, parsed_result, qa_item, error_msg, timeout, raw_response=response)
            return parsed_result, final_error

        # Post-process reasoning-type answers: convert back to "xxx Answer: xxx" format
        if split_type == "reasoning" and parsed_result["new_answer"]:
            formatted_answer, format_error = format_reasoning_answer(parsed_result["new_answer"])

            if formatted_answer is None:
                # Failed to format reasoning answer
                error_msg = f"Reasoning answer format error: {format_error}"
                print(f"Warning: {error_msg} for item {item_id}")
                final_success, final_error = save_results(output_dir, item_id, parsed_result, qa_item, error_msg, timeout, raw_response=response)
                return parsed_result, final_error

            # Replace with formatted answer
            parsed_result["new_answer"] = formatted_answer

        # Save results and get final status (includes rendering result)
        final_success, final_error = save_results(output_dir, item_id, parsed_result, qa_item, None, timeout, raw_response=response)

        # Return parsed result with final error status (which may include rendering errors)
        return parsed_result, final_error

    except Exception as e:
        error_msg = f"Exception during processing: {str(e)}"
        print(f"Error: {error_msg} for item {item_id}")
        final_success, final_error = save_results(output_dir, item_id, None, qa_item, error_msg, timeout)
        return None, final_error


def main():
    parser = argparse.ArgumentParser(description="Chart code modification")
    parser.add_argument(
        "--qa-path",
        type=str,
        default="data/ECD/ECD_QAs_one_per_image.json",
        help="Path to QA pairs JSON file"
    )
    parser.add_argument(
        "--codes-dir",
        type=str,
        default="data/ECD/codes_with_seed",
        help="Directory containing chart code files"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/ECD/rendered_images_with_seed",
        help="Directory containing chart images (for reference)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="prompts/prompt_data_syn_descriptive.md",
        help="Path to prompt template: prompts/prompt_data_syn_descriptive.md or prompts/prompt_data_syn_reasoning.md"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ECD/syn_data/descriptive",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for code execution in seconds (default: 300)"
    )
    parser.add_argument(
        "--split-type",
        type=str,
        choices=["descriptive", "reasoning"],
        default="reasoning",
        help="Filter samples by split type: 'descriptive' or 'reasoning'. If not specified, process all samples."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent workers for parallel processing. Default: None (uses CPU count)"
    )

    args = parser.parse_args()
    
    assert args.split_type in args.template, "Split type must match the template used."

    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    # Set up paths
    codes_dir = Path(args.codes_dir)
    images_dir = Path(args.images_dir)
    template_path = Path(args.template)

    # Set up output directory (use fixed directory for resume capability)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template and data
    print(f"Loading prompt template from {template_path}...")
    template = load_prompt_template(str(template_path))

    print(f"Loading QA data from {args.qa_path}...")
    qa_data = load_qa_data(args.qa_path)

    # Filter by split type if specified
    if args.split_type:
        original_count = len(qa_data)
        qa_data = [item for item in qa_data if item.get("split") == args.split_type]
        print(f"Filtered {original_count} items to {len(qa_data)} items with split='{args.split_type}'")

    # Apply limit
    if args.limit:
        qa_data = qa_data[:args.limit]
        print(f"Processing first {args.limit} items")

    print(f"Total items to process: {len(qa_data)}")
    print(f"Output directory: {output_dir}")

    # Build image_id to question index mapping
    # Each question for the same image_id gets a sequential index starting from 0
    from collections import defaultdict
    image_id_counter = defaultdict(int)
    image_id_to_question_idx = {}

    for idx, qa_item in enumerate(qa_data):
        image_id = qa_item['image_id']
        question_idx = image_id_counter[image_id]
        image_id_to_question_idx[idx] = question_idx
        image_id_counter[image_id] += 1

    # Check existing completion status
    print("\nChecking existing results...")
    already_completed = 0
    to_process = []
    for idx, qa_item in enumerate(qa_data):
        question_idx = image_id_to_question_idx[idx]
        item_id = f"{qa_item['image_id']}_{question_idx:06d}"
        is_completed, is_failed = check_completion_status(output_dir, item_id)

        if is_completed:
            already_completed += 1
        else:
            # Process new items and retry failed ones
            to_process.append((idx, qa_item, item_id, is_failed))

    print(f"Already completed: {already_completed}")
    print(f"To process (new + failed): {len(to_process)}")
    if already_completed > 0:
        print(f"  - New items: {len([x for x in to_process if not x[3]])}")
        print(f"  - Failed items (retry): {len([x for x in to_process if x[3]])}")

    # Determine number of workers
    num_workers = args.workers if args.workers else os.cpu_count()
    print(f"\nUsing {num_workers} concurrent workers for parallel processing")

    # Process each item with ThreadPoolExecutor
    results = []
    successful = 0
    failed = 0
    skipped = already_completed

    # Prepare arguments for all tasks
    task_args = []
    for idx, qa_item, item_id, is_retry in to_process:
        task_args.append((
            client, template, qa_item, codes_dir, images_dir,
            output_dir, item_id, args.model, args.timeout, is_retry
        ))

    # Process items in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_item_wrapper, args): args for args in task_args}

        # Process completed tasks with progress bar
        with tqdm(total=len(to_process), desc="Processing items") as pbar:
            for future in as_completed(futures):
                try:
                    result_dict = future.result()

                    # Count success/failure
                    if result_dict["success"]:
                        successful += 1
                    else:
                        failed += 1

                    # Track results
                    results.append(result_dict)

                except Exception as e:
                    # Handle any unexpected errors during processing
                    args = futures[future]
                    item_id = args[6]  # item_id is the 7th element in args tuple
                    print(f"\nUnexpected error processing {item_id}: {e}")
                    failed += 1
                    results.append({
                        "item_id": item_id,
                        "image_id": args[2]["image_id"],  # qa_item is the 3rd element
                        "success": False,
                        "error_message": f"Unexpected error: {str(e)}",
                        "feasibility": None,
                        "is_retry": args[9]  # is_retry is the 10th element
                    })

                finally:
                    pbar.update(1)

    # Count error types and rendering errors
    error_types = {}
    render_failed_count = 0
    for r in results:
        if r["error_message"]:
            error_type = r["error_message"].split(":")[0]  # Get first part of error message
            error_types[error_type] = error_types.get(error_type, 0) + 1
            # Count rendering failures separately
            if "Rendering failed" in r["error_message"]:
                render_failed_count += 1

    # Count feasibility
    feasibility_no = sum(1 for r in results if r["feasibility"] == "NO")
    feasibility_yes = sum(1 for r in results if r["feasibility"] == "YES")

    # Save summary
    summary = {
        "total_items": len(qa_data),
        "skipped": skipped,
        "processed": len(to_process),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(to_process) if to_process else 0,
        "overall_completion_rate": (successful + skipped) / len(qa_data) if qa_data else 0,
        "feasibility_stats": {
            "YES": feasibility_yes,
            "NO": feasibility_no,
            "None": len(to_process) - feasibility_yes - feasibility_no
        },
        "rendering_stats": {
            "successful": successful,  # All successful items have rendered images
            "failed": render_failed_count,
            "render_rate": successful / len(to_process) if to_process else 0
        },
        "error_types": error_types,
        "model": args.model,
        "split_type": args.split_type,
        "timestamp": datetime.now().isoformat(),
        "detailed_results": results
    }

    # Generate timestamp for summary filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{timestamp_str}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total items: {summary['total_items']}")
    print(f"Skipped (already completed): {summary['skipped']}")
    print(f"Processed in this run: {summary['processed']}")
    print(f"  - Successful: {summary['successful']}")
    print(f"  - Failed: {summary['failed']}")
    print(f"Success rate (this run): {summary['success_rate']:.2%}")
    print(f"Overall completion rate: {summary['overall_completion_rate']:.2%}")
    print(f"\nFeasibility breakdown:")
    print(f"  YES: {feasibility_yes}")
    print(f"  NO: {feasibility_no}")
    print(f"\nRendering statistics:")
    print(f"  Successfully rendered: {summary['rendering_stats']['successful']}")
    print(f"  Rendering failed: {summary['rendering_stats']['failed']}")
    print(f"  Render success rate: {summary['rendering_stats']['render_rate']:.2%}")
    if error_types:
        print(f"\nError types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Codes: {output_dir / 'codes'}")
    print(f"  - Images: {output_dir / 'images'}")
    print(f"  - QA pairs: {output_dir / 'qa_pairs'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
