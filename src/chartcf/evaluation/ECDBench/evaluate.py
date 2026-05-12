import argparse
import asyncio
import json
import logging
import os
import pathlib
import random
import re
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


# Answer Judge Prompt from original evaluate_on_ecdbench.py
ANSWER_JUDGE_PROMPT = """Your task is to rigorously evaluate whether the VLM's prediction aligns with the expected answer for a question regarding a chart (note: the chart image itself is not provided here). The evaluation should focus solely on factual alignment between the prediction and the ground truth. Minor differences in wording or phrasing are acceptable as long as the core meaning remains consistent. 

For numerical answers, **a margin of error up to 5% is acceptable** unless explicitly stated otherwise in the question. However, partial correctness or incomplete responses should not be considered correct.

- Question: {question}
- Expected Answer: {answer}
- Prediction: {prediction}

Please respond using the following format:
Correctness: (Yes or No)
"""


def post_process_model_response(response: str) -> int:
    """Extract correctness from model response. Returns 1 for Yes, 0 for No, -1 for parse failure."""
    match = re.search(r"Correctness:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

    if match:
        answer_string = match.group(1).strip()
        if "yes" in answer_string.lower():
            return 1
        elif "no" in answer_string.lower():
            return 0
        else:
            return -1
    else:
        logging.info("Failed to extract the correctness")
        return -1


def build_eval_prompt(question: str, gt_answer: str, pred_answer: str) -> str:
    """Build evaluation prompt."""
    return ANSWER_JUDGE_PROMPT.format(
        question=question,
        answer=gt_answer,
        prediction=pred_answer
    )


async def _run_eval(
    key: str,
    item: Dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    max_retries: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    last_error = None
    prompt = build_eval_prompt(
        item["question"],
        item["gt_answer"],
        item["pred_answer"]
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                    top_p=0.95,
                )
            content = response.choices[0].message.content
            correctness = post_process_model_response(content)
            return {
                "ok": True,
                "key": key,
                "correctness": correctness,
                "output": content,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                await asyncio.sleep(backoff + random.random())
    return {
        "ok": False,
        "key": key,
        "error": last_error or "unknown_error",
        "attempts": max_retries,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_data_path", required=True,
                        help="Path to the inference data (JSON file)")
    parser.add_argument("--output_file", type=str, default="",
                        help="Path to the output file")
    parser.add_argument("--eval_model_name", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to save logging output")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Add file handler if log_file is provided
    if args.log_file:
        log_path = pathlib.Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {args.log_file}")

    infer_data_path = pathlib.Path(args.infer_data_path)

    if not infer_data_path.exists():
        raise FileNotFoundError(f"Inference data file not found: {infer_data_path}")

    logging.info(f'infer_result: {infer_data_path}')

    logging.info(f"Reading {infer_data_path}...")
    with infer_data_path.open("r") as f:
        data = json.load(f)

    # Handle both dict format (from async inference) and list format (from qwen inference)
    if isinstance(data, dict):
        items = list(data.values())
        keys = list(data.keys())
    else:
        items = data
        keys = [str(i) for i in range(len(data))]

    logging.info(f"Total items: {len(items)}")
    logging.info(f'Total items: {len(items)}')

    # Setup OpenAI client
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    max_concurrent = max(1, args.max_concurrent)
    max_retries = max(1, args.max_retries)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build tasks for evaluation
    tasks = []
    for key, item in zip(keys, items):
        tasks.append(
            asyncio.create_task(
                _run_eval(
                    key,
                    item,
                    client,
                    args.eval_model_name,
                    max_retries,
                    semaphore,
                ))
        )

    # Store results in order
    results_map = {}
    failed_count = 0

    for coro in tqdm_asyncio.as_completed(tasks,
                                          total=len(tasks),
                                          desc="evaluating"):
        result = await coro
        key = result["key"]
        if result["ok"]:
            results_map[key] = {
                "correctness": result["correctness"],
                "output": result["output"],
            }
            # logging.info(f'***Correctness of {key}***: {result["correctness"]}')
        else:
            failed_count += 1
            results_map[key] = {
                "correctness": -1,
                "output": f"Error: {result['error']}",
            }
            logging.error(f'***Failed {key}***: {result["error"]}')

    await client.close()

    # Merge results back into items
    eval_results = []
    for key, item in zip(keys, items):
        item_result = dict(item)
        if key in results_map:
            item_result["correctness"] = results_map[key]["correctness"]
            item_result["output"] = results_map[key]["output"]
        else:
            item_result["correctness"] = -1
            item_result["output"] = "Not evaluated"
        eval_results.append(item_result)

    # Calculate statistics (same logic as original evaluate_on_ecdbench.py)
    correct_count = 0
    split_correct_counts = {}
    split_total_counts = {}

    for item in eval_results:
        correctness = item["correctness"]
        split = item.get("split", "unknown")

        if split not in split_correct_counts:
            split_correct_counts[split] = 0
            split_total_counts[split] = 0

        if correctness == 1:
            split_correct_counts[split] += 1
            correct_count += 1
        split_total_counts[split] += 1

    # logging.info results
    total_count = len(eval_results)
    accuracy = 100 * correct_count / total_count if total_count > 0 else 0

    logging.info("\n" + "=" * 50)
    logging.info("Evaluation Complete")
    logging.info("=" * 50)
    logging.info(f"Total: {total_count}\tAccuracy: {accuracy:.2f}%")
    logging.info(f"Failed evaluations: {failed_count}")

    # logging.info accuracy for each split
    for split_name, total in split_total_counts.items():
        split_accuracy = 100 * split_correct_counts[split_name] / total if total > 0 else 0
        logging.info(f"{split_name}: {total}\tAccuracy: {split_accuracy:.2f}%")

    logging.info('*************** Performance *****************')
    logging.info(f'Total Accuracy: {accuracy:.2f}%')
    for split_name, total in split_total_counts.items():
        split_accuracy = 100 * split_correct_counts[split_name] / total if total > 0 else 0
        logging.info(f'{split_name}: {split_accuracy:.2f}%')

    # Save results
    if args.output_file:
        output_path = pathlib.Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        logging.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
