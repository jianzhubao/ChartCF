import argparse
import asyncio
import json
import logging
import os
import pathlib
import random
from typing import Any, Dict

from dotenv import find_dotenv, load_dotenv
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


# Few-shot examples for evaluation
EXAMPLES = [
    {
        "query":
        "<question> What was the incremental increase in revenue from 2020 to 2021? <groundtruth answer> 5 million $ <answer> 20\n</s>",
        "answer": "False"
    },
    {
        "query":
        "<question> What percentage of government spending was allocated to infrastructure in 2020? <groundtruth answer> 10% <answer> 14-4=10\n</s>",
        "answer": "True"
    },
    {
        "query":
        "<question> What is the total production of Wind Energy in the four months from January to April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January to April 2021 is 2450 MW.",
        "answer": "True"
    },
    {
        "query":
        "<question> What is the total of manufactured goods for UK and Germany combined? <groundtruth answer> 5 <answer> Five",
        "answer": "True"
    },
]

# Create example template
EXAMPLE_TEMPLATE = """
    User: {query}
    AI: {answer}
    """

# Create prompt example from above template
EXAMPLE_PROMPT = PromptTemplate(input_variables=["query", "answer"],
                                template=EXAMPLE_TEMPLATE)

# Instruction prefix
PREFIX = """Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions. The output should be only "True" or "False". Note that if the groundtruth answer is a numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is marked as correct when groundtruth value is 100 million."""

# Suffix with user input and output indicator
SUFFIX = """
    User: {query}
    AI: """

# Build the few-shot prompt template
FEW_SHOT_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=EXAMPLE_PROMPT,
    prefix=PREFIX,
    suffix=SUFFIX,
    input_variables=["query"],
    example_separator="\n\n")


def build_eval_prompt(question: str, answer_gt: str, answer_pred: str) -> str:
    """Build evaluation prompt with few-shot examples using langchain."""
    query = f"<question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}"
    return FEW_SHOT_PROMPT_TEMPLATE.format(query=query)


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

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    n=1,
                    max_tokens=512,
                    temperature=0,
                    top_p=1,
                    seed=42,
                )
            content = response.choices[0].message.content
            # logging.info(f'response of {key}: {response}')
            # if "True" in content:
            #     score = 1
            # elif "False" in content:
            #     score = 0
            # else:
            #     score = 0  # Default to 0 if unclear
            score = 0
            if 'True' in content:
                score = 1
            if 'False' in content:
                score = 0
            return {
                "ok": True,
                "key": key,
                "score": score,
                "response": content,
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
                        help="Path to the output log file")
    parser.add_argument("--eval_model_name", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=32)
    args = parser.parse_args()

    # Setup logging
    if args.output_file:
        logging.basicConfig(filename=args.output_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    infer_data_path = pathlib.Path(args.infer_data_path)

    if not infer_data_path.exists():
        raise FileNotFoundError(f"Inference data file not found: {infer_data_path}")

    logging.info(f'infer_result: {infer_data_path}')

    print(f"Reading {infer_data_path}...")
    with infer_data_path.open("r") as f:
        data = json.load(f)

    # Handle both dict format (from async inference) and list format (from qwen inference)
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    print(f"Total items to evaluate: {len(items)}")
    logging.info(f'Total items to evaluate: {len(items)}')

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
    for idx, item in enumerate(items):
        key = str(idx)
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

    scores = []
    failed_count = 0

    for coro in tqdm_asyncio.as_completed(tasks,
                                          total=len(tasks),
                                          desc="evaluating"):
        result = await coro
        key = result["key"]
        if result["ok"]:
            score = result["score"]
            scores.append(score)
            # logging.info(f'***Score of {key}***: {score}')
        else:
            failed_count += 1
            # logging.error(f'***Failed {key}***: {result["error"]}')

    await client.close()

    # Calculate final accuracy
    if scores:
        avg_score = sum(scores) / len(scores)
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluation Complete")
        logging.info(f"{'='*50}")
        logging.info(f"Total evaluated: {len(scores)}")
        logging.info(f"Failed: {failed_count}")
        logging.info(f"Average Score (Accuracy) {sum(scores)}/{len(scores)}: {avg_score:.4f}")
        logging.info(f"{'='*50}")

        logging.info('*************** Performance *****************')
        logging.info('average')
        logging.info('%.4f' % avg_score)
    else:
        logging.info("No items were successfully evaluated.")
        logging.error("No items were successfully evaluated.")


if __name__ == "__main__":
    asyncio.run(main())
