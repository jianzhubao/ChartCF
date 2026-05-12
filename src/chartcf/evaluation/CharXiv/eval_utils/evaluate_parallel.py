import argparse
import functools
import json
import multiprocessing as mp
import os
from typing import Dict, Tuple

from dotenv import load_dotenv, find_dotenv
# from openai import OpenAI
from openai import OpenAI
from tqdm import tqdm


_CLIENT = None


def _init_client(api_key: str, base_url: str):
    """Initializer used by multiprocessing workers."""
    global _CLIENT
    _CLIENT = OpenAI(api_key=api_key, base_url=base_url)


def _ensure_client(api_key: str = None, base_url: str = None):
    global _CLIENT
    if _CLIENT is None:
        if api_key is None or base_url is None:
            raise RuntimeError("API credentials are required to create a client")
        _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT


def _run_descriptive_query(query: Dict,
                           api_key: str = None,
                           base_url: str = None,
                           client: OpenAI = None,
                           grading_model: str = None) -> Dict:
    local_client = client or _ensure_client(api_key, base_url)
    from descriptive_utils import get_descriptive_result_gpt
    result = get_descriptive_result_gpt(local_client, query['grading_query'],
                                        len(query['resp_keys']),
                                        model=grading_model)
    return {**query, **result}


def _run_reasoning_query(item: Tuple[str, Dict],
                         api_key: str = None,
                         base_url: str = None,
                         client: OpenAI = None,
                         grading_model: str = None):
    figure_id, query = item
    local_client = client or _ensure_client(api_key, base_url)
    from reasoning_utils import get_reasoning_result_gpt
    ext, scr = get_reasoning_result_gpt(local_client,
                                        query['grading_query'],
                                        model=grading_model)
    processed = dict(query)
    processed['extracted_answer'] = ext
    processed['score'] = scr
    processed.pop('grading_query', None)
    return figure_id, processed, ext, scr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--gen_prefix', type=str, default='gen-')
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help="Number of processes for parallel API calls.")
    parser.add_argument('--grading_model',
                        type=str,
                        default='gpt-4o-2024-08-06',
                        help="Model to use for grading API calls.")
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/public_benchmarks/CharXiv/data',
                        help="Directory containing CharXiv annotation JSON files.")
    args = parser.parse_args()

    num_workers = max(1, args.num_workers)

    dotenv_path_found = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path_found)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    args.input_file = f"{args.data_dir}/{args.mode}_{args.split}.json"
    args.resp_file = f"{args.output_name}/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}.json"
    args.output_file = args.resp_file.replace(args.gen_prefix, "scores-")
    print(f"Output file: {args.output_file}")

    data, response = json.load(open(args.input_file)), json.load(
        open(args.resp_file))
    mode = 'descriptive' if 'descriptive' in args.resp_file.split(
        '-')[-1] else 'reasoning'

    if mode == 'descriptive':
        from descriptive_utils import preprocess_descriptive_grading_queries, build_descriptive_grading_queries, \
                postprocess_descriptive_grading_queries
        groups = preprocess_descriptive_grading_queries(data, response)
        queries = build_descriptive_grading_queries(groups)

        if num_workers == 1:
            combined_queries = []
            for query in tqdm(queries, desc="grading-descriptive"):
                combined_queries.append(
                    _run_descriptive_query(query, client=client, grading_model=args.grading_model))
        else:
            ctx = mp.get_context("spawn")
            worker = functools.partial(_run_descriptive_query,
                                       api_key=api_key,
                                       base_url=base_url,
                                       grading_model=args.grading_model)
            with ctx.Pool(processes=num_workers,
                          initializer=_init_client,
                          initargs=(api_key, base_url)) as pool:
                combined_queries = list(
                    tqdm(pool.imap(worker, queries),
                         total=len(queries),
                         desc="grading-descriptive"))
        queries = postprocess_descriptive_grading_queries(combined_queries)

    elif mode == 'reasoning':
        from reasoning_utils import build_reasoning_grading_queries
        queries = build_reasoning_grading_queries(data, response)
        items = list(queries.items())

        if num_workers == 1:
            iterator = (
                _run_reasoning_query(item, client=client, grading_model=args.grading_model)
                for item in tqdm(items, desc="grading-reasoning"))
            for figure_id, processed, ext, scr in iterator:
                print('ext:', ext)
                print('score:', scr)
                queries[figure_id] = processed
        else:
            ctx = mp.get_context("spawn")
            worker = functools.partial(_run_reasoning_query,
                                       api_key=api_key,
                                       base_url=base_url,
                                       grading_model=args.grading_model)
            with ctx.Pool(processes=num_workers,
                          initializer=_init_client,
                          initargs=(api_key, base_url)) as pool:
                iterator = tqdm(pool.imap(worker, items),
                                total=len(items),
                                desc="grading-reasoning")
                for figure_id, processed, ext, scr in iterator:
                    print('ext:', ext)
                    print('score:', scr)
                    queries[figure_id] = processed
    else:
        raise ValueError("Mode not supported")

    with open(args.output_file, "w") as f:
        json.dump(queries, f, indent=4)
