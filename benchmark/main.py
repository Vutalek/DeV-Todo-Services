import argparse
from pathlib import Path

from config import (
    DEFAULT_CANDIDATE_MODELS,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_CANDIDATE_PROMPT_PATH,
    DEFAULT_JUDGE_PROMPT_PATH,
)
from io_utils import read_jsonl, write_jsonl, write_json, load_text
from llm import run_candidate, run_judge
from evaluator import average_judge_results
from aggregator import aggregate_results
from models import CandidateRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="benchmark runner for OpenRouter models")

    parser.add_argument("--input", required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for results.jsonl and summary.json")

    parser.add_argument(
        "--candidate-models",
        nargs="+",
        default=DEFAULT_CANDIDATE_MODELS,
        help="List of candidate models",
    )
    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=DEFAULT_JUDGE_MODELS,
        help="List of judge models",
    )

    parser.add_argument(
        "--candidate-system-prompt",
        default=str(DEFAULT_CANDIDATE_PROMPT_PATH),
        help="Path to candidate system prompt txt",
    )
    parser.add_argument(
        "--judge-system-prompt",
        default=str(DEFAULT_JUDGE_PROMPT_PATH),
        help="Path to judge system prompt txt",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Candidate generation temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Candidate max tokens",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Judge max tokens",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_examples = read_jsonl(args.input)
    candidate_prompt = load_text(args.candidate_system_prompt)
    judge_prompt = load_text(args.judge_system_prompt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    result_rows: list[dict] = []

    total_runs = len(input_examples) * len(args.candidate_models)
    current = 0

    for example in input_examples:
        for candidate_model in args.candidate_models:
            current += 1
            print(
                f"[{current}/{total_runs}] candidate={candidate_model} id={example.id}")

            candidate_result = run_candidate(
                model=candidate_model,
                system_prompt=candidate_prompt,
                question=example.question,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            judge_results = []
            for judge_model in args.judge_models:
                print(f"  -> judge={judge_model}")
                judge_result = run_judge(
                    judge_model=judge_model,
                    judge_system_prompt=judge_prompt,
                    question=example.question,
                    reference=example.reference,
                    answer=candidate_result["response"],
                    max_tokens=args.judge_max_tokens,
                )
                judge_results.append(judge_result)

            avg_scores = average_judge_results(judge_results)

            row = CandidateRunResult(
                id=example.id,
                question=example.question,
                reference=example.reference,
                response=candidate_result["response"],
                model_name=candidate_model,
                candidate_latency_sec=candidate_result["latency_sec"],
                candidate_prompt_tokens=candidate_result["prompt_tokens"],
                candidate_completion_tokens=candidate_result["completion_tokens"],
                candidate_total_tokens=candidate_result["total_tokens"],
                candidate_tokens_per_sec=candidate_result["tokens_per_sec"],
                judge_models=args.judge_models,
                judge_results=judge_results,
                avg_scores=avg_scores,
            ).model_dump()

            result_rows.append(row)

    write_jsonl(str(results_path), result_rows)

    summary = aggregate_results(result_rows)
    write_json(str(summary_path), summary)

    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
