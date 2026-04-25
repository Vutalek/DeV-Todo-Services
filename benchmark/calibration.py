import argparse
from pathlib import Path

from config import (
    DEFAULT_JUDGE_MODELS,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_PROMPT_PATH,
)
from io_utils import read_calibration_jsonl, write_jsonl, load_text
from llm import run_judge
from deviation_analysis import deviation_analysis
from models import CalibrationTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration of judges for benchmark")

    parser.add_argument("--input", required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for results.jsonl")

    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=DEFAULT_JUDGE_MODELS,
        help="List of judge models",
    )

    parser.add_argument(
        "--judge-system-prompt",
        default=str(DEFAULT_JUDGE_PROMPT_PATH),
        help="Path to judge system prompt txt",
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

    input_examples = read_calibration_jsonl(args.input)
    reference_ranks = [exmpl.reference_ranks for exmpl in input_examples]
    judge_prompt = load_text(args.judge_system_prompt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"

    result_rows: list[CalibrationTable] = []

    for judge_model in args.judge_models:
        judge_results = []
        for example in input_examples:
            print(f"  -> judge={judge_model}")
            judge_result = run_judge(
                judge_model=judge_model,
                judge_system_prompt=judge_prompt,
                question=example.question,
                reference=example.reference,
                answer=example.reference_from_llm,
                max_tokens=args.judge_max_tokens,
            )
            judge_results.append(judge_result)

        row = deviation_analysis(
            model_name=judge_model,
            judge_scores=judge_results,
            references=reference_ranks
        )

        result_rows.append(row)

    write_jsonl(str(results_path), result_rows)

    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
