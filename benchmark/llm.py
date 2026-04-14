import time
from typing import Optional
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from models import JudgeScores, Task

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def _tokens_per_sec(total_tokens: int, elapsed_sec: float) -> Optional[float]:
    if elapsed_sec <= 0:
        return None
    return round(total_tokens / elapsed_sec, 3)


def run_candidate(
    model: str,
    system_prompt: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()

    parsed = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=Task
    )

    elapsed = time.perf_counter() - started
    result = parsed.choices[0].message.parsed.model_dump_json(
        ensure_ascii=False)
    usage = parsed.usage

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    return {
        "response": result,
        "latency_sec": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": _tokens_per_sec(total_tokens, elapsed),
    }


def build_judge_user_prompt(question: str, reference: str, answer: str) -> str:
    return f"""Evaluate the assistant answer relative to the reference answer.

User question:
{question}

Reference answer:
{reference}

Assistant answer:
{answer}
"""


def run_judge(
    judge_model: str,
    judge_system_prompt: str,
    question: str,
    reference: str,
    answer: str,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()

    parsed = client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {
                "role": "user",
                "content": build_judge_user_prompt(question, reference, answer),
            },
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format=JudgeScores,
    )

    elapsed = time.perf_counter() - started
    result = parsed.choices[0].message.parsed.model_dump()
    usage = parsed.usage

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    return {
        "judge_model": judge_model,
        "scores": result,
        "latency_sec": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": _tokens_per_sec(total_tokens, elapsed),
    }
