from statistics import mean


def aggregate_results(rows: list[dict]) -> dict:
    grouped: dict[str, dict] = {}

    for row in rows:
        model = row["model_name"]
        scores = row["avg_scores"]

        if model not in grouped:
            grouped[model] = {
                "count": 0,
                "score": [],
                "correctness": [],
                "completeness": [],
                "clarity": [],
                "latency_sec": [],
                "tokens_per_sec": [],
                "prompt_tokens": [],
                "completion_tokens": [],
                "total_tokens": [],
            }

        agg = grouped[model]
        agg["count"] += 1
        agg["score"].append(scores["score"])
        agg["correctness"].append(scores["correctness"])
        agg["completeness"].append(scores["completeness"])
        agg["clarity"].append(scores["clarity"])

        agg["latency_sec"].append(row["candidate_latency_sec"])
        if row["candidate_tokens_per_sec"] is not None:
            agg["tokens_per_sec"].append(row["candidate_tokens_per_sec"])

        agg["prompt_tokens"].append(row["candidate_prompt_tokens"])
        agg["completion_tokens"].append(row["candidate_completion_tokens"])
        agg["total_tokens"].append(row["candidate_total_tokens"])

    summary_by_model = {}

    for model, agg in grouped.items():
        summary_by_model[model] = {
            "count": agg["count"],
            "avg_score": round(mean(agg["score"]), 3),
            "avg_correctness": round(mean(agg["correctness"]), 3),
            "avg_completeness": round(mean(agg["completeness"]), 3),
            "avg_clarity": round(mean(agg["clarity"]), 3),
            "grand_average": round(mean([
                mean(agg["score"]),
                mean(agg["correctness"]),
                mean(agg["completeness"]),
                mean(agg["clarity"]),
            ]), 3),
            "avg_latency_sec": round(mean(agg["latency_sec"]), 3),
            "avg_tokens_per_sec": round(mean(agg["tokens_per_sec"]), 3) if agg["tokens_per_sec"] else None,
            "avg_prompt_tokens": round(mean(agg["prompt_tokens"]), 3),
            "avg_completion_tokens": round(mean(agg["completion_tokens"]), 3),
            "avg_total_tokens": round(mean(agg["total_tokens"]), 3),
        }

    ranking = sorted(
        [{"model": model, **stats}
            for model, stats in summary_by_model.items()],
        key=lambda x: x["grand_average"],
        reverse=True,
    )

    return {
        "summary_by_model": summary_by_model,
        "ranking": ranking,
    }
