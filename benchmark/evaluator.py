from statistics import mean
from models import AveragedJudgeScores


def average_judge_results(judge_results: list[dict]) -> AveragedJudgeScores:
    if not judge_results:
        raise ValueError("judge_results is empty")

    score = mean(j["scores"]["score"] for j in judge_results)
    correctness = mean(j["scores"]["correctness"] for j in judge_results)
    completeness = mean(j["scores"]["completeness"] for j in judge_results)
    clarity = mean(j["scores"]["clarity"] for j in judge_results)

    justification = " | ".join(
        f'{j["judge_model"]}: {j["scores"]["justification"]}'
        for j in judge_results
    )

    return AveragedJudgeScores(
        score=round(score, 3),
        correctness=round(correctness, 3),
        completeness=round(completeness, 3),
        clarity=round(clarity, 3),
        justification=justification,
    )
