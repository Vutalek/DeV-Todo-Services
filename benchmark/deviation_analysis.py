from typing import Dict, List

from models import JudgeScores, CalibrationTable


def deviation_analysis(
    model_name: str,
    judge_scores: List[JudgeScores],
    references: List[Dict[str, float]]
) -> CalibrationTable:
    score = []
    correctness = []
    completeness = []
    clarity = []
    for judge_score, reference in zip(judge_scores, references):
        score.append(judge_score.score - reference["score"])
        correctness.append(judge_score.correctness - reference["correctness"])
        completeness.append(judge_score.completeness - reference["completeness"])
        clarity.append(judge_score.clarity - reference["clarity"])
    
    return CalibrationTable(
        model_name=model_name,
        score=sum(score) / len(score),
        correctness=sum(correctness) / len(correctness),
        completeness=sum(completeness) / len(completeness),
        clarity=sum(clarity) / len(clarity)
    )

