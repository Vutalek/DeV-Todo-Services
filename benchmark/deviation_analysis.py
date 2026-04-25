from typing import Dict, List, Any

from models import CalibrationTable


def deviation_analysis(
    model_name: str,
    judge_scores: Dict[str, Any],
    references: List[Dict[str, float]]
) -> CalibrationTable:
    score = []
    correctness = []
    completeness = []
    clarity = []
    for judge_score, reference in zip(judge_scores, references):
        score.append(judge_score["scores"].score - reference["score"])
        correctness.append(judge_score["scores"].correctness - reference["correctness"])
        completeness.append(judge_score["scores"].completeness - reference["completeness"])
        clarity.append(judge_score["scores"].clarity - reference["clarity"])
    
    return CalibrationTable(
        model_name=model_name,
        score=sum(score) / len(score),
        correctness=sum(correctness) / len(correctness),
        completeness=sum(completeness) / len(completeness),
        clarity=sum(clarity) / len(clarity)
    )

