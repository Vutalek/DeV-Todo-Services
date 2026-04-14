from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class InputExample(BaseModel):
    id: str
    question: str
    reference: str


class Task(BaseModel):
    name: str = Field(description="Название задачи")
    desc: str = Field(default="", max_length=600,
                      description="Краткое описание (не более 200 токенов)")
    prio: int = Field(ge=1, le=5, description="Приоритет от 1 до 5")
    time: int = Field(gt=0, description="Время в часах")
    roadmap: str = Field(default="", description="Roadmap для задачи")
    column: List[Literal['Беклог', 'В работе']] = Field(
        description='Колонка в которой будет находиться задача')


class JudgeScores(BaseModel):
    score: int = Field(..., ge=1, le=5)
    correctness: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    justification: str


class AveragedJudgeScores(BaseModel):
    score: float
    correctness: float
    completeness: float
    clarity: float
    justification: str = Field(max_length=500)


class CandidateRunResult(BaseModel):
    id: str
    question: str
    reference: str
    response: str
    model_name: str
    candidate_latency_sec: float
    candidate_prompt_tokens: int = 0
    candidate_completion_tokens: int = 0
    candidate_total_tokens: int = 0
    candidate_tokens_per_sec: Optional[float] = None
    judge_models: List[str]
    judge_results: List[dict]
    avg_scores: AveragedJudgeScores
