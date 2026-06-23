from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Type


class Message(BaseModel):
    text: str


class TaskRequest(BaseModel):
    name: str = Field(description="Название задачи")
    desc: str = Field(default="", description="Описание задачи")
    prio: str = Field(description="Приоритет задачи")
    label: str = Field(description="Метка/тип задачи")
    created_at: str | None = Field(
        default=None, description="Дата создания (ISO format)")
    finished_at: str | None = Field(
        default=None, description="Дата завершения (ISO format)")


class SearchRequest(BaseModel):
    query: str = Field(description="Поисковый запрос")
    n_results: int = Field(default=10, ge=1, le=50,
                           description="Количество результатов")
    min_days: int = Field(
        default=0, ge=0, description="Минимальное количество дней")
    max_days: int = Field(
        default=365, ge=0, description="Максимальное количество дней")


class Token(BaseModel):
    access_token: str
    token_type: str


def create_dynamic_task_model(columns: list, labels: list) -> Type[BaseModel]:
    TrelloColumns = Literal[tuple(columns)] if columns else str
    TrelloLabels = Literal[tuple(labels)] if labels else str

    return create_model(
        'Task',
        name=(str, Field(max_length=100,
              description="Короткое название: действие + объект")),
        desc=(str, Field(default="", max_length=400,
                         description="2-4 коротких предложения: сейчас ... нужно ...")),
        label=(List[TrelloLabels], Field(description="Метки задачи")),
        prio=(int, Field(ge=1, le=5, description="Приоритет от 1 до 5")),
        time=(int, Field(
            gt=0,
            description="Количество часов от момента создания до дедлайна",
        )),
        roadmap=(str, Field(default="", max_length=1200,
                            description="2-5 конкретных шагов без повторения описания")),
        column=(TrelloColumns, Field(
            description='Колонка в которой будет находиться задача')),
        __base__=BaseModel
    )
