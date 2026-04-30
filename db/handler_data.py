from pydantic import BaseModel, ValidationError
from typing import Literal
from datetime import datetime
import pandas as pd
import numpy as np


class RetrievalTask(BaseModel):
    name: str
    desc: str
    prio: str
    label: str
    created_at: str | None = None
    finished_at: str | None = None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f%z')


def compute_lead_time_hours(created_at: str | None, finished_at: str | None) -> float | None:
    if not created_at or not finished_at:
        return None

    created = parse_datetime(created_at)
    finished = parse_datetime(finished_at)

    start_date = np.datetime64(created.date())
    end_date = np.datetime64(finished.date())

    days = int(np.busday_count(start_date, end_date))
    if days == 0:
        return (finished - created).total_seconds() / 86400

    return days


def task_to_document(task: RetrievalTask) -> str:
    return f'''
        Название: {task.name}
        Описание: {task.desc}
        Метка: {task.label}
        Приоритет: {task.prio}
        '''.strip()


def task_to_metadata(task: RetrievalTask) -> dict:
    days = compute_lead_time_hours(task.created_at, task.finished_at)

    return {
        'desc': task.desc,
        'labels': task.label,
        'prio': task.prio,
        'created_at': task.created_at,
        'finished_at': task.finished_at,
        'business_days': days,
    }


def load_tasks_to_chroma(collection, tasks: list[RetrievalTask]):
    ids = []
    documents = []
    metadatas = []

    for i, task in enumerate(tasks):
        ids.append(f'task_{i}')
        documents.append(task_to_document(task))
        metadatas.append(task_to_metadata(task))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def csv_to_tasks(file_path: str) -> list[RetrievalTask]:

    df = pd.read_csv(file_path)
    df = df.drop(columns=['url'])

    df = df.dropna()
    df = df.drop_duplicates()

    tasks = []
    for row in df.iterrows():

        task = {
            'name': (row[1]['name']),
            'desc': (row[1]['desc']),
            'prio': (row[1]['priority']),
            'label': (row[1]['issue_type']),
            'created_at': (row[1]['created']),
            'finished_at': (row[1]['resolved']),
        }

        try:
            task = RetrievalTask.model_validate(task)
            tasks.append(task)
        except ValidationError as e:
            print('-' * 250)
            print(f'[ERROR] {e.errors()}')

    return tasks
