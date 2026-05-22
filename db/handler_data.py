from pydantic import BaseModel, ValidationError
from datetime import datetime
import pandas as pd
from common.deadline import WORK_TIMEZONE, calculate_working_hours_between

BUSINESS_DAY_HOURS = 8


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

    if value.endswith('Z'):
        value = f'{value[:-1]}+00:00'

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=WORK_TIMEZONE)

    return parsed


def compute_lead_time_hours(
    created_at: str | None,
    finished_at: str | None,
) -> float | None:
    if not created_at or not finished_at:
        return None

    created = parse_datetime(created_at)
    finished = parse_datetime(finished_at)

    if finished < created:
        return None

    return calculate_working_hours_between(created, finished)


def compute_business_days(
    created_at: str | None,
    finished_at: str | None,
) -> float | None:
    if not created_at or not finished_at:
        return None

    created = parse_datetime(created_at)
    finished = parse_datetime(finished_at)

    if finished < created:
        return None

    lead_time_hours = compute_lead_time_hours(created_at, finished_at)
    if lead_time_hours is None:
        return None

    return lead_time_hours / BUSINESS_DAY_HOURS


def task_to_document(task: RetrievalTask) -> str:
    return f'''
        Название: {task.name}
        Описание: {task.desc}
        Метка: {task.label}
        Приоритет: {task.prio}
        '''.strip()


def task_to_metadata(task: RetrievalTask) -> dict:
    business_days = compute_business_days(task.created_at, task.finished_at)
    lead_time_hours = compute_lead_time_hours(task.created_at, task.finished_at)

    return {
        'name': task.name,
        'desc': task.desc,
        'labels': task.label,
        'prio': task.prio,
        'created_at': task.created_at or '',
        'finished_at': task.finished_at or '',
        'business_days': (
            round(business_days, 3) if business_days is not None else 0
        ),
        'lead_time_hours': (
            round(lead_time_hours, 2) if lead_time_hours is not None else 0
        ),
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
