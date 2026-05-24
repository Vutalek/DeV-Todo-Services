from pydantic import BaseModel
from datetime import datetime
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

    return calculate_working_hours_between(created, finished) / BUSINESS_DAY_HOURS


def task_metadata_to_business_days(metadata: dict | None) -> float:
    metadata = metadata or {}

    try:
        business_days = compute_business_days(
            metadata.get('created_at'),
            metadata.get('finished_at'),
        )
    except (TypeError, ValueError):
        return 0

    if business_days is None:
        return 0

    return round(business_days, 3)


def task_metadata_to_time_hours(metadata: dict | None) -> int | None:
    metadata = metadata or {}

    try:
        lead_time_hours = compute_lead_time_hours(
            metadata.get('created_at'),
            metadata.get('finished_at'),
        )
    except (TypeError, ValueError):
        return None

    if lead_time_hours is None or lead_time_hours <= 0:
        return None

    return max(1, round(lead_time_hours))


def task_matches_day_filter(
    task: dict,
    min_days: int,
    max_days: int,
) -> bool:
    business_days = task_metadata_to_business_days(task.get('metadata'))
    return min_days <= business_days <= max_days


def task_to_document(task: RetrievalTask) -> str:
    return f'''
        Название: {task.name}
        Описание: {task.desc}
        Метка: {task.label}
        Приоритет: {task.prio}
        '''.strip()


def get_document_field(document: str | None, field_name: str) -> str:
    if not document:
        return ''

    prefix = f'{field_name}:'
    for line in document.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()

    return ''


def task_from_document_metadata(
    document: str | None,
    metadata: dict | None,
) -> RetrievalTask:
    metadata = metadata or {}

    return RetrievalTask(
        name=get_document_field(document, 'Название'),
        desc=get_document_field(document, 'Описание'),
        prio=get_document_field(document, 'Приоритет'),
        label=get_document_field(document, 'Метка'),
        created_at=metadata.get('created_at') or None,
        finished_at=metadata.get('finished_at') or None,
    )


def task_payload_to_fields(task: dict) -> dict:
    metadata = task.get('metadata') or {}
    document = task.get('document') or ''

    return {
        'name': get_document_field(document, 'Название'),
        'desc': get_document_field(document, 'Описание'),
        'priority': get_document_field(document, 'Приоритет'),
        'label': get_document_field(document, 'Метка'),
        'business_days': task_metadata_to_business_days(metadata),
        'time_hours': task_metadata_to_time_hours(metadata),
    }


def task_to_metadata(task: RetrievalTask) -> dict:
    return {
        'created_at': task.created_at or '',
        'finished_at': task.finished_at or '',
    }
