from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


WORK_TIMEZONE = ZoneInfo("Europe/Moscow")
WORKDAY_START_HOUR = 10
WORKDAY_END_HOUR = 18


def to_work_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=WORK_TIMEZONE)

    return value.astimezone(WORK_TIMEZONE)


def move_to_work_time(value: datetime) -> datetime:
    current = to_work_timezone(value)

    if current.weekday() >= 5:
        days_until_monday = 7 - current.weekday()
        current = current + timedelta(days=days_until_monday)
        return current.replace(
            hour=WORKDAY_START_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )

    workday_start = current.replace(
        hour=WORKDAY_START_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    workday_end = current.replace(
        hour=WORKDAY_END_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )

    if current < workday_start:
        return workday_start

    if current >= workday_end:
        current = current + timedelta(days=1)
        while current.weekday() >= 5:
            current = current + timedelta(days=1)
        return current.replace(
            hour=WORKDAY_START_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )

    return current


def add_working_hours(start: datetime, hours: int) -> datetime:
    current = move_to_work_time(start)
    remaining = timedelta(hours=hours)

    while remaining > timedelta(0):
        workday_end = current.replace(
            hour=WORKDAY_END_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        available = workday_end - current

        if remaining <= available:
            return current + remaining

        remaining -= available
        current = move_to_work_time(workday_end + timedelta(seconds=1))

    return current


def calculate_working_hours_between(start: datetime, end: datetime) -> float:
    current = move_to_work_time(start)
    finish = to_work_timezone(end)

    if finish <= current:
        return 0.0

    worked = timedelta(0)

    while current < finish:
        workday_end = current.replace(
            hour=WORKDAY_END_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        segment_end = min(workday_end, finish)

        if segment_end > current:
            worked += segment_end - current

        current = move_to_work_time(workday_end + timedelta(seconds=1))

    return worked.total_seconds() / 3600


def calculate_deadline(time_hours: int, created_at: datetime | None = None) -> str:
    created = created_at or datetime.now(WORK_TIMEZONE)
    deadline = add_working_hours(created, time_hours)
    return deadline.astimezone(timezone.utc).isoformat()
