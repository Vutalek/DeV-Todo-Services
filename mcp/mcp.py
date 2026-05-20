import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trello import TrelloApi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trello = TrelloApi(os.getenv("TRELLO_API_KEY"), os.getenv("TRELLO_TOKEN"))
WORK_TIMEZONE = ZoneInfo("Europe/Moscow")
WORKDAY_START_HOUR = 10
WORKDAY_END_HOUR = 18

# body
class Card(BaseModel):
    name: str
    desc: str
    prio: int
    time: int


def move_to_work_time(value: datetime) -> datetime:
    current = value.astimezone(WORK_TIMEZONE)

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


def calculate_deadline(time_hours: int, created_at: datetime | None = None) -> str:
    created = created_at or datetime.now(WORK_TIMEZONE)
    deadline = add_working_hours(created, time_hours)
    return deadline.astimezone(timezone.utc).isoformat()


@app.post("/mcp/v1/sendtask")
def sendtask(card: Card):
    deadline_iso = calculate_deadline(card.time)
    extended_desc = (
        f"{card.desc}\n"
        f"Priority: {card.prio}\n"
        f"Time: {card.time}h\n"
        f"Deadline: {deadline_iso}"
    )
    result = trello.cards.new(
        name=card.name,
        desc=extended_desc,
        idList=os.getenv("TRELLO_LIST_ID"),
        due=deadline_iso,
    )
    return {"status": "success", "result": result}
