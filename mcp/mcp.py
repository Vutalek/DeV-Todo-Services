import os
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from common.deadline import calculate_deadline

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

TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID")

# тело запроса
class Card(BaseModel):
    name: str
    desc: str
    prio: int
    time: int = Field(gt=0)


@app.post("/mcp/v1/sendtask")
def sendtask(card: Card):
    missing_env = [
        name for name, value in (
            ("TRELLO_API_KEY", TRELLO_API_KEY),
            ("TRELLO_TOKEN", TRELLO_TOKEN),
            ("TRELLO_LIST_ID", TRELLO_LIST_ID),
        )
        if not value
    ]
    if missing_env:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": (
                    "Trello card creation failed: missing env "
                    f"{', '.join(missing_env)}"
                ),
            },
        )

    deadline_iso = calculate_deadline(card.time)
    extended_desc = (
        f"{card.desc}\n"
        f"Priority: {card.prio}\n"
        f"Time: {card.time}h\n"
        f"Deadline: {deadline_iso}"
    )

    try:
        response = requests.post(
            "https://api.trello.com/1/cards",
            params={
                "key": TRELLO_API_KEY,
                "token": TRELLO_TOKEN,
                "idList": TRELLO_LIST_ID,
                "name": card.name,
                "desc": extended_desc,
                "due": deadline_iso,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": f"Trello card creation failed: {exc}",
            },
        )

    return {"status": "success", "result": result}
