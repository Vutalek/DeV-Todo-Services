import os

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

# body
class Card(BaseModel):
    name: str
    desc: str
    prio: int
    time: int

@app.post("/mcp/v1/sendtask")
def sendtask(card: Card):
    extended_desc = f"{card.desc}\nPriority: {card.prio}\nTime: {card.time}h"
    result = trello.cards.new(name=card.name, desc=extended_desc, idList=os.getenv("TRELLO_LIST_ID"))
    return {"status": "success", "result": result}