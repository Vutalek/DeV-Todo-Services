import os
import requests
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Type
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

TRELLO_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BOARD_ID = os.getenv("TRELLO_BOARD_ID")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")


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

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
prompt_path = os.path.join(BASE_DIR, 'prompt', 'prompt.txt')

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY
)


def get_trello_data():
    lists_url = f"https://api.trello.com/1/boards/{BOARD_ID}/lists"
    labels_url = f"https://api.trello.com/1/boards/{BOARD_ID}/labels"
    query = {'key': TRELLO_KEY, 'token': TRELLO_TOKEN}

    lists_data = requests.get(lists_url, params=query).json()
    labels_data = requests.get(labels_url, params=query).json()

    col_map = {l['name']: l['id'] for l in lists_data}
    lab_map = {lb['name']: lb['id'] for lb in labels_data if lb['name']}

    return col_map, lab_map


def create_dynamic_task_model(columns: list, labels: list) -> Type[BaseModel]:
    TrelloColumns = Literal[tuple(columns)] if columns else str
    TrelloLabels = Literal[tuple(labels)] if labels else str

    return create_model(
        'Task',
        name=(str, Field(description="Название задачи")),
        desc=(str, Field(default="", max_length=250,
                         description="Краткое описание (не более 250 токенов)")),
        label=(List[TrelloLabels], Field(description="Метки задачи")),
        prio=(int, Field(ge=1, le=5, description="Приоритет от 1 до 5")),
        time=(int, Field(gt=0, description="Время в часах")),
        roadmap=(str, Field(default="", description="Roadmap для задачи")),
        column=(TrelloColumns, Field(
            description='Колонка в которой будет находиться задача')),
        __base__=BaseModel
    )


class Message(BaseModel):
    text: str


@app.post("/app/v1/send")
def sendtask(message: Message):
    col_map, lab_map = get_trello_data()

    labels_str = ", ".join(lab_map.keys())
    columns_str = ", ".join(col_map.keys())

    Task = create_dynamic_task_model(
        columns=list(col_map.keys()),
        labels=list(lab_map.keys())
    )
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system", "content": prompt.format(
                    labels_list=labels_str,
                    columns_list=columns_str
                )
            },
            {
                "role": "user", "content": message.text
            },
        ],
        temperature=0.3,
        response_format=Task,
    )

    return {"status": "success", "result": response.choices[0].message.parsed}
