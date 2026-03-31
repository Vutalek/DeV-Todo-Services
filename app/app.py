import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
prompt_path = os.path.join(BASE_DIR, 'prompt', 'prompt.txt')

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("ROUTER_API_KEY")
)


class Task(BaseModel):
    name: str = Field(description="Название задачи")
    desc: str = Field(default="", max_length=250,
                      description="Краткое описание (не более 200 токенов)")
    label: List[Literal['refactor', 'feature', 'bug', 'security',
                        'improvement', 'technical-debt']] = Field(default=['bug'], description="Метки задачи")
    prio: int = Field(ge=1, le=5, description="Приоритет от 1 до 5")
    time: int = Field(gt=0, description="Время в часах")
    roadmap: str = Field(default="", description="Roadmap для задачи")
    column: str = Field(default=['В'])


class Message(BaseModel):
    text: str


@app.post("/app/v1/send")
def sendtask(message: Message):
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system", "content": prompt
            },
            {
                "role": "user", "content": message.text
            },
        ],
        temperature=0.3,
        response_format=Task,
    )
    return {"status": "success", "result": response.choices[0].message.parsed}
