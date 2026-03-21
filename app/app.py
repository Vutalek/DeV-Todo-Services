import os

import openai
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

client = openai.OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://ai.api.cloud.yandex.net/v1",
    project="b1gccpjnou3q4l9pegs9"
)

# body
class Message(BaseModel):
    text: str

@app.post("/app/v1/send")
def sendtask(message: Message):
    response = client.responses.create(
        prompt={
            "id": "fvtalhr0tvl0hd6mlpgg",
        },
        input=message.text,
    )
    return {"status": "success", "result": response}