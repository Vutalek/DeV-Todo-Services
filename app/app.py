import os
import requests
import sys
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Type
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb

# Add parent directory to path for db imports
APP_DIR = Path(__file__).parent.resolve()
BASE_DIR = APP_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from db.embedding import PplxEmbedding
from db.handler_data import RetrievalTask, task_to_document, task_to_metadata
from db.bm25 import BM25TaskSearch, tasks_to_records, rrf_fusion

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

MODEL = "deepseek/deepseek-v3.2"

prompt_path = os.path.join(BASE_DIR, 'prompt', 'prompt.txt')

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY
)

# Initialize RAG system
CHROMA_PATH = BASE_DIR / 'db' / 'chroma_db'
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

client_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

ef = PplxEmbedding(model='perplexity/pplx-embed-v1-0.6b', client=client)
collection = client_chroma.get_or_create_collection(
    name='TODO', embedding_function=ef)

# Store for BM25 search
tasks_store = []
bm25_index = None


def update_bm25_index():
    """Update BM25 index from current tasks"""
    global bm25_index
    if tasks_store:
        ids, documents, metadatas = tasks_to_records(tasks_store)
        bm25_index = BM25TaskSearch(ids=ids, documents=documents, metadatas=metadatas)
    else:
        bm25_index = None


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
        desc=(str, Field(default="", max_length=750,
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


class TaskRequest(BaseModel):
    name: str = Field(description="Название задачи")
    desc: str = Field(default="", description="Описание задачи")
    prio: str = Field(description="Приоритет задачи")
    label: str = Field(description="Метка/тип задачи")
    created_at: str | None = Field(default=None, description="Дата создания (ISO format)")
    finished_at: str | None = Field(default=None, description="Дата завершения (ISO format)")


class SearchRequest(BaseModel):
    query: str = Field(description="Поисковый запрос")
    n_results: int = Field(default=10, ge=1, le=50, description="Количество результатов")
    min_days: int = Field(default=0, ge=0, description="Минимальное количество дней")
    max_days: int = Field(default=365, ge=0, description="Максимальное количество дней")


@app.get("/app/v1/heartbeat")
def heartbeat():
    return {"status": "alive"}


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


@app.post("/app/v1/tasks")
def add_or_update_task(task_req: TaskRequest):
    """
    Добавить или обновить задачу в БД RAG-поиска.
    """
    try:
        # Создать объект RetrievalTask
        task = RetrievalTask(
            name=task_req.name,
            desc=task_req.desc,
            prio=task_req.prio,
            label=task_req.label,
            created_at=task_req.created_at,
            finished_at=task_req.finished_at
        )
        
        # Добавить в tasks_store
        tasks_store.append(task)
        
        # Создать документ и метаданные
        doc = task_to_document(task)
        metadata = task_to_metadata(task)
        
        # Добавить в ChromaDB
        task_id = f"task_{len(tasks_store) - 1}_{task.name[:20]}"
        collection.add(
            ids=[task_id],
            documents=[doc],
            metadatas=[metadata],
        )
        
        # Обновить BM25 индекс
        update_bm25_index()
        
        return {
            "status": "success",
            "message": f"Task '{task.name}' added successfully",
            "task_id": task_id
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to add task: {str(e)}"
        }


@app.post("/app/v1/search")
def search_tasks(search_req: SearchRequest):
    """
    Поиск задач в БД с использованием гибридного поиска (BM25 + Vector).
    """
    try:
        if not tasks_store or bm25_index is None:
            return {
                "status": "warning",
                "message": "No tasks in database",
                "results": []
            }
        
        # Подготовить query
        query_text = search_req.query
        
        # Выполнить BM25 поиск
        where_days = (search_req.min_days, search_req.max_days)
        bm25_results = bm25_index.search(query_text, n_results=search_req.n_results, where_days=where_days)
        
        # Выполнить vector поиск через ChromaDB
        try:
            vector_results = collection.query(
                query_texts=[query_text],
                n_results=search_req.n_results,
                include=['distances'],
                where={
                    '$and': [
                        {'business_days': {'$gte': search_req.min_days}},
                        {'business_days': {'$lte': search_req.max_days}},
                    ]
                }
            )
        except Exception as e:
            print(f"Vector search error: {e}")
            vector_results = {'ids': [[]]}
        
        # Гибридный поиск (RRF fusion)
        hybrid_results = rrf_fusion(vector_results, bm25_results, k=60, top_n=search_req.n_results)
        
        # Обогатить результаты метаданными
        enriched_results = []
        for result in hybrid_results:
            # Найти задачу по ID
            for i, task in enumerate(tasks_store):
                if result['id'] == f"task_{i}_{task.name[:20]}" or result['id'].startswith(f"task_{i}"):
                    enriched_results.append({
                        "task_id": result['id'],
                        "name": task.name,
                        "description": task.desc,
                        "priority": task.prio,
                        "label": task.label,
                        "created_at": task.created_at,
                        "finished_at": task.finished_at,
                        "hybrid_score": result.get('hybrid_score', 0),
                        "bm25_score": next((r['bm25_score'] for r in bm25_results if r['id'] == result['id']), 0),
                    })
                    break
        
        return {
            "status": "success",
            "query": query_text,
            "results_count": len(enriched_results),
            "results": enriched_results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }
