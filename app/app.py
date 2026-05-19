from db.bm25 import BM25TaskSearch, rrf_fusion
from db.handler_data import RetrievalTask, task_to_document, task_to_metadata
from db.embedding import PplxEmbedding
import asyncio
import os
import requests
import sys
import time
from threading import RLock
from pathlib import Path
from uuid import uuid4
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb

# Add parent directory to path for db imports
APP_DIR = Path(__file__).parent.resolve()
BASE_DIR = APP_DIR.parent
sys.path.insert(0, str(BASE_DIR))


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
    api_key=ROUTER_API_KEY,
    timeout=30.0,
    max_retries=2,
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY,
    timeout=30.0,
    max_retries=2,
)

# Initialize RAG system
CHROMA_PATH = BASE_DIR / 'db' / 'chroma_db'
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

client_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

ef = PplxEmbedding(model='perplexity/pplx-embed-v1-0.6b', client=client)
collection = client_chroma.get_or_create_collection(
    name='TODO', embedding_function=ef)

# Store for BM25 search
tasks_store: dict[str, RetrievalTask] = {}
bm25_index = None
state_lock = RLock()
chroma_lock = RLock()
trello_cache_lock = RLock()
trello_cache: dict[str, object] = {
    "expires_at": 0.0,
    "data": None,
}
TRELLO_CACHE_TTL_SECONDS = 300


def update_bm25_index_locked():
    """Update BM25 index from current tasks"""
    global bm25_index
    if tasks_store:
        ids = list(tasks_store.keys())
        documents = [task_to_document(task) for task in tasks_store.values()]
        metadatas = [task_to_metadata(task) for task in tasks_store.values()]
        bm25_index = BM25TaskSearch(
            ids=ids, documents=documents, metadatas=metadatas)
    else:
        bm25_index = None


def get_document_field(document: str | None, field_name: str) -> str:
    if not document:
        return ""

    prefix = f"{field_name}:"
    for line in document.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()

    return ""


def load_tasks_from_chroma():
    """Restore in-memory BM25 state from persistent ChromaDB data."""
    with chroma_lock:
        chroma_data = collection.get(include=["documents", "metadatas"])

    with state_lock:
        tasks_store.clear()

        for task_id, document, metadata in zip(
            chroma_data.get("ids", []),
            chroma_data.get("documents", []),
            chroma_data.get("metadatas", []),
        ):
            metadata = metadata or {}
            tasks_store[task_id] = RetrievalTask(
                name=metadata.get("name") or get_document_field(
                    document, "Название"),
                desc=metadata.get("desc") or get_document_field(
                    document, "Описание"),
                prio=metadata.get("prio") or get_document_field(
                    document, "Приоритет"),
                label=metadata.get("labels") or get_document_field(
                    document, "Метка"),
                created_at=metadata.get("created_at") or None,
                finished_at=metadata.get("finished_at") or None,
            )

        update_bm25_index_locked()


def get_trello_data():
    now = time.monotonic()
    with trello_cache_lock:
        cached_data = trello_cache["data"]
        if cached_data is not None and now < trello_cache["expires_at"]:
            return cached_data

        lists_url = f"https://api.trello.com/1/boards/{BOARD_ID}/lists"
        labels_url = f"https://api.trello.com/1/boards/{BOARD_ID}/labels"
        query = {'key': TRELLO_KEY, 'token': TRELLO_TOKEN}

        try:
            lists_response = requests.get(lists_url, params=query, timeout=15)
            labels_response = requests.get(
                labels_url, params=query, timeout=15)
            lists_response.raise_for_status()
            labels_response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Trello data: {exc}",
            ) from exc

        lists_data = lists_response.json()
        labels_data = labels_response.json()

        col_map = {l['name']: l['id'] for l in lists_data}
        lab_map = {lb['name']: lb['id'] for lb in labels_data if lb['name']}

        data = (col_map, lab_map)
        trello_cache["data"] = data
        trello_cache["expires_at"] = time.monotonic() + \
            TRELLO_CACHE_TTL_SECONDS

        return data


def create_dynamic_task_model(columns: list, labels: list) -> Type[BaseModel]:
    TrelloColumns = Literal[tuple(columns)] if columns else str
    TrelloLabels = Literal[tuple(labels)] if labels else str

    return create_model(
        'Task',
        name=(str, Field(max_length=100,
              description="Короткое название: действие + объект")),
        desc=(str, Field(default="", max_length=400,
                         description="2-4 коротких предложения: сейчас ... нужно ...")),
        label=(List[TrelloLabels], Field(description="Метки задачи")),
        prio=(int, Field(ge=1, le=5, description="Приоритет от 1 до 5")),
        time=(int, Field(gt=0, description="Время в часах")),
        roadmap=(str, Field(default="", max_length=100,
                            description="2-5 конкретных шагов без повторения описания")),
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
    created_at: str | None = Field(
        default=None, description="Дата создания (ISO format)")
    finished_at: str | None = Field(
        default=None, description="Дата завершения (ISO format)")


class SearchRequest(BaseModel):
    query: str = Field(description="Поисковый запрос")
    n_results: int = Field(default=10, ge=1, le=50,
                           description="Количество результатов")
    min_days: int = Field(
        default=0, ge=0, description="Минимальное количество дней")
    max_days: int = Field(
        default=365, ge=0, description="Максимальное количество дней")


load_tasks_from_chroma()


@app.get("/app/v1/heartbeat")
def heartbeat():
    return {"status": "alive"}


@app.post("/app/v1/send")
async def sendtask(message: Message):
    col_map, lab_map = await asyncio.to_thread(get_trello_data)

    labels_str = ", ".join(lab_map.keys())
    columns_str = ", ".join(col_map.keys())

    Task = create_dynamic_task_model(
        columns=list(col_map.keys()),
        labels=list(lab_map.keys())
    )
    response = await async_client.chat.completions.parse(
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
        temperature=0.1,
        top_p=0.8,
        frequency_penalty=0.3,
        max_tokens=1500,
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

        # Создать документ и метаданные
        doc = task_to_document(task)
        metadata = task_to_metadata(task)

        task_id = f"task_{uuid4().hex}"

        with chroma_lock:
            collection.add(
                ids=[task_id],
                documents=[doc],
                metadatas=[metadata],
            )

        with state_lock:
            tasks_store[task_id] = task
            update_bm25_index_locked()

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
        with state_lock:
            if not tasks_store or bm25_index is None:
                return {
                    "status": "warning",
                    "message": "No tasks in database",
                    "results": []
                }

            # Подготовить query
            query_text = search_req.query
            bm25_snapshot = bm25_index
            tasks_snapshot = dict(tasks_store)

        # Выполнить BM25 поиск
        where_days = (search_req.min_days, search_req.max_days)
        bm25_results = bm25_snapshot.search(
            query_text,
            n_results=search_req.n_results,
            where_days=where_days,
        )

        # Выполнить vector поиск через ChromaDB
        try:
            with chroma_lock:
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
        hybrid_results = rrf_fusion(
            vector_results, bm25_results, k=60, top_n=search_req.n_results)

        # Обогатить результаты метаданными (исключая дубликаты)
        enriched_results = []
        seen_task_ids = set()

        for result in hybrid_results:
            # Пропустить если уже видели эту задачу
            task_id = result['id']
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)

            # Получить задачу из хранилища
            task = tasks_snapshot.get(task_id)
            if task is not None:
                task_metadata = task_to_metadata(task)
                enriched_results.append({
                    "task_id": task_id,
                    "name": task.name,
                    "business_days": task_metadata["business_days"],
                    "description": task.desc,
                    "priority": task.prio,
                    "label": task.label,
                    "created_at": task.created_at or "not set",
                    "finished_at": task.finished_at or "not set",
                    "hybrid_score": float(result.get('hybrid_score', 0)),
                    "bm25_score": float(next((r['bm25_score'] for r in bm25_results if r['id'] == task_id), 0)),
                })

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
