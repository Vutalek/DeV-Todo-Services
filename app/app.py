from db.bm25 import BM25TaskSearch, rrf_fusion
from db.handler_data import (
    RetrievalTask,
    csv_to_tasks,
    load_tasks_to_chroma,
    task_to_document,
    task_to_metadata,
)
from db.embedding import PplxEmbedding
import asyncio
import os
import requests
import sys
import time
from threading import RLock
from pathlib import Path
from uuid import uuid4
from typing import Annotated

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Type
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
import chromadb

from common.deadline import calculate_deadline
from .db import DBFacade
from .auth import AuthHandler

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
    "*",
    "http://localhost",
    "http://localhost:8081",
    "vscode-webview://*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = DBFacade()
auth = AuthHandler()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

MODEL = "deepseek/deepseek-v4-flash"
RERANK_MODEL = "cohere/rerank-4-fast"
RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_TOP_N = 5
BUSINESS_DAY_HOURS = 8

prompt_path = os.path.join(BASE_DIR, 'prompt', 'prompt.txt')

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY or "missing-router-api-key",
    timeout=30.0,
    max_retries=2,
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=ROUTER_API_KEY or "missing-router-api-key",
    timeout=30.0,
    max_retries=2,
)

# Initialize RAG system
CHROMA_PATH = BASE_DIR / 'db' / 'chroma_db'
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
SEED_TASKS_LIMIT = 1000
IMAGE_SEED_TASKS_CSV = BASE_DIR / 'seed' / 'apache_issues.csv'
LEGACY_SEED_TASKS_CSV = CHROMA_PATH / 'apache_issues.csv'

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
trello_cache_refresh_lock = RLock()
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


def normalize_chroma_metadata(metadata: dict | None, document: str | None) -> dict:
    metadata = dict(metadata or {})

    if not metadata.get("created_at") or not metadata.get("finished_at"):
        return metadata

    try:
        task = RetrievalTask(
            name=metadata.get("name") or get_document_field(document, "Название"),
            desc=metadata.get("desc") or get_document_field(document, "Описание"),
            prio=metadata.get("prio") or get_document_field(document, "Приоритет"),
            label=metadata.get("labels") or get_document_field(document, "Метка"),
            created_at=metadata.get("created_at"),
            finished_at=metadata.get("finished_at"),
        )
        recalculated = task_to_metadata(task)
    except Exception:
        return metadata

    metadata["business_days"] = recalculated["business_days"]
    metadata["lead_time_hours"] = recalculated["lead_time_hours"]
    return metadata


def load_tasks_from_chroma():
    """Restore in-memory BM25 state from persistent ChromaDB data."""
    with chroma_lock:
        chroma_data = collection.get(include=["documents", "metadatas"])

    metadata_updates = []

    with state_lock:
        tasks_store.clear()

        for task_id, document, metadata in zip(
            chroma_data.get("ids", []),
            chroma_data.get("documents", []),
            chroma_data.get("metadatas", []),
        ):
            original_metadata = dict(metadata or {})
            metadata = normalize_chroma_metadata(metadata, document)
            if metadata != original_metadata:
                metadata_updates.append((task_id, metadata))

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

    if metadata_updates:
        try:
            with chroma_lock:
                collection.update(
                    ids=[task_id for task_id, _ in metadata_updates],
                    metadatas=[metadata for _, metadata in metadata_updates],
                )
        except Exception as exc:
            print(f"Failed to update ChromaDB metadata: {exc}")


def seed_chroma_if_empty():
    with chroma_lock:
        current_count = collection.count()

    seed_tasks_csv = (
        IMAGE_SEED_TASKS_CSV
        if IMAGE_SEED_TASKS_CSV.exists()
        else LEGACY_SEED_TASKS_CSV
    )

    if current_count > 0 or not seed_tasks_csv.exists():
        return

    try:
        seed_tasks = csv_to_tasks(str(seed_tasks_csv))[:SEED_TASKS_LIMIT]
        if not seed_tasks:
            return

        with chroma_lock:
            if collection.count() == 0:
                load_tasks_to_chroma(collection, seed_tasks)
    except Exception as exc:
        print(f"Failed to seed ChromaDB from {seed_tasks_csv}: {exc}")


def get_trello_data():
    now = time.monotonic()
    with trello_cache_lock:
        cached_data = trello_cache["data"]
        if cached_data is not None and now < trello_cache["expires_at"]:
            return cached_data

    with trello_cache_refresh_lock:
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
        with trello_cache_lock:
            trello_cache["data"] = data
            trello_cache["expires_at"] = time.monotonic() + \
                TRELLO_CACHE_TTL_SECONDS

        return data


def get_chroma_tasks_by_ids(task_ids: list[str]) -> list[dict]:
    if not task_ids:
        return []

    with chroma_lock:
        chroma_data = collection.get(
            ids=task_ids,
            include=["documents", "metadatas"],
        )

    payloads_by_id = {}
    for task_id, document, metadata in zip(
        chroma_data.get("ids", []),
        chroma_data.get("documents", []),
        chroma_data.get("metadatas", []),
    ):
        metadata = normalize_chroma_metadata(metadata, document)
        payloads_by_id[task_id] = {
            "id": task_id,
            "document": document or "",
            "metadata": metadata,
        }

    return [
        payloads_by_id[task_id]
        for task_id in task_ids
        if task_id in payloads_by_id
    ]


def task_payload_to_rerank_document(task: dict) -> str:
    metadata = task.get("metadata") or {}
    document = task.get("document") or ""
    name = metadata.get("name") or get_document_field(document, "Название")
    desc = metadata.get("desc") or get_document_field(document, "Описание")
    priority = metadata.get("prio") or get_document_field(
        document, "Приоритет")
    label = metadata.get("labels") or get_document_field(document, "Метка")

    return "\n".join([
        f"Name: {name}",
        f"Description: {desc}",
        f"Priority: {priority}",
        f"Label: {label}",
        f"Business days: {metadata.get('business_days', 0)}",
        f"Time hours: {task_metadata_to_time_hours(metadata)}",
    ])


def business_days_to_time_hours(business_days) -> int | None:
    try:
        days = float(business_days)
    except (TypeError, ValueError):
        return None

    if days <= 0:
        return None

    return max(1, round(days * BUSINESS_DAY_HOURS))


def task_metadata_to_time_hours(metadata: dict) -> int | None:
    lead_time_hours = metadata.get("lead_time_hours")
    try:
        hours = float(lead_time_hours)
    except (TypeError, ValueError):
        hours = 0

    if hours > 0:
        return max(1, round(hours))

    return business_days_to_time_hours(metadata.get("business_days", 0))


def task_payload_to_search_result(
    task: dict,
    reranker_score: float | None = None,
) -> dict:
    metadata = task.get("metadata") or {}
    document = task.get("document") or ""
    name = metadata.get("name") or get_document_field(document, "Название")
    desc = metadata.get("desc") or get_document_field(
        document, "Описание") or document
    priority = metadata.get("prio") or get_document_field(
        document, "Приоритет")
    label = metadata.get("labels") or get_document_field(document, "Метка")
    business_days = metadata.get("business_days", 0)

    return {
        "name": name,
        "desc": desc,
        "priority": priority,
        "label": label,
        "reranker_score": reranker_score,
        "business_days": business_days,
        "time_hours": task_metadata_to_time_hours(metadata),
    }


def rerank_tasks(
    query: str,
    tasks: list[dict],
    top_n: int = RERANK_TOP_N,
) -> tuple[list[dict], str | None]:
    if not tasks:
        return [], None

    if not ROUTER_API_KEY:
        return [], "Reranker skipped: ROUTER_API_KEY is not set"

    documents = [task_payload_to_rerank_document(task) for task in tasks]
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    headers = {
        "Authorization": f"Bearer {ROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            RERANK_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )
    except requests.Timeout:
        return [], "Reranker fallback: OpenRouter request timed out"
    except requests.RequestException as exc:
        return [], f"Reranker fallback: OpenRouter request failed: {exc}"

    if response.status_code != 200:
        return [], (
            "Reranker fallback: OpenRouter returned "
            f"HTTP {response.status_code}: {response.text}"
        )

    try:
        response_data = response.json()
    except ValueError:
        return [], "Reranker fallback: OpenRouter returned invalid JSON"

    results = response_data.get("results")
    if not isinstance(results, list):
        return [], "Reranker fallback: OpenRouter response has no results list"
    if not results:
        return [], "Reranker fallback: OpenRouter returned no reranked results"

    reranked_tasks = []
    for result in results:
        if not isinstance(result, dict):
            return [], "Reranker fallback: invalid result item"

        if "index" not in result:
            return [], "Reranker fallback: result item has no index"
        if "relevance_score" not in result:
            return [], "Reranker fallback: result item has no relevance_score"

        index = result["index"]
        if not isinstance(index, int) or index < 0 or index >= len(tasks):
            return [], "Reranker fallback: result index is out of range"

        try:
            reranker_score = float(result["relevance_score"])
        except (TypeError, ValueError):
            return [], "Reranker fallback: relevance_score is not numeric"

        reranked_tasks.append(
            task_payload_to_search_result(
                tasks[index],
                reranker_score=reranker_score,
            )
        )

    return reranked_tasks[:top_n], None


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
        time=(int, Field(
            gt=0,
            description="Количество часов от момента создания до дедлайна",
        )),
        roadmap=(str, Field(default="", max_length=1200,
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


seed_chroma_if_empty()
load_tasks_from_chroma()


@app.get("/app/v1/heartbeat")
def heartbeat():
    return {"status": "alive"}

@app.post("/register")
async def register(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    exists = await db.get_user_id(form_data.username)
    if exists != "":
        raise HTTPException(status_code=409, detail="User already exists")

    is_created = await db.create_user(form_data.username, form_data.password)
    if not is_created:
        raise HTTPException(status_code=500, detail="Failed to create user")

    token = auth.create_access_token(data={"sub": form_data.username})
    return Token(access_token=token, token_type="bearer")

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/token")
async def token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    is_authenticated = await auth.authenticate_user(
        form_data.username,
        form_data.password,
    )
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = auth.create_access_token(data={"sub": form_data.username})
    return Token(access_token=token, token_type="bearer")

@app.post("/app/v1/send")
async def sendtask(message: Message, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    if not ROUTER_API_KEY:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": (
                    "OpenRouter task generation failed: "
                    "ROUTER_API_KEY is not set"
                ),
            },
        )

    col_map, lab_map = await asyncio.to_thread(get_trello_data)

    labels_str = ", ".join(lab_map.keys())
    columns_str = ", ".join(col_map.keys())

    Task = create_dynamic_task_model(
        columns=list(col_map.keys()),
        labels=list(lab_map.keys())
    )
    try:
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

        parsed_task = response.choices[0].message.parsed
        if parsed_task is None:
            raise ValueError("OpenRouter returned no parsed task")
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": f"OpenRouter task generation failed: {exc}",
            },
        )

    result = parsed_task.model_dump()
    result["deadline"] = calculate_deadline(parsed_task.time)

    return {"status": "success", "result": result}

@app.post("/app/v1/create_project")
async def create_project(name: str, description: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.create_project(login, name, description)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}

@app.get("/app/v1/projects")
async def get_projects(token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.get_user_projects(login)
    result = [
        {
            "id": str(entry[0]),
            "name": entry[1],
            "description": entry[2]
        }
    for entry in result]
    return {"status": "success", "result": result}

@app.get("/app/v1/project_members")
async def get_projects(project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.get_project_members(login, project)
    result = [
        {
            "id": str(entry[0]),
            "login": entry[1],
            "role": entry[2]
        }
    for entry in result]
    return {"status": "success", "result": result}

@app.get("/app/v1/add_member")
async def add_member(member_login: str, project: str, token: Annotated[str, Depends(oauth2_scheme)]):
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
    token_payload = auth.decode_access_token(token)
    login = token_payload.get("sub", "")
    result = await db.add_member_to_project(login, member_login, project)
    if result:
        return {"status": "success"}
    else:
        return {"status": "fail"}

@app.post("/app/v1/tasks")
def add_or_update_task(task_req: TaskRequest, token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Добавить или обновить задачу в БД RAG-поиска.
    """
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
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
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "message": f"Failed to add task to search index: {str(e)}",
            },
        )


@app.post("/app/v1/search")
def search_tasks(search_req: SearchRequest, token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Поиск задач в БД с использованием гибридного поиска (BM25 + Vector).
    """
    if not auth.verify_token(token):
        return {"status": "error", "message": "Invalid token"}
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

        # Выполнить BM25 поиск
        candidate_count = max(search_req.n_results, RERANK_TOP_N)
        where_days = (search_req.min_days, search_req.max_days)
        bm25_results = bm25_snapshot.search(
            query_text,
            n_results=candidate_count,
            where_days=where_days,
        )

        # Выполнить vector поиск через ChromaDB
        try:
            with chroma_lock:
                vector_results = collection.query(
                    query_texts=[query_text],
                    n_results=candidate_count,
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
            vector_results, bm25_results, k=60, top_n=candidate_count)

        candidate_ids = []
        seen_task_ids = set()

        for result in hybrid_results:
            task_id = result['id']
            if task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            candidate_ids.append(task_id)

        candidate_tasks = get_chroma_tasks_by_ids(candidate_ids)
        reranked_results, warning = rerank_tasks(
            query_text,
            candidate_tasks,
            top_n=RERANK_TOP_N,
        )

        if warning is not None:
            search_results = [
                task_payload_to_search_result(task)
                for task in candidate_tasks[:RERANK_TOP_N]
            ]
        else:
            search_results = reranked_results

        response = {
            "status": "success",
            "query": query_text,
            "results_count": len(search_results),
            "results": search_results
        }

        if warning is not None:
            response["warning"] = warning

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }
