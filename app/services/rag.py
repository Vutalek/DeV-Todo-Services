from threading import RLock
from uuid import uuid4

import requests

from rag.bm25 import BM25TaskSearch, rrf_fusion
from rag.handler_data import (
    RetrievalTask,
    task_from_document_metadata,
    task_payload_to_fields,
    task_to_document,
    task_to_metadata,
)
from app import config
from app.models import SearchRequest

tasks_store: dict[str, RetrievalTask] = {}
task_metadata_store: dict[str, dict] = {}
bm25_index = None
state_lock = RLock()
chroma_lock = RLock()


def update_bm25_index_locked():
    global bm25_index
    if tasks_store:
        ids = list(tasks_store.keys())
        documents = [task_to_document(tasks_store[task_id]) for task_id in ids]
        metadatas = [task_metadata_store.get(task_id, {}) for task_id in ids]
        bm25_index = BM25TaskSearch(
            ids=ids, documents=documents, metadatas=metadatas)
    else:
        bm25_index = None


def load_tasks_from_chroma():
    with chroma_lock:
        chroma_data = config.collection.get(include=["documents", "metadatas"])

    with state_lock:
        tasks_store.clear()
        task_metadata_store.clear()

        for task_id, document, metadata in zip(
            chroma_data.get("ids", []),
            chroma_data.get("documents", []),
            chroma_data.get("metadatas", []),
        ):
            metadata = metadata or {}
            tasks_store[task_id] = task_from_document_metadata(
                document,
                metadata,
            )
            task_metadata_store[task_id] = metadata

        update_bm25_index_locked()


def get_chroma_tasks_by_ids(task_ids: list[str]) -> list[dict]:
    if not task_ids:
        return []

    with chroma_lock:
        chroma_data = config.collection.get(
            ids=task_ids,
            include=["documents", "metadatas"],
        )

    payloads_by_id = {}
    for task_id, document, metadata in zip(
        chroma_data.get("ids", []),
        chroma_data.get("documents", []),
        chroma_data.get("metadatas", []),
    ):
        payloads_by_id[task_id] = {
            "id": task_id,
            "document": document or "",
            "metadata": metadata or {},
        }

    return [
        payloads_by_id[task_id]
        for task_id in task_ids
        if task_id in payloads_by_id
    ]


def task_payload_to_rerank_document(task: dict) -> str:
    fields = task_payload_to_fields(task)

    return "\n".join([
        f"Name: {fields['name']}",
        f"Description: {fields['desc']}",
        f"Priority: {fields['priority']}",
        f"Label: {fields['label']}",
        f"Business days: {fields['business_days']}",
        f"Time hours: {fields['time_hours']}",
    ])


def task_payload_to_search_result(
    task: dict,
    reranker_score: float | None = None,
) -> dict:
    fields = task_payload_to_fields(task)

    return {
        "name": fields["name"],
        "desc": fields["desc"],
        "priority": fields["priority"],
        "label": fields["label"],
        "reranker_score": reranker_score,
        "business_days": fields["business_days"],
        "time_hours": fields["time_hours"],
    }


def format_similar_tasks_context(similar_tasks: list[dict]) -> str:
    if not similar_tasks:
        return ""

    fields = (
        "name",
        "desc",
        "priority",
        "label",
        "reranker_score",
        "business_days",
        "time_hours",
    )
    task_blocks = []
    for index, task in enumerate(similar_tasks, start=1):
        lines = [f"{index}. Похожая задача"]
        for field in fields:
            value = task.get(field)
            lines.append(f"   {field}: {'' if value is None else value}")

        task_blocks.append("\n".join(lines))

    return "\n\n".join(task_blocks)


def build_message_text_with_similar_tasks(
    original_text: str,
    similar_tasks: list[dict],
) -> str:
    similar_tasks_context = format_similar_tasks_context(similar_tasks)
    if not similar_tasks_context:
        return original_text

    return "\n".join([
        "Новая задача:",
        original_text.strip(),
        "",
        "Самые похожие задачи из истории:",
        similar_tasks_context,
        ""
    ])


def rerank_tasks(
    query: str,
    tasks: list[dict],
    top_n: int = config.RERANK_TOP_N,
) -> tuple[list[dict], str | None]:
    if not tasks:
        return [], None

    if not config.ROUTER_API_KEY:
        return [], "Reranker skipped: ROUTER_API_KEY is not set"

    documents = [task_payload_to_rerank_document(task) for task in tasks]
    payload = {
        "model": config.RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    headers = {
        "Authorization": f"Bearer {config.ROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            config.RERANK_URL,
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


def find_relevant_tasks(search_req: SearchRequest) -> dict:
    with state_lock:
        if not tasks_store or bm25_index is None:
            return {
                "status": "warning",
                "message": "No tasks in database",
                "results": []
            }

        query_text = search_req.query
        bm25_snapshot = bm25_index

    candidate_count = max(search_req.n_results, config.RERANK_TOP_N)
    where_days = (search_req.min_days, search_req.max_days)
    bm25_results = bm25_snapshot.search(
        query_text,
        n_results=candidate_count,
        where_days=where_days,
    )

    try:
        with chroma_lock:
            vector_results = config.collection.query(
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
        top_n=config.RERANK_TOP_N,
    )

    if warning is not None:
        search_results = [
            task_payload_to_search_result(task)
            for task in candidate_tasks[:config.RERANK_TOP_N]
        ]
    else:
        search_results = [
            task for task in reranked_results
            if task.get("reranker_score") is not None and task["reranker_score"] > 0.5
        ]

    response = {
        "status": "success",
        "query": query_text,
        "results_count": len(search_results),
        "results": search_results
    }

    if warning is not None:
        response["warning"] = warning

    return response


def get_similar_tasks_for_message(message_text: str) -> list[dict]:
    try:
        search_response = find_relevant_tasks(
            SearchRequest(
                query=message_text,
                n_results=config.SIMILAR_TASKS_CONTEXT_TOP_N,
            )
        )
    except Exception as exc:
        print(f"Similar tasks search failed: {exc}")
        return []

    if search_response.get("status") != "success":
        return []

    return search_response.get("results", [])[:config.SIMILAR_TASKS_CONTEXT_TOP_N]


def add_task_to_index(task: RetrievalTask) -> str:
    doc = task_to_document(task)
    metadata = task_to_metadata(task)
    task_id = f"task_{uuid4().hex}"

    with chroma_lock:
        config.collection.add(
            ids=[task_id],
            documents=[doc],
            metadatas=[metadata],
        )

    try:
        with state_lock:
            tasks_store[task_id] = task
            task_metadata_store[task_id] = metadata
            update_bm25_index_locked()
    except Exception:
        with state_lock:
            tasks_store.pop(task_id, None)
            task_metadata_store.pop(task_id, None)
            try:
                update_bm25_index_locked()
            except Exception as rollback_exc:
                print(
                    f"Failed to rebuild BM25 after rollback: {rollback_exc}")

        try:
            with chroma_lock:
                config.collection.delete(ids=[task_id])
        except Exception as rollback_exc:
            print(
                f"Failed to rollback ChromaDB task {task_id}: {rollback_exc}")

        raise

    return task_id
