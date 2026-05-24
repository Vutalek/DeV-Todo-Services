from zoneinfo import ZoneInfo
from fastapi.testclient import TestClient

from db.handler_data import parse_datetime
from common.deadline import WORK_TIMEZONE
import app.app as app_mod


def test_parse_datetime_timezone_safety():
    # Наивное время должно получить часовой пояс Europe/Moscow
    dt_naive = parse_datetime("2026-05-22T10:00:00")
    assert dt_naive.tzinfo == WORK_TIMEZONE
    assert dt_naive.hour == 10

    # Время с указанным часовым поясом должно сохранить его
    dt_aware = parse_datetime("2026-05-22T10:00:00+03:00")
    assert dt_aware.tzinfo == ZoneInfo("Europe/Moscow") or dt_aware.utcoffset().total_seconds() == 10800
    assert dt_aware.hour == 10

    # Время в UTC/Z должно сохранить свой часовой пояс
    dt_utc = parse_datetime("2026-05-22T10:00:00Z")
    assert dt_utc.utcoffset().total_seconds() == 0


def test_add_or_update_task_rollback_on_failure(monkeypatch):
    client = TestClient(app_mod.app)

    # 1. Имитируем вызов удаления в ChromaDB для проверки
    deleted_ids = []
    def mock_delete(ids=None):
        if ids:
            deleted_ids.extend(ids)

    monkeypatch.setattr(app_mod.collection, "delete", mock_delete)
    monkeypatch.setattr(app_mod.collection, "add", lambda **kwargs: None)

    # 2. Имитируем ошибку в работе update_bm25_index_locked
    def mock_update_fail():
        raise RuntimeError("Rebuilding BM25 failed intentionally")

    monkeypatch.setattr(app_mod, "update_bm25_index_locked", mock_update_fail)

    task_payload = {
        "name": "Rollback Test Task",
        "desc": "Testing transactional rollback",
        "prio": "High",
        "label": "Bug",
        "created_at": "2026-05-22T10:00:00Z",
        "finished_at": "2026-05-22T12:00:00Z",
    }

    # 3. Вызываем API и убеждаемся, что возвращается код 502
    response = client.post("/app/v1/add_task", json=task_payload)
    assert response.status_code == 502
    assert "Rebuilding BM25 failed intentionally" in response.json()["message"]

    # 4. Проверяем, что был вызван метод удаления в ChromaDB
    assert len(deleted_ids) == 1

    # 5. Проверяем, что задача НЕ осталась во временном хранилище tasks_store
    task_id = deleted_ids[0]
    assert task_id not in app_mod.tasks_store
    assert task_id not in app_mod.task_metadata_store


def test_search_tasks_filters_by_relevance_score(monkeypatch):
    client = TestClient(app_mod.app)

    # 1. Заполняем tasks_store и настраиваем bm25_index в app_mod
    from db.handler_data import RetrievalTask
    task_high = RetrievalTask(name="High task", desc="Auth bug", prio="5", label="Bug")
    task_low = RetrievalTask(name="Low task", desc="Auth bug", prio="1", label="Bug")

    app_mod.tasks_store = {
        "task_high_id": task_high,
        "task_low_id": task_low,
    }
    app_mod.task_metadata_store = {
        "task_high_id": {"business_days": 0, "lead_time_hours": 0},
        "task_low_id": {"business_days": 0, "lead_time_hours": 0},
    }
    app_mod.update_bm25_index_locked()

    # 2. Мокаем query в ChromaDB, чтобы возвращал те же id
    def mock_query(*args, **kwargs):
        return {"ids": [["task_high_id", "task_low_id"]]}
    monkeypatch.setattr(app_mod.collection, "query", mock_query)

    # 3. Мокаем get_chroma_tasks_by_ids в app_mod
    def mock_get_chroma_tasks(ids):
        return [
            {
                "id": "task_high_id",
                "document": (
                    "Название: High task\n"
                    "Описание: Auth bug\n"
                    "Метка: Bug\n"
                    "Приоритет: 5"
                ),
                "metadata": {
                    "created_at": "2026-05-22T10:00:00Z",
                    "finished_at": "2026-05-22T12:00:00Z",
                    "business_days": 0.25,
                    "lead_time_hours": 2.0,
                },
            },
            {
                "id": "task_low_id",
                "document": (
                    "Название: Low task\n"
                    "Описание: Auth bug\n"
                    "Метка: Bug\n"
                    "Приоритет: 1"
                ),
                "metadata": {
                    "created_at": "2026-05-22T10:00:00Z",
                    "finished_at": "2026-05-22T12:00:00Z",
                    "business_days": 0.25,
                    "lead_time_hours": 2.0,
                },
            },
        ]
    monkeypatch.setattr(app_mod, "get_chroma_tasks_by_ids", mock_get_chroma_tasks)

    # 4. Мокаем requests.post для Cohere Reranker
    monkeypatch.setattr(app_mod, "ROUTER_API_KEY", "dummy_key")
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.3},
                ]
            }
    monkeypatch.setattr(app_mod.requests, "post", lambda *args, **kwargs: MockResponse())

    # 5. Делаем запрос к поисковому API
    search_payload = {
        "query": "auth bug",
        "n_results": 10,
        "min_days": 0,
        "max_days": 365,
    }
    response = client.post("/app/v1/search", json=search_payload)
    assert response.status_code == 200

    results = response.json()["results"]
    # Должна остаться только задача с оценкой релевантности > 0.5 (то есть High task)
    assert len(results) == 1
    assert results[0]["name"] == "High task"
    assert results[0]["reranker_score"] == 0.9
