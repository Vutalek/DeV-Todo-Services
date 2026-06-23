from types import SimpleNamespace

import requests
from fastapi.testclient import TestClient

from app.main import app
import app.config as config
import app.services.rag as rag
import app.services.trello as trello_svc
from app.models import create_dynamic_task_model


def _task_payload(document=None, **metadata):
    return {
        "id": "task_1",
        "document": document or (
            "Название: Doc name\n"
            "Описание: Doc desc\n"
            "Метка: Doc label\n"
            "Приоритет: Doc priority"
        ),
        "metadata": metadata,
    }


def test_task_payload_to_rerank_document_uses_document_and_time_metadata():
    document = rag.task_payload_to_rerank_document(
        _task_payload(
            created_at="2024-05-06T10:00:00+03:00",
            finished_at="2024-05-06T12:00:00+03:00",
            business_days=0.25,
            lead_time_hours=2.0,
        )
    )

    assert "Name: Doc name" in document
    assert "Business days: 0.25" in document
    assert "Time hours: 2" in document


def test_task_payload_to_search_result_uses_document_fields():
    result = rag.task_payload_to_search_result(_task_payload(), reranker_score=None)

    assert result == {
        "name": "Doc name",
        "desc": "Doc desc",
        "priority": "Doc priority",
        "label": "Doc label",
        "reranker_score": None,
        "business_days": 0,
        "time_hours": None,
    }


def test_rerank_tasks_empty_tasks_returns_no_warning():
    results, warning = rag.rerank_tasks("query", [])

    assert results == []
    assert warning is None


def test_rerank_tasks_missing_key_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", None)

    results, warning = rag.rerank_tasks("query", [_task_payload()])

    assert results == []
    assert "ROUTER_API_KEY" in warning


def test_rerank_tasks_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(rag.requests, "post", raise_timeout)

    results, warning = rag.rerank_tasks("query", [_task_payload()])

    assert results == []
    assert "timed out" in warning


def test_rerank_tasks_non_200_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")

    response = type("Response", (), {"status_code": 429, "text": "rate limit"})()
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: response)

    results, warning = rag.rerank_tasks("query", [_task_payload()])

    assert results == []
    assert "HTTP 429" in warning


def test_rerank_tasks_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")

    class Response:
        status_code = 200
        text = "not json"

        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: Response())

    results, warning = rag.rerank_tasks("query", [_task_payload()])

    assert results == []
    assert "invalid JSON" in warning


def test_rerank_tasks_empty_results_falls_back(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")

    response = type(
        "Response",
        (),
        {"status_code": 200, "text": "{}", "json": lambda self: {"results": []}},
    )()
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: response)

    results, warning = rag.rerank_tasks("query", [_task_payload()])

    assert results == []
    assert "no reranked results" in warning


def test_rerank_tasks_happy_path_orders_by_response_indexes(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")
    tasks = [
        _task_payload(
            document=(
                "Название: First\n"
                "Описание: A\n"
                "Метка: Task\n"
                "Приоритет: Low"
            ),
        ),
        _task_payload(
            document=(
                "Название: Second\n"
                "Описание: B\n"
                "Метка: Bug\n"
                "Приоритет: High"
            ),
        ),
    ]

    response = type(
        "Response",
        (),
        {
            "status_code": 200,
            "text": "{}",
            "json": lambda self: {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        },
    )()
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: response)

    results, warning = rag.rerank_tasks("query", tasks, top_n=2)

    assert warning is None
    assert [item["name"] for item in results] == ["Second", "First"]
    assert results[0]["reranker_score"] == 0.9


def test_create_dynamic_task_model_validates_literals():
    Task = create_dynamic_task_model(columns=["Backlog"], labels=["Bug"])

    parsed = Task(
        name="Fix",
        desc="Bug",
        label=["Bug"],
        prio=3,
        time=2,
        roadmap="Шаг 1 (2 часов): fix",
        column="Backlog",
    )

    assert parsed.column == "Backlog"


def test_build_message_text_with_similar_tasks_adds_context():
    message_text = rag.build_message_text_with_similar_tasks(
        "Починить авторизацию",
        [
            {
                "name": "Fix login",
                "desc": "Token refresh bug",
                "label": "Bug",
                "priority": "High",
                "reranker_score": 0.91,
                "business_days": 0.5,
                "time_hours": 4,
            }
        ],
    )

    assert "Новая задача:\nПочинить авторизацию" in message_text
    assert "Самые похожие задачи из истории:" in message_text
    assert "name: Fix login" in message_text
    assert "desc: Token refresh bug" in message_text
    assert "priority: High" in message_text
    assert "label: Bug" in message_text
    assert "reranker_score: 0.91" in message_text
    assert "business_days: 0.5" in message_text
    assert "time_hours: 4" in message_text


def test_send_enriches_message_with_similar_tasks(monkeypatch):
    client = TestClient(app)
    captured = {}

    monkeypatch.setattr(config, "ROUTER_API_KEY", "key")
    monkeypatch.setattr(
        trello_svc,
        "get_trello_data",
        lambda: ({"Backlog": "list_1"}, {"Bug": "label_1"}),
    )
    monkeypatch.setattr(
        rag,
        "get_similar_tasks_for_message",
        lambda text: [
            {
                "name": "Fix login",
                "desc": "Token refresh bug",
                "label": "Bug",
                "priority": "High",
                "reranker_score": 0.91,
                "business_days": 0.5,
                "time_hours": 4,
            }
        ],
    )

    class ParsedTask:
        time = 3

        def model_dump(self):
            return {
                "name": "Починить авторизацию",
                "desc": "Ошибка refresh token",
                "label": ["Bug"],
                "prio": 4,
                "time": self.time,
                "roadmap": "Проверить refresh flow",
                "column": "Backlog",
            }

    class FakeCompletions:
        async def parse(self, **kwargs):
            captured["user_content"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(parsed=ParsedTask())
                    )
                ]
            )

    monkeypatch.setattr(
        config,
        "async_client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    response = client.post(
        "/app/v1/send",
        json={"text": "Починить авторизацию"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "Починить авторизацию" in captured["user_content"]
    assert "Fix login" in captured["user_content"]
