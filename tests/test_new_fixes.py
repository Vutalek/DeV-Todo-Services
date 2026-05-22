from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from fastapi.testclient import TestClient

from db.handler_data import parse_datetime
from common.deadline import WORK_TIMEZONE
import app.app as app_mod


def test_parse_datetime_timezone_safety():
    # Naive datetime should get Europe/Moscow timezone
    dt_naive = parse_datetime("2026-05-22T10:00:00")
    assert dt_naive.tzinfo == WORK_TIMEZONE
    assert dt_naive.hour == 10

    # Aware datetime should keep its timezone
    dt_aware = parse_datetime("2026-05-22T10:00:00+03:00")
    assert dt_aware.tzinfo == ZoneInfo("Europe/Moscow") or dt_aware.utcoffset().total_seconds() == 10800
    assert dt_aware.hour == 10

    # UTC/Z datetime should keep its timezone
    dt_utc = parse_datetime("2026-05-22T10:00:00Z")
    assert dt_utc.utcoffset().total_seconds() == 0


def test_add_or_update_task_rollback_on_failure(monkeypatch):
    client = TestClient(app_mod.app)

    # 1. Mock ChromaDB delete call to track it
    deleted_ids = []
    def mock_delete(ids=None):
        if ids:
            deleted_ids.extend(ids)

    monkeypatch.setattr(app_mod.collection, "delete", mock_delete)

    # 2. Mock update_bm25_index_locked to fail
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

    # 3. Call the API and ensure it returns 502
    response = client.post("/app/v1/tasks", json=task_payload)
    assert response.status_code == 502
    assert "Rebuilding BM25 failed intentionally" in response.json()["message"]

    # 4. Verify that deletion in Chroma was called
    assert len(deleted_ids) == 1

    # 5. Verify that the task was NOT left in the in-memory tasks_store
    task_id = deleted_ids[0]
    assert task_id not in app_mod.tasks_store
