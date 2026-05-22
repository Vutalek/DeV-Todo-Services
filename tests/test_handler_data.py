from pathlib import Path

from db.handler_data import (
    RetrievalTask,
    compute_business_days,
    compute_lead_time_hours,
    parse_datetime,
    task_to_document,
    task_to_metadata,
)


def test_parse_datetime_accepts_z_suffix():
    value = parse_datetime("2024-05-06T10:00:00Z")

    assert value.isoformat() == "2024-05-06T10:00:00+00:00"


def test_parse_datetime_empty_returns_none():
    assert parse_datetime("") is None
    assert parse_datetime(None) is None


def test_compute_lead_time_hours_without_finished_returns_none():
    assert compute_lead_time_hours("2024-05-06T10:00:00+03:00", None) is None


def test_compute_lead_time_hours_finished_before_created_returns_none():
    hours = compute_lead_time_hours(
        "2024-05-06T12:00:00+03:00",
        "2024-05-06T10:00:00+03:00",
    )

    assert hours is None


def test_compute_lead_time_hours_uses_working_calendar():
    hours = compute_lead_time_hours(
        "2024-05-03T17:00:00+03:00",
        "2024-05-06T11:00:00+03:00",
    )

    assert hours == 2.0


def test_compute_business_days_from_working_hours():
    days = compute_business_days(
        "2024-05-06T10:00:00+03:00",
        "2024-05-06T12:00:00+03:00",
    )

    assert days == 0.25


def test_task_to_document_contains_main_fields():
    task = RetrievalTask(name="Fix auth", desc="Token bug", prio="High", label="Bug")
    document = task_to_document(task)

    assert "Название: Fix auth" in document
    assert "Описание: Token bug" in document
    assert "Метка: Bug" in document
    assert "Приоритет: High" in document


def test_task_to_metadata_contains_working_duration():
    task = RetrievalTask(
        name="Fix auth",
        desc="Token bug",
        prio="High",
        label="Bug",
        created_at="2024-05-06T10:00:00+03:00",
        finished_at="2024-05-06T12:00:00+03:00",
    )

    metadata = task_to_metadata(task)

    assert metadata["name"] == "Fix auth"
    assert metadata["labels"] == "Bug"
    assert metadata["business_days"] == 0.25
    assert metadata["lead_time_hours"] == 2.0

