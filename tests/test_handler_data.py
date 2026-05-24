from db.handler_data import (
    RetrievalTask,
    compute_business_days,
    compute_lead_time_hours,
    get_document_field,
    parse_datetime,
    task_from_document_metadata,
    task_metadata_to_business_days,
    task_metadata_to_time_hours,
    task_payload_to_fields,
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


def test_get_document_field_extracts_value():
    document = "Название: Fix auth\nОписание: Token bug"

    assert get_document_field(document, "Описание") == "Token bug"


def test_get_document_field_missing_returns_empty_string():
    assert get_document_field("Название: Fix auth", "Описание") == ""


def test_task_metadata_duration_helpers_use_timestamps():
    metadata = {
        "created_at": "2024-05-06T10:00:00+03:00",
        "finished_at": "2024-05-06T12:00:00+03:00",
    }

    assert task_metadata_to_business_days(metadata) == 0.25
    assert task_metadata_to_time_hours(metadata) == 2


def test_task_from_document_metadata_builds_retrieval_task():
    document = (
        "Название: Fix auth\n"
        "Описание: Token bug\n"
        "Метка: Bug\n"
        "Приоритет: High"
    )
    metadata = {
        "created_at": "2024-05-06T10:00:00+03:00",
        "finished_at": "2024-05-06T12:00:00+03:00",
    }

    task = task_from_document_metadata(document, metadata)

    assert task == RetrievalTask(
        name="Fix auth",
        desc="Token bug",
        prio="High",
        label="Bug",
        created_at="2024-05-06T10:00:00+03:00",
        finished_at="2024-05-06T12:00:00+03:00",
    )


def test_task_payload_to_fields_uses_document_and_metadata():
    task = {
        "document": (
            "Название: Fix auth\n"
            "Описание: Token bug\n"
            "Метка: Bug\n"
            "Приоритет: High"
        ),
        "metadata": {
            "created_at": "2024-05-06T10:00:00+03:00",
            "finished_at": "2024-05-06T12:00:00+03:00",
        },
    }

    assert task_payload_to_fields(task) == {
        "name": "Fix auth",
        "desc": "Token bug",
        "priority": "High",
        "label": "Bug",
        "business_days": 0.25,
        "time_hours": 2,
    }


def test_task_to_metadata_contains_only_timestamps():
    task = RetrievalTask(
        name="Fix auth",
        desc="Token bug",
        prio="High",
        label="Bug",
        created_at="2024-05-06T10:00:00+03:00",
        finished_at="2024-05-06T12:00:00+03:00",
    )

    metadata = task_to_metadata(task)

    assert metadata == {
        "created_at": "2024-05-06T10:00:00+03:00",
        "finished_at": "2024-05-06T12:00:00+03:00",
    }
