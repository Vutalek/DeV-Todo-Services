from datetime import datetime
from zoneinfo import ZoneInfo

from common.deadline import (
    add_working_hours,
    calculate_deadline,
    calculate_working_hours_between,
    move_to_work_time,
    to_work_timezone,
)


MSK = ZoneInfo("Europe/Moscow")
UTC = ZoneInfo("UTC")


def test_to_work_timezone_keeps_naive_as_moscow():
    value = to_work_timezone(datetime(2024, 5, 6, 12, 0))

    assert value.tzinfo == MSK
    assert value.hour == 12


def test_to_work_timezone_converts_aware_datetime():
    value = to_work_timezone(datetime(2024, 5, 6, 7, 0, tzinfo=UTC))

    assert value.hour == 10
    assert value.tzinfo == MSK


def test_move_to_work_time_before_day_start():
    value = move_to_work_time(datetime(2024, 5, 6, 8, 30, tzinfo=MSK))

    assert value == datetime(2024, 5, 6, 10, 0, tzinfo=MSK)


def test_move_to_work_time_after_day_end():
    value = move_to_work_time(datetime(2024, 5, 6, 19, 0, tzinfo=MSK))

    assert value == datetime(2024, 5, 7, 10, 0, tzinfo=MSK)


def test_move_to_work_time_weekend_to_monday():
    value = move_to_work_time(datetime(2024, 5, 4, 12, 0, tzinfo=MSK))

    assert value == datetime(2024, 5, 6, 10, 0, tzinfo=MSK)


def test_add_working_hours_within_same_day():
    value = add_working_hours(datetime(2024, 5, 6, 10, 0, tzinfo=MSK), 2)

    assert value == datetime(2024, 5, 6, 12, 0, tzinfo=MSK)


def test_add_working_hours_crosses_workday_boundary():
    value = add_working_hours(datetime(2024, 5, 6, 17, 0, tzinfo=MSK), 2)

    assert value == datetime(2024, 5, 7, 11, 0, tzinfo=MSK)


def test_add_working_hours_crosses_weekend():
    value = add_working_hours(datetime(2024, 5, 3, 17, 0, tzinfo=MSK), 2)

    assert value == datetime(2024, 5, 6, 11, 0, tzinfo=MSK)


def test_calculate_working_hours_between_same_day():
    hours = calculate_working_hours_between(
        datetime(2024, 5, 6, 10, 0, tzinfo=MSK),
        datetime(2024, 5, 6, 12, 30, tzinfo=MSK),
    )

    assert hours == 2.5


def test_calculate_working_hours_between_excludes_weekend():
    hours = calculate_working_hours_between(
        datetime(2024, 5, 3, 17, 0, tzinfo=MSK),
        datetime(2024, 5, 6, 11, 0, tzinfo=MSK),
    )

    assert hours == 2.0


def test_calculate_deadline_returns_utc_iso():
    deadline = calculate_deadline(
        2,
        created_at=datetime(2024, 5, 6, 17, 0, tzinfo=MSK),
    )

    assert deadline == "2024-05-07T08:00:00+00:00"
