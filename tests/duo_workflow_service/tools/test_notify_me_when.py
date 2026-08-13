# pylint: disable=file-naming-for-tests
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from contract import contract_pb2
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD
from duo_workflow_service.tools.notify_me_when import (
    NotifyMeWhen,
    NotifyMeWhenInput,
    NotifyWhen,
    _to_rfc3339,
)


def _ack_event(
    accepted: bool, reason: str = "", fires_at: str = ""
) -> contract_pb2.ClientEvent:
    event = contract_pb2.ClientEvent()
    event.actionResponse.requestID = "test-request-id"
    ack = event.actionResponse.scheduleNotificationResponse
    ack.accepted = accepted
    ack.reason = reason
    ack.fires_at = fires_at
    return event


@pytest.fixture(name="mock_outbox")
def mock_outbox_fixture():
    outbox = MagicMock()
    outbox.put_action_and_wait_for_response = AsyncMock()
    return outbox


@pytest.fixture(name="notify_tool")
def notify_tool_fixture(mock_outbox):
    tool = NotifyMeWhen()
    tool.metadata = {"outbox": mock_outbox}
    return tool


@pytest.fixture(name="accepting_outbox")
def accepting_outbox_fixture(mock_outbox):
    mock_outbox.put_action_and_wait_for_response.return_value = _ack_event(
        accepted=True
    )
    return mock_outbox


@pytest.mark.parametrize("delay_minutes", [1, 5, 24 * 60])
def test_input_accepts_a_delay_within_bounds(delay_minutes):
    model = NotifyMeWhenInput(
        message="hi", when={"type": "timer", "delay_minutes": delay_minutes}
    )

    assert model.when.delay_minutes == delay_minutes
    assert model.when.type == "timer"


def test_input_defaults_the_condition_type():
    model = NotifyMeWhenInput(message="hi", when={"delay_minutes": 5})

    assert model.when.type == "timer"


@pytest.mark.parametrize("delay_minutes", [0, -1, 24 * 60 + 1])
def test_input_rejects_out_of_range_delay(delay_minutes):
    with pytest.raises(ValidationError):
        NotifyMeWhenInput(
            message="hi", when={"type": "timer", "delay_minutes": delay_minutes}
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"when": {"type": "timer", "delay_minutes": 5}},
        {"message": "hi"},
        {"message": "hi", "when": {"type": "timer"}},
    ],
    ids=["missing message", "missing when", "missing delay_minutes"],
)
def test_input_requires_both_fields(kwargs):
    with pytest.raises(ValidationError):
        NotifyMeWhenInput(**kwargs)


@pytest.mark.parametrize("bad_type", ["pipeline", "", "TIMER", "event"])
def test_input_rejects_unknown_condition_type(bad_type):
    with pytest.raises(ValidationError):
        NotifyMeWhenInput(message="hi", when={"type": bad_type, "delay_minutes": 5})


def test_to_rfc3339_is_utc_with_second_precision():
    value = datetime(2026, 1, 1, 10, 0, 30, 500000, tzinfo=timezone.utc)

    assert _to_rfc3339(value) == "2026-01-01T10:00:30Z"


@pytest.mark.asyncio
async def test_execute_schedules_and_reports_the_task_id(notify_tool, accepting_outbox):
    result = await notify_tool._execute(
        message="check the pipeline",
        when={"type": "timer", "delay_minutes": 10},
    )

    action = accepting_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.WhichOneof("action") == "scheduleNotification"
    assert action.scheduleNotification.message == "check the pipeline"
    assert action.scheduleNotification.task_id

    condition = json.loads(action.scheduleNotification.condition)
    assert condition["type"] == "timer"

    # The model is told the task_id and the deadline that was sent to the client.
    assert action.scheduleNotification.task_id in result
    assert condition["deadline"] in result


@pytest.mark.asyncio
async def test_execute_never_schedules_earlier_than_requested(
    notify_tool, accepting_outbox
):
    before = datetime.now(timezone.utc)

    await notify_tool._execute(
        message="later", when={"type": "timer", "delay_minutes": 1}
    )

    action = accepting_outbox.put_action_and_wait_for_response.call_args[0][0]
    deadline = json.loads(action.scheduleNotification.condition)["deadline"]
    assert deadline >= _to_rfc3339(before)


@pytest.mark.asyncio
async def test_execute_accepts_a_notify_when_instance(notify_tool, accepting_outbox):
    result = await notify_tool._execute(
        message="check the pipeline",
        when=NotifyWhen(type="timer", delay_minutes=10),
    )

    assert "scheduled" in result


@pytest.mark.asyncio
async def test_execute_generates_a_distinct_task_id_per_call(
    notify_tool, accepting_outbox
):
    first = await notify_tool._execute(message="one", when={"delay_minutes": 5})
    second = await notify_tool._execute(message="two", when={"delay_minutes": 5})

    task_ids = [
        call[0][0].scheduleNotification.task_id
        for call in accepting_outbox.put_action_and_wait_for_response.call_args_list
    ]
    assert task_ids[0] != task_ids[1]
    assert task_ids[0] in first
    assert task_ids[1] in second


@pytest.mark.asyncio
async def test_execute_reports_the_delivery_time_from_the_client(
    notify_tool, mock_outbox
):
    """The executor holds the timer, so its fires_at wins over our computed one."""
    mock_outbox.put_action_and_wait_for_response.return_value = _ack_event(
        accepted=True, fires_at="2099-01-01T10:00:00Z"
    )

    result = await notify_tool._execute(message="later", when={"delay_minutes": 10})

    assert "2099-01-01T10:00:00Z" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "match"),
    [
        ("too many outstanding", "too many outstanding"),
        ("", "no reason given"),
    ],
    ids=["with reason", "without reason"],
)
async def test_execute_raises_when_the_client_refuses(
    notify_tool, mock_outbox, reason, match
):
    mock_outbox.put_action_and_wait_for_response.return_value = _ack_event(
        accepted=False, reason=reason
    )

    with pytest.raises(ToolException, match=match):
        await notify_tool._execute(
            message="later", when={"type": "timer", "delay_minutes": 10}
        )


@pytest.mark.asyncio
async def test_execute_raises_when_the_client_answers_with_another_response_type(
    notify_tool, mock_outbox
):
    event = contract_pb2.ClientEvent()
    event.actionResponse.requestID = "test-request-id"
    event.actionResponse.plainTextResponse.response = "ok"
    mock_outbox.put_action_and_wait_for_response.return_value = event

    with pytest.raises(ToolException, match="did not acknowledge"):
        await notify_tool._execute(
            message="later", when={"type": "timer", "delay_minutes": 10}
        )


def test_tool_is_experimental():
    """Keeps the tool out of the ListTools API while its argument shape can change."""
    assert NotifyMeWhen.tool_version < STABLE_VERSION_THRESHOLD


def test_format_display_message():
    tool = NotifyMeWhen()
    args = NotifyMeWhenInput(message="hi", when={"type": "timer", "delay_minutes": 5})

    assert tool.format_display_message(args) == "Schedule a notification in 5 minute(s)"
