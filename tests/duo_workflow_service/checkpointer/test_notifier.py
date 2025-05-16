import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from langchain.load.dump import dumps
from langchain_core.messages import HumanMessage

from contract import contract_pb2
from duo_workflow_service.checkpointer.notifier import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
    UserInterface,
)
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum


@pytest.fixture
def outbox():
    return asyncio.Queue()


@pytest.fixture
def checkpoint_notifier(outbox):
    return UserInterface(outbox=outbox, goal="test_goal")


@pytest.mark.asyncio
async def test_send_event_with_non_values_type(checkpoint_notifier):
    state = {"not_values_state": "state"}
    result = await checkpoint_notifier.send_event("not_values", state, False)
    assert result is None
    assert checkpoint_notifier.outbox.empty()


@pytest.mark.asyncio
async def test_send_event_with_values_type(checkpoint_notifier):
    state = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ["message1", "message2"],
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state, False)
    assert checkpoint_notifier.ui_chat_log == ["message1", "message2"]
    assert checkpoint_notifier.status == WorkflowStatusEnum.COMPLETED
    assert checkpoint_notifier.steps == ["step1", "step2"]
    assert not checkpoint_notifier.outbox.empty()
    action = await checkpoint_notifier.outbox.get()
    assert action.newCheckpoint.goal == "test_goal"
    assert action.newCheckpoint.status == "FINISHED"
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": ["message1", "message2"],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint


@pytest.mark.asyncio
async def test_send_event_with_missing_plan_steps(checkpoint_notifier):
    state = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": ["message"],
        "plan": {},
    }
    await checkpoint_notifier.send_event("values", state, False)
    action = await checkpoint_notifier.outbox.get()
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": ["message"],
                "plan": {"steps": []},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint


def test_workflow_status_mapping():
    expected_mapping = {
        WorkflowStatusEnum.EXECUTION: "RUNNING",
        WorkflowStatusEnum.ERROR: "FAILED",
        WorkflowStatusEnum.INPUT_REQUIRED: "INPUT_REQUIRED",
        WorkflowStatusEnum.PLANNING: "RUNNING",
        WorkflowStatusEnum.PAUSED: "PAUSED",
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED: "PLAN_APPROVAL_REQUIRED",
        WorkflowStatusEnum.NOT_STARTED: "CREATED",
        WorkflowStatusEnum.COMPLETED: "FINISHED",
        WorkflowStatusEnum.CANCELLED: "STOPPED",
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED: "REQUIRE_TOOL_CALL_APPROVAL",
    }

    for workflow_status, checkpoint_status in expected_mapping.items():
        assert (
            WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[workflow_status] == checkpoint_status
        )

    assert len(WORKFLOW_STATUS_TO_CHECKPOINT_STATUS) == len(expected_mapping)


@pytest.mark.asyncio
async def test_init_sets_attributes(outbox):
    notifier = UserInterface(outbox=outbox, goal="custom_goal")
    assert notifier.outbox == outbox
    assert notifier.goal == "custom_goal"
    assert notifier.ui_chat_log == []
    assert notifier.status == WorkflowStatusEnum.NOT_STARTED
    assert notifier.steps == []


@pytest.mark.parametrize(
    ("existing_messages", "message_content", "expected_messages"),
    [
        (
            [],
            "New message",
            [
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "New message",
                    "tool_info": None,
                }
            ],
        ),
        (
            [
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Existing ",
                    "tool_info": None,
                }
            ],
            "content",
            [
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Existing content",
                    "tool_info": None,
                }
            ],
        ),
        (
            [
                {
                    "status": "COMPLETED",
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Completed message",
                    "tool_info": None,
                }
            ],
            "New message",
            [
                {
                    "status": "COMPLETED",
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Completed message",
                    "tool_info": None,
                },
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "New message",
                    "tool_info": None,
                },
            ],
        ),
        (
            [
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.USER,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "User message",
                    "tool_info": None,
                }
            ],
            "Agent response",
            [
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.USER,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "User message",
                    "tool_info": None,
                },
                {
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Agent response",
                    "tool_info": None,
                },
            ],
        ),
        (
            [],
            "",
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_send_event_messages_stream(
    checkpoint_notifier, existing_messages, message_content, expected_messages
):
    checkpoint_notifier.ui_chat_log = existing_messages

    with patch("duo_workflow_service.checkpointer.notifier.datetime") as mock_datetime:
        mock_now = Mock()
        mock_now.now.return_value.isoformat.return_value = "2023-01-01T00:00:00+00:00"
        mock_datetime.now = mock_now.now

        message = HumanMessage(content=message_content)
        await checkpoint_notifier.send_event("messages", (message, {}), True)

        assert checkpoint_notifier.ui_chat_log == expected_messages

        assert not checkpoint_notifier.outbox.empty()
        action = await checkpoint_notifier.outbox.get()
        assert action.newCheckpoint.goal == "test_goal"
        assert action.newCheckpoint.checkpoint is not None
