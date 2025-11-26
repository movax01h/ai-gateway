import asyncio
from json import dumps
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.checkpointer.gitlab_workflow import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
)
from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum
from duo_workflow_service.executor.outbox import Outbox, OutboxSignal
from duo_workflow_service.workflows.type_definitions import AdditionalContext


@pytest.fixture(name="outbox")
def outbox_fixture() -> MagicMock:
    return MagicMock(spec=Outbox())


@pytest.fixture(name="checkpoint_notifier")
def checkpoint_notifier_fixture(outbox):
    return UserInterface(outbox=outbox, goal="test_goal")


@pytest.mark.asyncio
async def test_send_event_with_non_values_type(checkpoint_notifier):
    state = {"not_values_state": "state"}
    result = await checkpoint_notifier.send_event("not_values", state, False)
    assert result is None


@pytest.mark.asyncio
async def test_send_event_with_values_type(checkpoint_notifier):
    ui_chat_log = [
        {
            "content": "message",
            "role": "user",
            "status": "success",
            "additional_context": AdditionalContext(
                category="file", content="content", id="1"
            ),
        }
    ]
    state = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ui_chat_log,
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state, False)
    assert checkpoint_notifier.ui_chat_log == ui_chat_log
    assert checkpoint_notifier.status == WorkflowStatusEnum.COMPLETED
    assert checkpoint_notifier.steps == ["step1", "step2"]
    action = checkpoint_notifier.outbox.put_action.call_args[0][0]

    # Action in outbox is a placeholder. We need to load the full latest checkpoint.
    action.newCheckpoint.CopyFrom(checkpoint_notifier.most_recent_new_checkpoint())

    assert action.newCheckpoint.goal == "test_goal"
    assert action.newCheckpoint.status == "FINISHED"
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {
                        "content": "message",
                        "role": "user",
                        "status": "success",
                        "additional_context": {
                            "category": "file",
                            "id": "1",
                            "content": "content",
                            "metadata": None,
                            "type": "AdditionalContext",
                        },
                    }
                ],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint


@pytest.mark.asyncio
async def test_send_event_with_missing_plan_steps(checkpoint_notifier):
    state = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [
            {
                "content": "message",
                "message_type": MessageTypeEnum.AGENT,
                "message_sub_type": None,
                "timestamp": "2023-01-01T00:00:00+00:00",
                "status": None,
                "correlation_id": None,
                "tool_info": None,
                "additional_context": None,
                "message_id": None,
            }
        ],
        "plan": {},
    }
    await checkpoint_notifier.send_event("values", state, False)
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {
                        "content": "message",
                        "message_type": MessageTypeEnum.AGENT,
                        "message_sub_type": None,
                        "timestamp": "2023-01-01T00:00:00+00:00",
                        "status": None,
                        "correlation_id": None,
                        "tool_info": None,
                        "additional_context": None,
                        "message_id": None,
                    }
                ],
                "plan": {"steps": []},
            }
        }
    )
    action = checkpoint_notifier.outbox.put_action.call_args[0][0]

    # Action in outbox is a placeholder. We need to load the full latest checkpoint.
    action.newCheckpoint.CopyFrom(checkpoint_notifier.most_recent_new_checkpoint())

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
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED: "TOOL_CALL_APPROVAL_REQUIRED",
        WorkflowStatusEnum.APPROVAL_ERROR: "RUNNING",
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
    (
        "existing_messages",
        "message_content",
        "expected_messages",
        "should_execute_action",
    ),
    [
        (
            [],
            "New message",
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "New message",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            True,
        ),
        (
            [],
            [{"text": "Nested content", "type": "text"}],
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Nested content",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            True,
        ),
        (
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Existing ",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            "content",
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Existing content",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            True,
        ),
        (
            [
                {
                    "message_id": None,
                    "status": "COMPLETED",
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Completed message",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            "New message",
            [
                {
                    "message_id": None,
                    "status": "COMPLETED",
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Completed message",
                    "tool_info": None,
                    "additional_context": None,
                },
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "New message",
                    "tool_info": None,
                    "additional_context": None,
                },
            ],
            True,
        ),
        (
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.USER,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "User message",
                    "tool_info": None,
                    "additional_context": None,
                }
            ],
            "Agent response",
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.USER,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "User message",
                    "tool_info": None,
                    "additional_context": None,
                },
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Agent response",
                    "tool_info": None,
                    "additional_context": None,
                },
            ],
            True,
        ),
        (
            [],
            "",
            [
                {
                    "message_id": None,
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "",
                    "tool_info": None,
                    "additional_context": None,
                },
            ],
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_send_event_messages_stream(
    checkpoint_notifier,
    existing_messages,
    message_content,
    expected_messages,
    should_execute_action,
):
    checkpoint_notifier.ui_chat_log = existing_messages

    with patch("duo_workflow_service.checkpointer.notifier.datetime") as mock_datetime:
        mock_now = Mock()
        mock_now.now.return_value.isoformat.return_value = "2023-01-01T00:00:00+00:00"
        mock_datetime.now = mock_now.now

        message = HumanMessage(content=message_content)
        await checkpoint_notifier.send_event("messages", (message, {}), True)

        assert checkpoint_notifier.ui_chat_log == expected_messages

        if should_execute_action:
            action = checkpoint_notifier.outbox.put_action.call_args[0][0]

            # Action in outbox is a placeholder. We need to load the full latest checkpoint.
            action.newCheckpoint.CopyFrom(
                checkpoint_notifier.most_recent_new_checkpoint()
            )

            assert action.newCheckpoint.goal == "test_goal"
            assert action.newCheckpoint.checkpoint is not None
        else:
            checkpoint_notifier.outbox.put_action.assert_not_called()


@pytest.mark.asyncio
async def test_checkpoint_number_increments_on_send_event(checkpoint_notifier):
    assert checkpoint_notifier.checkpoint_number == 0

    state = {"status": WorkflowStatusEnum.PLANNING, "ui_chat_log": [], "plan": {}}
    await checkpoint_notifier.send_event("values", state, False)
    assert checkpoint_notifier.checkpoint_number == 1

    await checkpoint_notifier.send_event("values", state, False)
    assert checkpoint_notifier.checkpoint_number == 2


def test_most_recent_new_checkpoint(checkpoint_notifier):
    checkpoint_notifier.status = WorkflowStatusEnum.EXECUTION
    checkpoint_notifier.ui_chat_log = [{"content": "test"}]
    checkpoint_notifier.steps = [{"step": "1"}]

    checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    assert checkpoint.goal == "test_goal"
    assert checkpoint.status == "RUNNING"
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [{"content": "test"}],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint.checkpoint == expected_checkpoint
