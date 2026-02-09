import asyncio
from json import dumps
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from duo_workflow_service.checkpointer.gitlab_workflow import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
)
from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum
from duo_workflow_service.executor.outbox import Outbox, OutboxSignal
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.context import client_capabilities, gitlab_version


@pytest.fixture(name="outbox")
def outbox_fixture() -> MagicMock:
    return MagicMock(spec=Outbox())


@pytest.fixture(name="gl_version_18_7")
def gl_version_18_7_fixture():
    """Set GitLab version to 18.7.0 for client capabilities support."""
    gitlab_version.set("18.7.0")
    yield
    gitlab_version.set(None)


@pytest.fixture(name="checkpoint_notifier")
def checkpoint_notifier_fixture(
    outbox, gl_version_18_7
):  # pylint: disable=unused-argument
    client_capabilities.set({"incremental_streaming"})
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
            "message_id": "msg-123",
        }
    ]
    state = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ui_chat_log,
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state, False)
    assert checkpoint_notifier.ui_chat_log == ui_chat_log
    assert checkpoint_notifier.ui_chat_log[0]["message_id"] == "msg-123"
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
                        "message_id": "msg-123",
                    }
                ],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint

    # Verify that last_sent_ui_message_id is tracked
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-123"

    # Send a second event with additional messages
    ui_chat_log_updated = ui_chat_log + [
        {
            "content": "second message",
            "role": "user",
            "status": "success",
            "additional_context": None,
            "message_id": "msg-456",
        }
    ]
    state_updated = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ui_chat_log_updated,
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state_updated, False)

    # Get the most recent checkpoint
    action_updated = checkpoint_notifier.outbox.put_action.call_args[0][0]
    action_updated.newCheckpoint.CopyFrom(
        checkpoint_notifier.most_recent_new_checkpoint()
    )

    expected_checkpoint_updated = dumps(
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
                        "message_id": "msg-123",
                    },
                    {
                        "content": "second message",
                        "role": "user",
                        "status": "success",
                        "additional_context": None,
                        "message_id": "msg-456",
                    },
                ],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action_updated.newCheckpoint.checkpoint == expected_checkpoint_updated
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-456"


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
                "message_id": "agent-msg-id",
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
                        "message_id": "agent-msg-id",
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

    # Verify empty ui_chat_log returns empty list on subsequent calls
    state_empty = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "plan": {},
    }
    await checkpoint_notifier.send_event("values", state_empty, False)
    action_empty = checkpoint_notifier.outbox.put_action.call_args[0][0]
    action_empty.newCheckpoint.CopyFrom(
        checkpoint_notifier.most_recent_new_checkpoint()
    )

    expected_checkpoint_empty = dumps(
        {
            "channel_values": {
                "ui_chat_log": [],
                "plan": {"steps": []},
            }
        }
    )
    assert action_empty.newCheckpoint.checkpoint == expected_checkpoint_empty


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
    assert notifier.latest_ai_message is None


@pytest.mark.parametrize(
    (
        "received_messages",
        "expected_messages",
    ),
    [
        (
            [
                AIMessageChunk(id="agent-msg-id", content="New message"),
            ],
            [
                {
                    "message_id": "agent-msg-id",
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
        ),
        (
            [
                AIMessageChunk(
                    id="agent-msg-id",
                    content=[{"text": "Nested content", "type": "text"}],
                ),
            ],
            [
                {
                    "message_id": "agent-msg-id",
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
        ),
        (
            [
                AIMessageChunk(id="different-msg-id", content="Different content"),
                AIMessageChunk(id="agent-msg-id", content="New content"),
            ],
            [
                {
                    "message_id": "different-msg-id",
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Different content",
                    "tool_info": None,
                    "additional_context": None,
                },
                {
                    "message_id": "agent-msg-id",
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "New content",
                    "tool_info": None,
                    "additional_context": None,
                },
            ],
        ),
        (
            [
                AIMessageChunk(id="agent-msg-id", content="Existing "),
                AIMessageChunk(id="agent-msg-id", content="content"),
            ],
            [
                {
                    "message_id": "agent-msg-id",
                    "status": None,
                    "correlation_id": None,
                    "message_type": MessageTypeEnum.AGENT,
                    "message_sub_type": None,
                    "timestamp": "2023-01-01T00:00:00+00:00",
                    "content": "Existing content",
                    "tool_info": None,
                    "additional_context": None,
                },
            ],
        ),
        (
            [
                AIMessage(id="agent-msg-id", content="Existing "),
                AIMessage(id="agent-msg-id", content="content"),
            ],
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_send_event_messages_stream(
    checkpoint_notifier,
    received_messages,
    expected_messages,
):
    with patch("duo_workflow_service.checkpointer.notifier.datetime") as mock_datetime:
        mock_now = Mock()
        mock_now.now.return_value.isoformat.return_value = "2023-01-01T00:00:00+00:00"
        mock_datetime.now = mock_now.now

        for message in received_messages:
            await checkpoint_notifier.send_event("messages", (message, {}), True)

        assert checkpoint_notifier.ui_chat_log == expected_messages

        action = checkpoint_notifier.outbox.put_action.call_args[0][0]

        # Action in outbox is a placeholder. We need to load the full latest checkpoint.
        action.newCheckpoint.CopyFrom(checkpoint_notifier.most_recent_new_checkpoint())

        assert action.newCheckpoint.goal == "test_goal"
        assert action.newCheckpoint.checkpoint is not None


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
    checkpoint_notifier.ui_chat_log = [{"content": "test", "message_id": "msg-1"}]
    checkpoint_notifier.steps = [{"step": "1"}]

    checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    assert checkpoint.goal == "test_goal"
    assert checkpoint.status == "RUNNING"
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [{"content": "test", "message_id": "msg-1"}],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint.checkpoint == expected_checkpoint
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-1"

    # Add more messages and verify the last sent message plus new ones are included
    checkpoint_notifier.ui_chat_log.append({"content": "test2", "message_id": "msg-2"})
    checkpoint_notifier.ui_chat_log.append({"content": "test3", "message_id": "msg-3"})

    checkpoint2 = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint2 = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test", "message_id": "msg-1"},
                    {"content": "test2", "message_id": "msg-2"},
                    {"content": "test3", "message_id": "msg-3"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint2.checkpoint == expected_checkpoint2
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-3"

    # Verify calling again with no new messages returns only the last sent message
    checkpoint3 = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint3 = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test3", "message_id": "msg-3"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint3.checkpoint == expected_checkpoint3


@pytest.mark.asyncio
async def test_send_event_with_values_type_without_incremental_streaming(
    checkpoint_notifier,
):
    """Test that full chat log is sent when client doesn't support incremental streaming."""
    client_capabilities.set(set())

    ui_chat_log = [
        {
            "content": "message",
            "role": "user",
            "status": "success",
            "additional_context": None,
            "message_id": "msg-123",
        }
    ]
    state = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ui_chat_log,
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state, False)

    action = checkpoint_notifier.outbox.put_action.call_args[0][0]
    action.newCheckpoint.CopyFrom(checkpoint_notifier.most_recent_new_checkpoint())

    # First checkpoint should contain the full chat log
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {
                        "content": "message",
                        "role": "user",
                        "status": "success",
                        "additional_context": None,
                        "message_id": "msg-123",
                    }
                ],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint

    # Send a second event with additional messages
    ui_chat_log_updated = ui_chat_log + [
        {
            "content": "second message",
            "role": "user",
            "status": "success",
            "additional_context": None,
            "message_id": "msg-456",
        }
    ]
    state_updated = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ui_chat_log_updated,
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state_updated, False)

    action_updated = checkpoint_notifier.outbox.put_action.call_args[0][0]
    action_updated.newCheckpoint.CopyFrom(
        checkpoint_notifier.most_recent_new_checkpoint()
    )

    expected_checkpoint_updated = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {
                        "content": "message",
                        "role": "user",
                        "status": "success",
                        "additional_context": None,
                        "message_id": "msg-123",
                    },
                    {
                        "content": "second message",
                        "role": "user",
                        "status": "success",
                        "additional_context": None,
                        "message_id": "msg-456",
                    },
                ],
                "plan": {"steps": ["step1", "step2"]},
            }
        }
    )
    assert action_updated.newCheckpoint.checkpoint == expected_checkpoint_updated


def test_most_recent_new_checkpoint_without_incremental_streaming(checkpoint_notifier):
    """Test that full chat log is always sent when incremental streaming is disabled."""
    client_capabilities.set(set())

    checkpoint_notifier.status = WorkflowStatusEnum.EXECUTION
    checkpoint_notifier.ui_chat_log = [{"content": "test", "message_id": "msg-1"}]
    checkpoint_notifier.steps = [{"step": "1"}]

    checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    assert checkpoint.goal == "test_goal"
    assert checkpoint.status == "RUNNING"
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [{"content": "test", "message_id": "msg-1"}],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint.checkpoint == expected_checkpoint

    checkpoint_notifier.ui_chat_log.append({"content": "test2", "message_id": "msg-2"})
    checkpoint_notifier.ui_chat_log.append({"content": "test3", "message_id": "msg-3"})

    checkpoint2 = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint2 = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test", "message_id": "msg-1"},
                    {"content": "test2", "message_id": "msg-2"},
                    {"content": "test3", "message_id": "msg-3"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint2.checkpoint == expected_checkpoint2

    # Verify calling again with no new messages still returns the full chat log
    checkpoint3 = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint3 = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test", "message_id": "msg-1"},
                    {"content": "test2", "message_id": "msg-2"},
                    {"content": "test3", "message_id": "msg-3"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint3.checkpoint == expected_checkpoint3


def test_most_recent_new_checkpoint_with_missing_message_ids(checkpoint_notifier):
    """Test that messages without message_id are handled gracefully."""
    client_capabilities.set({"incremental_streaming"})

    checkpoint_notifier.status = WorkflowStatusEnum.EXECUTION
    checkpoint_notifier.ui_chat_log = [
        {"content": "test1"},  # No message_id
        {"content": "test2", "message_id": "msg-2"},
        {"content": "test3"},  # No message_id
        {"content": "test4", "message_id": "msg-4"},
    ]
    checkpoint_notifier.steps = [{"step": "1"}]

    checkpoint = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test1"},
                    {"content": "test2", "message_id": "msg-2"},
                    {"content": "test3"},
                    {"content": "test4", "message_id": "msg-4"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint.checkpoint == expected_checkpoint
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-4"

    checkpoint_notifier.ui_chat_log.append({"content": "test5"})
    checkpoint_notifier.ui_chat_log.append({"content": "test6", "message_id": "msg-6"})

    checkpoint2 = checkpoint_notifier.most_recent_new_checkpoint()
    expected_checkpoint2 = dumps(
        {
            "channel_values": {
                "ui_chat_log": [
                    {"content": "test4", "message_id": "msg-4"},
                    {"content": "test5"},
                    {"content": "test6", "message_id": "msg-6"},
                ],
                "plan": {"steps": [{"step": "1"}]},
            }
        }
    )
    assert checkpoint2.checkpoint == expected_checkpoint2
    assert checkpoint_notifier.last_sent_ui_message_id == "msg-6"
