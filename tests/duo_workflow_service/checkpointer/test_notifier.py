# pylint: disable=too-many-lines
import asyncio
from json import dumps, loads
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from duo_workflow_service.checkpointer.gitlab_workflow import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
)
from duo_workflow_service.checkpointer.node_lifecycle import NodeEventLog, NodePhase
from duo_workflow_service.checkpointer.notifier import (
    UserInterface,
    _agent_token_totals,
)
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.executor.outbox import Outbox
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
    outbox,
    gl_version_18_7,  # pylint: disable=unused-argument  # fixture-on-fixture ordering dep
):
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
                    "component_name": None,
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
                    "component_name": None,
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
                    "component_name": None,
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
                    "component_name": None,
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
                    "component_name": None,
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
async def test_send_event_messages_stream_attributes_component_from_node(
    checkpoint_notifier,
):
    # The messages stream yields (chunk, metadata); metadata["langgraph_node"] is
    # the runtime node, e.g. "researcher#agent". The streamed entry must be stamped
    # with the bare component name so the client can attribute it to a graph node.
    await checkpoint_notifier.send_event(
        "messages",
        (
            AIMessageChunk(id="agent-msg-id", content="hello"),
            {"langgraph_node": "researcher#agent"},
        ),
        True,
    )

    assert checkpoint_notifier.ui_chat_log[-1]["component_name"] == "researcher"


@pytest.mark.asyncio
async def test_send_event_messages_stream_non_dict_metadata_yields_no_component(
    checkpoint_notifier,
):
    # When the messages-stream metadata is not a dict, attribution is skipped
    # gracefully: the entry is still created with component_name=None and no
    # exception is raised.
    await checkpoint_notifier.send_event(
        "messages",
        (AIMessageChunk(id="agent-msg-id", content="hello"), None),
        True,
    )

    assert checkpoint_notifier.ui_chat_log[-1]["component_name"] is None


@pytest.mark.asyncio
async def test_send_event_messages_stream_continuation_keeps_component(
    checkpoint_notifier,
):
    # The first chunk stamps component_name; a continuation chunk with the same
    # message id extends that entry and must not overwrite it, even when the
    # continuation carries no node metadata.
    await checkpoint_notifier.send_event(
        "messages",
        (
            AIMessageChunk(id="agent-msg-id", content="hello "),
            {"langgraph_node": "researcher#agent"},
        ),
        True,
    )
    await checkpoint_notifier.send_event(
        "messages",
        (AIMessageChunk(id="agent-msg-id", content="world"), {}),
        True,
    )

    assert len(checkpoint_notifier.ui_chat_log) == 1
    assert checkpoint_notifier.ui_chat_log[-1]["component_name"] == "researcher"
    assert checkpoint_notifier.ui_chat_log[-1]["content"] == "hello world"


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


def test_most_recent_new_checkpoint_logs_tool_approval_request(checkpoint_notifier):
    """Checkpoint with a REQUEST tool_info triggers the approval logging path."""
    checkpoint_notifier.status = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    checkpoint_notifier.ui_chat_log = [
        {
            "message_type": MessageTypeEnum.REQUEST,
            "tool_info": {
                "name": "run_command",
                "args": {"command": "git checkout feature/x"},
                "suggested_patterns": ["git checkout *"],
            },
            "message_id": "msg-1",
        }
    ]

    checkpoint = checkpoint_notifier.most_recent_new_checkpoint()
    data = checkpoint.checkpoint
    assert "suggested_patterns" in data
    assert "git checkout *" in data


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


@pytest.mark.parametrize(
    ("chunks", "expected_log_ids", "expected_log_contents"),
    [
        pytest.param(
            [
                AIMessageChunk(id="resp_abc", content="Hello "),
                AIMessageChunk(id="lc_run_xyz", content="world"),
            ],
            ["resp_abc"],
            ["Hello world"],
        ),
        pytest.param(
            [
                AIMessageChunk(id="lc_run_xyz", content="Hello "),
                AIMessageChunk(id="lc_run_xyz", content="world"),
            ],
            ["lc_run_xyz"],
            ["Hello world"],
        ),
        pytest.param(
            [
                AIMessageChunk(id="lc_run_1", content="Hello "),
                AIMessageChunk(id="lc_run_2", content="world"),
            ],
            ["lc_run_1", "lc_run_2"],
            ["Hello ", "world"],
        ),
        pytest.param(
            [
                AIMessageChunk(id="resp_first", content="First "),
                AIMessageChunk(id="lc_run_a", content="chunk"),
                AIMessageChunk(id="resp_second", content="Second "),
                AIMessageChunk(id="lc_run_b", content="chunk"),
            ],
            ["resp_first", "resp_second"],
            ["First chunk", "Second chunk"],
        ),
        pytest.param(
            [AIMessageChunk(id=None, content="No id chunk")],
            [None],
            ["No id chunk"],
        ),
    ],
)
def test_append_chunk_to_ui_chat_log_with_id_replacement(
    checkpoint_notifier,
    chunks,
    expected_log_ids,
    expected_log_contents,
):
    for chunk in chunks:
        checkpoint_notifier._append_chunk_to_ui_chat_log(chunk)

    ui_chat_log = checkpoint_notifier.ui_chat_log
    message_ids = [msg["message_id"] for msg in ui_chat_log]
    contents = [msg["content"] for msg in ui_chat_log]

    assert len(ui_chat_log) == len(expected_log_ids)
    assert message_ids == expected_log_ids
    assert contents == expected_log_contents


@pytest.mark.parametrize(
    ("incoming_messages", "expected_final_id", "expected_current_resp_id"),
    [
        pytest.param(
            [AIMessageChunk(id="resp_abc123", content="hello")],
            "resp_abc123",
            "resp_abc123",
        ),
        pytest.param(
            [
                AIMessageChunk(id="resp_abc123", content="start"),
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
            ],
            "resp_abc123",
            "resp_abc123",
        ),
        pytest.param(
            [AIMessageChunk(id="lc_run_xyz", content="chunk")],
            "lc_run_xyz",
            None,
        ),
        pytest.param(
            [
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
            ],
            "lc_run_xyz",
            None,
        ),
        pytest.param(
            [
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
                AIMessageChunk(id="lc_run_2", content="chunk"),
                AIMessageChunk(id="lc_run_3", content="chunk"),
            ],
            "lc_run_3",
            None,
        ),
        pytest.param(
            [
                AIMessageChunk(id="resp_abc123", content="start"),
                AIMessageChunk(id="some_other_id", content="other"),
            ],
            "some_other_id",
            "resp_abc123",
        ),
        pytest.param(
            [
                AIMessageChunk(id="resp_first", content="first"),
                AIMessageChunk(id="resp_second", content="second"),
                AIMessageChunk(id="lc_run_xyz", content="chunk"),
            ],
            "resp_second",
            "resp_second",
        ),
        pytest.param(
            [AIMessageChunk(id=None, content="no id")],
            None,
            None,
        ),
        pytest.param(
            [
                AIMessageChunk(id="resp_abc123", content="start"),
                AIMessageChunk(id=None, content="no id"),
            ],
            None,
            "resp_abc123",
        ),
    ],
)
def test_replace_langchain_id_with_open_ai_id(
    checkpoint_notifier,
    incoming_messages,
    expected_final_id,
    expected_current_resp_id,
):
    for message in incoming_messages:
        checkpoint_notifier._replace_langchain_id_with_open_ai_id(message)

    assert incoming_messages[-1].id == expected_final_id
    assert checkpoint_notifier.current_resp_id == expected_current_resp_id


@pytest.mark.asyncio
async def test_throttle_skips_checkpoint_within_window(checkpoint_notifier):
    """Rapid message events within the throttle window should only enqueue once."""
    message = AIMessageChunk(id="msg-1", content="token")

    with patch(
        "duo_workflow_service.checkpointer.notifier.CHECKPOINT_THROTTLE_SECONDS", 0.1
    ):
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        await checkpoint_notifier.send_event("messages", (message, {}), True)

    # Only the first call should have enqueued - the rest are within the throttle window
    assert checkpoint_notifier.outbox.put_action.call_count == 1


@pytest.mark.asyncio
async def test_throttle_allows_checkpoint_after_window(checkpoint_notifier):
    """A message event after the throttle window should enqueue a new checkpoint."""
    message = AIMessageChunk(id="msg-1", content="token")

    with patch(
        "duo_workflow_service.checkpointer.notifier.CHECKPOINT_THROTTLE_SECONDS", 0.05
    ):
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        # Simulate time passing beyond the throttle window
        checkpoint_notifier._throttle.last_enqueued_at -= 0.1
        await checkpoint_notifier.send_event("messages", (message, {}), True)

    assert checkpoint_notifier.outbox.put_action.call_count == 2


@pytest.mark.asyncio
async def test_throttle_trailing_edge_delivers_last_chunk(checkpoint_notifier):
    """The trailing edge task must fire after the throttle window to deliver the last chunk."""
    message = AIMessageChunk(id="msg-1", content="token")

    with patch(
        "duo_workflow_service.checkpointer.notifier.CHECKPOINT_THROTTLE_SECONDS", 0.05
    ):
        # First call enqueues immediately
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        assert checkpoint_notifier.outbox.put_action.call_count == 1

        # Second call is within window - skipped but trailing task scheduled
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        assert checkpoint_notifier.outbox.put_action.call_count == 1
        assert checkpoint_notifier._throttle.trailing_task is not None
        trailing_task = checkpoint_notifier._throttle.trailing_task

        # Verify the trailing task will eventually fire (without real sleep to avoid flakiness)
        assert not trailing_task.done()

        # Clean up: cancel the pending trailing task so it does not leak into
        # subsequent tests or cause the event loop to wait for it.
        trailing_task.cancel()
        try:
            await trailing_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_throttle_trailing_edge_replaced_by_new_chunk(checkpoint_notifier):
    message = AIMessageChunk(id="msg-1", content="token")

    with patch(
        "duo_workflow_service.checkpointer.notifier.CHECKPOINT_THROTTLE_SECONDS", 0.05
    ):
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        first_task = checkpoint_notifier._throttle.trailing_task

        await checkpoint_notifier.send_event("messages", (message, {}), True)
        second_task = checkpoint_notifier._throttle.trailing_task

    # Each within-window call replaces the trailing task with a new one
    assert second_task is not first_task
    assert first_task is not None
    assert second_task is not None

    # Clean up: cancel the pending trailing task so it does not leak into
    # subsequent tests or cause the event loop to wait for it.
    # Note: first_task was already cancelled by the third send_event call above.
    if second_task and not second_task.done():
        second_task.cancel()
        try:
            await second_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_values_event_bypasses_throttle(checkpoint_notifier):
    """Values events (status updates) must always enqueue immediately, ignoring throttle."""
    state = {"status": WorkflowStatusEnum.EXECUTION, "ui_chat_log": [], "plan": {}}
    message = AIMessageChunk(id="msg-1", content="token")

    with patch(
        "duo_workflow_service.checkpointer.notifier.CHECKPOINT_THROTTLE_SECONDS", 0.1
    ):
        # Exhaust the throttle window with a messages event
        await checkpoint_notifier.send_event("messages", (message, {}), True)
        assert checkpoint_notifier.outbox.put_action.call_count == 1

        # Values event must still enqueue immediately despite being within the window
        await checkpoint_notifier.send_event("values", state, False)
        assert checkpoint_notifier.outbox.put_action.call_count == 2


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


def _ai(total_tokens: int) -> AIMessage:
    """Build a token-bearing AIMessage with the given cumulative total."""
    return AIMessage(
        content="response",
        usage_metadata={
            "input_tokens": total_tokens - 1,
            "output_tokens": 1,
            "total_tokens": total_tokens,
        },
    )


@pytest.mark.parametrize(
    ("conversation_history", "expected"),
    [
        ({}, {}),
        ({"agent": []}, {}),  # empty list skipped
        (
            {"agent": [_ai(1234)], "executor": [_ai(5678)]},
            {"agent": 1234, "executor": 5678},
        ),
    ],
)
def test_agent_token_totals(conversation_history, expected):
    assert _agent_token_totals(conversation_history) == expected


@pytest.mark.asyncio
async def test_agent_context_usage_emission(checkpoint_notifier):
    """Stamped per-agent limit wins; unstamped agents fall back to global; empty agents skipped."""
    state = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "plan": {},
        "conversation_history": {
            "agent": [_ai(1234)],
            "developer_agent": [_ai(5678)],
            "empty_agent": [],
        },
        # Only "agent" runs a non-default model; developer_agent falls back.
        "agent_context_limits": {"agent": 64_000},
    }

    with patch(
        "duo_workflow_service.checkpointer.notifier.get_current_model_max_context_token_limit",
        return_value=200_000,
    ):
        await checkpoint_notifier.send_event("values", state, False)
        checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    usage = checkpoint.agent_context_usage
    assert "empty_agent" not in usage and len(usage) == 2
    assert usage["agent"].total_tokens == 1234
    assert usage["agent"].max_tokens == 64_000
    assert usage["developer_agent"].total_tokens == 5678
    assert usage["developer_agent"].max_tokens == 200_000


@pytest.mark.asyncio
async def test_agent_context_usage_reset_on_subsequent_event(checkpoint_notifier):
    """A later values event with no conversation_history clears the map."""
    populated_state = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "plan": {},
        "conversation_history": {"agent": [_ai(1234)]},
    }

    with patch(
        "duo_workflow_service.checkpointer.notifier.get_current_model_max_context_token_limit",
        return_value=200_000,
    ):
        await checkpoint_notifier.send_event("values", populated_state, False)
        first_checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    assert "agent" in first_checkpoint.agent_context_usage
    assert first_checkpoint.agent_context_usage["agent"].total_tokens == 1234

    # Second event without conversation_history must clear the stored map.
    reset_state = {
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "plan": {},
    }
    await checkpoint_notifier.send_event("values", reset_state, False)
    second_checkpoint = checkpoint_notifier.most_recent_new_checkpoint()

    assert len(second_checkpoint.agent_context_usage) == 0


def _node_events_from(checkpoint) -> list[dict]:
    return loads(checkpoint.checkpoint)["channel_values"].get("node_events")


def test_checkpoint_includes_node_events_when_present(
    outbox,
    gl_version_18_7,  # pylint: disable=unused-argument  # sets capabilities version
):
    client_capabilities.set({"incremental_streaming"})
    log = NodeEventLog()
    log.record("run-1", "researcher", NodePhase.STARTED)
    log.record("run-1", "researcher", NodePhase.ENDED)
    notifier = UserInterface(outbox=outbox, goal="test_goal", node_event_log=log)
    notifier.status = WorkflowStatusEnum.EXECUTION

    assert _node_events_from(notifier.most_recent_new_checkpoint()) == [
        {"run_id": "run-1", "component": "researcher", "phase": "started"},
        {"run_id": "run-1", "component": "researcher", "phase": "ended"},
    ]


def test_checkpoint_omits_node_events_when_log_empty(outbox):
    notifier = UserInterface(
        outbox=outbox, goal="test_goal", node_event_log=NodeEventLog()
    )
    notifier.status = WorkflowStatusEnum.EXECUTION

    channel_values = loads(notifier.most_recent_new_checkpoint().checkpoint)[
        "channel_values"
    ]

    assert "node_events" not in channel_values


def test_checkpoint_omits_node_events_when_no_log(outbox):
    notifier = UserInterface(outbox=outbox, goal="test_goal")
    notifier.status = WorkflowStatusEnum.EXECUTION

    channel_values = loads(notifier.most_recent_new_checkpoint().checkpoint)[
        "channel_values"
    ]

    assert "node_events" not in channel_values


def test_node_events_sent_incrementally_advancing_only_on_send(
    outbox,
    gl_version_18_7,  # pylint: disable=unused-argument  # sets capabilities version
):
    # The cursor advances only when a checkpoint is composed for sending, so each
    # checkpoint carries only the events appended since the previous one.
    client_capabilities.set({"incremental_streaming"})
    log = NodeEventLog()
    notifier = UserInterface(outbox=outbox, goal="test_goal", node_event_log=log)
    notifier.status = WorkflowStatusEnum.EXECUTION

    log.record("run-1", "build_context", NodePhase.STARTED)
    assert _node_events_from(notifier.most_recent_new_checkpoint()) == [
        {"run_id": "run-1", "component": "build_context", "phase": "started"},
    ]

    log.record("run-1", "build_context", NodePhase.ENDED)
    log.record("run-2", "researcher", NodePhase.STARTED)
    assert _node_events_from(notifier.most_recent_new_checkpoint()) == [
        {"run_id": "run-1", "component": "build_context", "phase": "ended"},
        {"run_id": "run-2", "component": "researcher", "phase": "started"},
    ]

    # Nothing new appended -> key omitted entirely.
    channel_values = loads(notifier.most_recent_new_checkpoint().checkpoint)[
        "channel_values"
    ]
    assert "node_events" not in channel_values


def test_node_events_sent_in_full_without_incremental_streaming(outbox):
    # Without the incremental capability the full log is resent each checkpoint;
    # the client deduplicates by (run_id, phase).
    client_capabilities.set(set())
    log = NodeEventLog()
    notifier = UserInterface(outbox=outbox, goal="test_goal", node_event_log=log)
    notifier.status = WorkflowStatusEnum.EXECUTION

    log.record("run-1", "researcher", NodePhase.STARTED)
    assert _node_events_from(notifier.most_recent_new_checkpoint()) == [
        {"run_id": "run-1", "component": "researcher", "phase": "started"},
    ]

    log.record("run-1", "researcher", NodePhase.ENDED)
    assert _node_events_from(notifier.most_recent_new_checkpoint()) == [
        {"run_id": "run-1", "component": "researcher", "phase": "started"},
        {"run_id": "run-1", "component": "researcher", "phase": "ended"},
    ]


# ---------------------------------------------------------------------------
# Tests for secret redaction in streamed content
# ---------------------------------------------------------------------------


class TestStreamingSecretRedaction:
    """Tests that secret redaction is applied to streamed LLM chunks."""

    @pytest.fixture(name="outbox")
    def outbox_fixture(self) -> MagicMock:
        return MagicMock(spec=Outbox())

    @pytest.mark.asyncio
    async def test_redact_secrets_for_ui_called_on_first_chunk(self, outbox):
        """redact_secrets_for_ui is called when the first chunk of a message arrives."""
        notifier = UserInterface(outbox=outbox, goal="goal")
        original_content = "some content"
        redacted_content = "redacted content"
        message = AIMessageChunk(id="msg-1", content=original_content)

        with patch(
            "duo_workflow_service.checkpointer.notifier.redact_secrets_for_ui",
            return_value=redacted_content,
        ) as mock_redact:
            await notifier.send_event("messages", (message, {}), True)

        mock_redact.assert_called_once_with(original_content, tool_name="streaming")
        assert len(notifier.ui_chat_log) == 1
        assert notifier.ui_chat_log[0]["content"] == redacted_content

    @pytest.mark.asyncio
    async def test_redact_secrets_for_ui_called_on_accumulated_chunks(self, outbox):
        """redact_secrets_for_ui is called with the full accumulated text on each chunk."""
        notifier = UserInterface(outbox=outbox, goal="goal")
        chunk1 = AIMessageChunk(id="msg-1", content="Hello ")
        chunk2 = AIMessageChunk(id="msg-1", content="world")

        with patch(
            "duo_workflow_service.checkpointer.notifier.redact_secrets_for_ui",
            side_effect=lambda text, **_: text,
        ) as mock_redact:
            await notifier.send_event("messages", (chunk1, {}), True)
            await notifier.send_event("messages", (chunk2, {}), True)

        # First call: just the first chunk text
        assert mock_redact.call_args_list[0].args[0] == "Hello "
        # Second call: accumulated text from both chunks
        assert mock_redact.call_args_list[1].args[0] == "Hello world"
        assert len(notifier.ui_chat_log) == 1
        assert notifier.ui_chat_log[0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_safe_content_not_modified(self, outbox):
        """When redact_secrets_for_ui returns the same content, it is stored as-is."""
        notifier = UserInterface(outbox=outbox, goal="goal")
        safe_content = "Here is the result of your request."
        message = AIMessageChunk(id="msg-1", content=safe_content)

        # Use the real redact_secrets_for_ui -- safe content should pass through unchanged
        await notifier.send_event("messages", (message, {}), True)

        assert len(notifier.ui_chat_log) == 1
        assert notifier.ui_chat_log[0]["content"] == safe_content


# ---------------------------------------------------------------------------
# Tests for _find_ui_chat_log_entry
# ---------------------------------------------------------------------------


class TestFindUiChatLogEntry:
    """Tests for looking up UI chat log entries by identity rather than position."""

    def test_returns_none_when_message_id_is_falsy(self, checkpoint_notifier):
        checkpoint_notifier.ui_chat_log = [{"message_id": "msg-1"}]

        assert checkpoint_notifier._find_ui_chat_log_entry(None) is None
        assert checkpoint_notifier._find_ui_chat_log_entry("") is None

    def test_returns_none_when_no_entry_matches(self, checkpoint_notifier):
        checkpoint_notifier.ui_chat_log = [{"message_id": "msg-1"}]

        assert checkpoint_notifier._find_ui_chat_log_entry("missing") is None

    def test_finds_entry_by_id_regardless_of_position(self, checkpoint_notifier):
        target = {"message_id": "msg-1", "message_type": MessageTypeEnum.AGENT}
        checkpoint_notifier.ui_chat_log = [
            target,
            {"message_id": "msg-2", "message_type": MessageTypeEnum.TOOL},
        ]

        assert checkpoint_notifier._find_ui_chat_log_entry("msg-1") is target

    def test_message_type_filter_excludes_mismatched_type(self, checkpoint_notifier):
        checkpoint_notifier.ui_chat_log = [
            {"message_id": "msg-1", "message_type": MessageTypeEnum.TOOL},
        ]

        assert (
            checkpoint_notifier._find_ui_chat_log_entry(
                "msg-1", message_type=MessageTypeEnum.AGENT
            )
            is None
        )

    def test_returns_most_recent_match_when_ids_repeat(self, checkpoint_notifier):
        # Ids aren't expected to repeat in practice, but the lookup scans from
        # the end, so if they ever did, the most recent one wins.
        older = {"message_id": "msg-1", "message_type": MessageTypeEnum.AGENT}
        newer = {"message_id": "msg-1", "message_type": MessageTypeEnum.AGENT}
        checkpoint_notifier.ui_chat_log = [older, newer]

        assert checkpoint_notifier._find_ui_chat_log_entry("msg-1") is newer


# ---------------------------------------------------------------------------
# Tests for identity-based lookup in _append_chunk_to_ui_chat_log
# ---------------------------------------------------------------------------


class TestAppendChunkIdentityLookup:
    """The AGENT entry being streamed into is located by id, not by assuming it is the last item in the log, since other
    entries may be appended after it in between chunks of the same message."""

    def test_continuation_updates_agent_entry_even_if_not_last(
        self, checkpoint_notifier
    ):
        first_chunk = AIMessageChunk(id="run-1", content="Hello")
        checkpoint_notifier._append_chunk_to_ui_chat_log(first_chunk)

        # Something else (e.g. a tool card from another code path) gets
        # appended after the AGENT entry before the next chunk arrives.
        other_entry = {
            "message_id": "other-id",
            "message_type": MessageTypeEnum.TOOL,
            "content": "unrelated",
        }
        checkpoint_notifier.ui_chat_log.append(other_entry)

        second_chunk = AIMessageChunk(id="run-1", content=" world")
        checkpoint_notifier._append_chunk_to_ui_chat_log(second_chunk)

        agent_entries = [
            entry
            for entry in checkpoint_notifier.ui_chat_log
            if entry["message_type"] == MessageTypeEnum.AGENT
        ]
        assert len(agent_entries) == 1
        assert agent_entries[0]["content"] == "Hello world"
        # The unrelated entry appended in between is untouched.
        assert other_entry in checkpoint_notifier.ui_chat_log
        assert other_entry["content"] == "unrelated"

    def test_continuation_silently_skips_when_agent_entry_removed(
        self, checkpoint_notifier
    ):
        # Establish the first chunk so latest_ai_message is set.
        first_chunk = AIMessageChunk(id="run-1", content="Hello")
        checkpoint_notifier._append_chunk_to_ui_chat_log(first_chunk)

        # Simulate the AGENT entry being removed from ui_chat_log between
        # chunks (e.g. a state reset).  _find_ui_chat_log_entry will return
        # None for the continuation chunk, and the update must be silently
        # dropped rather than raising an exception.
        checkpoint_notifier.ui_chat_log.clear()

        second_chunk = AIMessageChunk(id="run-1", content=" world")
        # Must not raise.
        checkpoint_notifier._append_chunk_to_ui_chat_log(second_chunk)

        # The log remains empty — the silent-drop behavior is intentional.
        assert checkpoint_notifier.ui_chat_log == []


# ---------------------------------------------------------------------------
# Tests for _merge_ui_chat_log
# ---------------------------------------------------------------------------


class TestMergeUiChatLog:
    """Tests for merging the authoritative graph-state log with local PENDING entries not yet reflected in state."""

    def test_new_log_is_returned_unchanged_when_nothing_pending_locally(
        self, checkpoint_notifier
    ):
        new_log = [{"message_id": "msg-1", "status": None}]
        checkpoint_notifier.ui_chat_log = list(new_log)

        assert checkpoint_notifier._merge_ui_chat_log(new_log) == new_log

    def test_pending_entry_missing_from_new_log_is_preserved(self, checkpoint_notifier):
        pending_entry = {"message_id": "call-1", "status": ToolStatus.PENDING}
        checkpoint_notifier.ui_chat_log = [pending_entry]
        new_log = [{"message_id": "msg-1", "status": None}]

        merged = checkpoint_notifier._merge_ui_chat_log(new_log)

        assert merged == [{"message_id": "msg-1", "status": None}, pending_entry]

    def test_non_pending_entry_missing_from_new_log_is_not_resurrected(
        self, checkpoint_notifier
    ):
        # Regression test: entries deliberately removed from authoritative
        # state (e.g. a reset to an empty log) must not be brought back just
        # because they aren't PENDING placeholders.
        stale_entry = {"message_id": "msg-1", "status": None}
        checkpoint_notifier.ui_chat_log = [stale_entry]

        assert checkpoint_notifier._merge_ui_chat_log([]) == []

    def test_pending_entry_already_reflected_in_new_log_is_not_duplicated(
        self, checkpoint_notifier
    ):
        checkpoint_notifier.ui_chat_log = [
            {"message_id": "call-1", "status": ToolStatus.PENDING}
        ]
        new_log = [{"message_id": "call-1", "status": ToolStatus.SUCCESS}]

        merged = checkpoint_notifier._merge_ui_chat_log(new_log)

        assert merged == new_log

    def test_merge_with_no_local_state_returns_new_log_unchanged(
        self, checkpoint_notifier
    ):
        new_log = [{"message_id": "msg-1", "status": None}]

        assert checkpoint_notifier._merge_ui_chat_log(new_log) == new_log


# ---------------------------------------------------------------------------
# Tests for the incremental-send cursor in _pop_recent_ui_chat_log_changes
# ---------------------------------------------------------------------------


class TestPopRecentUiChatLogChangesCursor:
    """Tests that the incremental-send cursor tracks the entry that might still receive further updates, rather than
    whatever is last in the list."""

    def test_falls_back_to_last_entry_id_when_not_streaming(self, checkpoint_notifier):
        # Plain "values" updates never set latest_ai_message; the cursor must
        # still advance to the last entry, exactly as before this was
        # generalized to consider latest_ai_message.
        checkpoint_notifier.ui_chat_log = [
            {"message_id": "msg-1"},
            {"message_id": "msg-2"},
        ]

        checkpoint_notifier._pop_recent_ui_chat_log_changes()

        assert checkpoint_notifier.last_sent_ui_message_id == "msg-2"

    def test_tracks_latest_ai_message_id_while_streaming(self, checkpoint_notifier):
        # While streaming, entries appended after the current AI message's
        # entry (e.g. by another component) must still be resent on the next
        # call, so the cursor tracks the AI message's id rather than the
        # position of the last entry in the list.
        checkpoint_notifier.latest_ai_message = AIMessageChunk(
            id="run-1", content="hello"
        )
        checkpoint_notifier.ui_chat_log = [
            {"message_id": "run-1"},
            {"message_id": "other-id"},
        ]

        checkpoint_notifier._pop_recent_ui_chat_log_changes()

        assert checkpoint_notifier.last_sent_ui_message_id == "run-1"

    def test_entry_updated_after_cursor_moved_past_it_is_still_resent(
        self, checkpoint_notifier
    ):
        # Regression test: previously the cursor was always set to the last
        # entry's id, so once a later entry had been sent, an earlier entry
        # that kept changing would never be included in a future checkpoint.
        checkpoint_notifier.latest_ai_message = AIMessageChunk(
            id="run-1", content="hello"
        )
        entry_1 = {"message_id": "run-1", "content": "v1"}
        entry_2 = {"message_id": "other-id", "content": "unchanged"}
        checkpoint_notifier.ui_chat_log = [entry_1, entry_2]

        first_sent = checkpoint_notifier._pop_recent_ui_chat_log_changes()
        assert [e["message_id"] for e in first_sent] == ["run-1", "other-id"]

        # entry_1 keeps changing even though entry_2 (positioned after it)
        # was already sent.
        entry_1["content"] = "v2"

        second_sent = checkpoint_notifier._pop_recent_ui_chat_log_changes()
        entry_1_resent = next(
            (e for e in second_sent if e["message_id"] == "run-1"), None
        )
        assert entry_1_resent is not None
        assert entry_1_resent["content"] == "v2"
