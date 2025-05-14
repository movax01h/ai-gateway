import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from langchain.load.dump import dumps

from contract import contract_pb2
from duo_workflow_service.checkpointer.notifier import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
    UserInterface,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum


@pytest.fixture
def outbox():
    return asyncio.Queue()


@pytest.fixture
def checkpoint_notifier(outbox):
    return UserInterface(outbox=outbox, goal="test_goal")


@pytest.mark.asyncio
async def test_send_event_with_non_values_type(checkpoint_notifier):
    state = {"status": WorkflowStatusEnum.EXECUTION, "ui_chat_log": []}
    await checkpoint_notifier.send_event("not_values", None)
    assert checkpoint_notifier.outbox.empty()


@pytest.mark.asyncio
async def test_send_event_with_values_type(checkpoint_notifier):
    state = {
        "status": WorkflowStatusEnum.COMPLETED,
        "ui_chat_log": ["message1", "message2"],
        "plan": {"steps": ["step1", "step2"]},
    }
    await checkpoint_notifier.send_event("values", state)
    assert checkpoint_notifier.ui_chat_log == ["message1", "message2"]
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
    await checkpoint_notifier.send_event("values", state)
    action = await checkpoint_notifier.outbox.get()
    expected_checkpoint = dumps(
        {
            "channel_values": {
                "ui_chat_log": ["message"],
                "plan": {"steps": None},
            }
        }
    )
    assert action.newCheckpoint.checkpoint == expected_checkpoint


def test_workflow_status_mapping():
    # Test that the mapping dictionary contains all necessary workflow statuses
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

    # Check that all expected keys and values are in the actual mapping
    for workflow_status, checkpoint_status in expected_mapping.items():
        assert (
            WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[workflow_status] == checkpoint_status
        )

    # Check that there are no extra keys
    assert len(WORKFLOW_STATUS_TO_CHECKPOINT_STATUS) == len(expected_mapping)


@pytest.mark.asyncio
async def test_init_sets_attributes(outbox):
    notifier = UserInterface(outbox=outbox, goal="custom_goal")
    assert notifier.outbox == outbox
    assert notifier.goal == "custom_goal"
    assert notifier.ui_chat_log == []
