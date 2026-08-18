# An end-to-end test spanning the server, the flow and the notifier, so there is no
# single source file for the checker to pair this name with.
# pylint: disable=file-naming-for-tests
"""Verify that pausing a flow sends a single client notification."""

from typing import Any

import grpc
import pytest

from duo_workflow_service.server import DuoWorkflowService
from tests.duo_workflow_service.integration.conftest import (
    FakeExecutor,
    run_exchange,
    start_inline_config_event,
)

PAUSING_FLOW_CONFIG: dict[str, Any] = {
    "version": "v1",
    "environment": "chat",
    "components": [
        {
            "name": "chat_agent",
            "type": "AgentComponent",
            "prompt_id": "pausing_flow_prompt",
            "inputs": [{"from": "context:goal", "as": "goal"}],
            "toolset": [],
            "ui_log_events": ["on_agent_final_answer"],
        },
        {
            "name": "user_input",
            "type": "HumanInputComponent",
            "sends_response_to": "chat_agent",
            "interaction_type": "input",
            "message_template": "no-op",
            "ui_log_events": ["on_user_response"],
        },
    ],
    "routers": [
        {"from": "chat_agent", "to": "user_input"},
        {"from": "user_input", "to": "chat_agent"},
    ],
    "flow": {"entry_point": "chat_agent"},
    "prompts": [
        {
            "name": "pausing_flow_prompt",
            "prompt_id": "pausing_flow_prompt",
            "unit_primitives": ["duo_chat"],
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{goal}}",
            },
        }
    ],
}


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
async def test_flow_pause_notifies_the_client_once(
    servicer: DuoWorkflowService,
    executor: FakeExecutor,
):
    """The request node's checkpoint is withheld; the client hears about the pause once."""
    exchange = await run_exchange(
        servicer,
        executor,
        start_inline_config_event(
            goal="What does this project do?",
            workflow_definition="ai_catalog_agent",
            flow_config=PAUSING_FLOW_CONFIG,
        ),
    )

    assert exchange.code == grpc.StatusCode.OK, exchange.details

    pause_notifications = [
        checkpoint
        for checkpoint in executor.checkpoints()
        if checkpoint.status == "INPUT_REQUIRED"
    ]
    assert len(pause_notifications) == 1, (
        "expected exactly one INPUT_REQUIRED checkpoint notification, got "
        f"{[c.status for c in executor.checkpoints()]}"
    )
