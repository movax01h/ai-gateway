# pylint: disable=file-naming-for-tests
"""Integration test for a legacy AI Catalog item: an inline flow config sent with a `chat` workflow definition.

An AI Catalog item does not live in the server-side flow registry — the client sends the whole flow config in
the start request, and ``workflowDefinition`` is just a label carried along for reporting and billing.

Current catalog items send the generic ``"ai_catalog_agent"`` for that label (Rails:
``Ai::Catalog::ExecuteWorkflowService``). Legacy ones send ``"chat"``, which is also the name of a flow this
service owns natively — but the presence of an inline flowConfig already settles which flow to run, so the
label being a known flow name changes nothing. Both must be served from the inline config.

See ``conftest.py`` for the harness and what it does and does not stand in for.
"""

from typing import Any

import grpc
import pytest

from duo_workflow_service.server import DuoWorkflowService
from tests.duo_workflow_service.integration.conftest import (
    FakeExecutor,
    run_exchange,
    start_inline_config_event,
)

# The flow a catalog item ships in its start request: the same `agent -> user input -> agent` shape as the
# registry's agentic_chat, trimmed to what these assertions need. It carries its own prompt, so nothing has to
# be registered server-side.
CATALOG_ITEM_FLOW_CONFIG: dict[str, Any] = {
    "version": "v1",
    "environment": "chat",
    "components": [
        {
            "name": "chat_agent",
            "type": "AgentComponent",
            "prompt_id": "catalog_item_prompt",
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
            "name": "catalog_item_prompt",
            "prompt_id": "catalog_item_prompt",
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
@pytest.mark.parametrize("workflow_definition", ["ai_catalog_agent", "chat"])
async def test_catalog_item_runs_its_inline_flow_config(
    servicer: DuoWorkflowService,
    executor: FakeExecutor,
    workflow_definition: str,
):
    """A catalog item runs from its inline config, whichever label it sends as the workflow definition.

    ``"ai_catalog_agent"`` is what current catalog items send; ``"chat"`` is what legacy ones send. The
    config is byte-for-byte the same in both cases, so the outcome must be too: the agent answers and the
    flow parks on the human-input interrupt, exactly as the registry-backed chat flow does.
    """
    exchange = await run_exchange(
        servicer,
        executor,
        start_inline_config_event(
            goal="What does this project do?",
            workflow_definition=workflow_definition,
            flow_config=CATALOG_ITEM_FLOW_CONFIG,
        ),
    )

    assert exchange.code == grpc.StatusCode.OK, exchange.details

    checkpoints = executor.checkpoints()
    assert checkpoints, "expected the inline flow to stream at least one checkpoint"

    assert any("mock" in message for message in executor.agent_messages()), (
        f"agent answer missing from the UI chat log: {executor.agent_messages()}"
    )

    assert checkpoints[-1].status == "INPUT_REQUIRED"
