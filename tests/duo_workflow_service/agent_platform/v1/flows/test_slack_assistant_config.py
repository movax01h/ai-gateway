"""Guard for the shipped slack_assistant/1.0.0 flow config.

Rails sends the Slack reply rules as an ``agent_platform_client_context`` envelope. Each step from envelope to prompt
fails silently: an undeclared category is skipped, an optional binding with a wrong path resolves to ``None``, and a
bound variable the template never renders is dropped. This test runs the envelope through all three.
"""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from ai_gateway.prompts.base import jinja2_formatter
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.agent_platform.v1.state.base import IOKey
from duo_workflow_service.workflows.type_definitions import AdditionalContext

CLIENT_CONTEXT = (
    "You were mentioned in a Slack conversation.\n<formatting>Use <@U...>.</formatting>"
)

# The shape Rails' Envelope.to_wire sends: content is a JSON object, metadata carries the version.
RAILS_ENVELOPE = AdditionalContext(
    category="agent_platform_client_context",
    content=json.dumps({"content": CLIENT_CONTEXT}),
    metadata={"version": "1.0.0"},
)


def _rendered_user_prompt(additional_context: list[AdditionalContext]) -> str:
    config = FlowConfig.from_yaml_config("slack_assistant", "1.0.0")
    agent = next(c for c in config.components if c["name"] == "slack_agent")

    # Only the input processing is under test, so skip Flow.__init__ and its container wiring.
    flow = Flow.__new__(Flow)
    flow._config = config
    flow.log = MagicMock()
    # Only the keys the bindings read; the rest of FlowState is irrelevant here.
    state = cast(
        FlowState,
        {
            "context": {
                "goal": "the goal",
                "current_date": "2026-10-05",
                "inputs": flow._process_additional_context(additional_context),
            }
        },
    )

    variables: dict = {}
    for key in IOKey.parse_keys(agent["inputs"]):
        variables.update(key.template_variable_from_state(state))

    assert config.prompts is not None
    prompt = next(p for p in config.prompts if p.prompt_id == agent["prompt_id"])
    user = prompt.prompt_template["user"]
    assert isinstance(user, str)
    return jinja2_formatter(user, **variables)


@pytest.mark.parametrize(
    ("additional_context", "expected_block"),
    [
        ([RAILS_ENVELOPE], f"<client_context>\n{CLIENT_CONTEXT}\n</client_context>"),
        ([], None),
    ],
    ids=["with_client_context", "without_client_context"],
)
def test_client_context_reaches_the_user_prompt(additional_context, expected_block):
    rendered = _rendered_user_prompt(additional_context)

    if expected_block:
        assert expected_block in rendered
    else:
        assert "<client_context>" not in rendered
    assert rendered.rstrip().endswith("the goal")
