# These tests cover a YAML flow config, so there is no module to name the file after.
# pylint: disable=file-naming-for-tests
"""Tests for the experimental slack_assistant config, which selects its tools with the tools: key."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from ai_gateway.prompts.base import jinja2_formatter
from duo_workflow_service.agent_platform.experimental.flows.base import Flow
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.tools_config import (
    parse_component_tools,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState
from duo_workflow_service.agent_platform.experimental.state.base import IOKey
from duo_workflow_service.agent_platform.v1.flows.validation import (
    _make_validation_tools_registry,
)
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


def _agent():
    config = FlowConfig.from_yaml_config("slack_assistant", "1.0.0")
    return next(c for c in config.components if c["name"] == "slack_agent")


def _selectors():
    selectors = parse_component_tools(_agent())
    assert selectors is not None
    return {selector.server: selector.names for selector in selectors}


def test_the_agent_selects_slack_orbit_and_gitlab_tools_by_server_and_name():
    selectors = _selectors()

    assert set(selectors) == {"slack", "orbit", "gitlab"}
    assert selectors["slack"] == [
        "slack_search_public",
        "slack_search_channels",
        "slack_read_channel",
        "slack_read_thread",
    ]
    assert selectors["orbit"] == ["list_commands", "invoke_command"]
    assert "toolset" not in _agent()


def test_every_gitlab_name_is_a_native_tool():
    """No name under gitlab needs the GitLab MCP server, so the flow still works where Rails offers none."""
    registry = _make_validation_tools_registry()

    missing = [name for name in _selectors()["gitlab"] if registry.get(name) is None]

    assert missing == []


def _rendered_user_prompt(additional_context: list[AdditionalContext]) -> str:
    """Render the user prompt from a Rails envelope, the way the experimental Flow parses it."""
    config = FlowConfig.from_yaml_config("slack_assistant", "1.0.0")
    agent = _agent()

    # Only the input processing is under test, so skip Flow.__init__ and its container wiring.
    flow = Flow.__new__(Flow)
    flow._config = config
    flow.log = MagicMock()
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
    assert prompt.prompt_template is not None
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
