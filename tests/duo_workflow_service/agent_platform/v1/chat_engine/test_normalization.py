import pytest

from duo_workflow_service.agent_platform.v1.chat_engine import (
    ENGINE_FLOOR_UI_LOG_EVENTS,
    normalize_engine_owned_config,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    PartialFlowConfig,
)


def build_config(component=None, **overrides):
    if component is None:
        component = {"name": "chat_agent", "type": "AgentComponent", "toolset": []}

    return PartialFlowConfig(
        version="v1",
        environment="chat-partial",
        components=[component],
        **overrides,
    )


def root(config):
    return config.components[0]


def test_floor_applied_when_ui_log_events_absent():
    normalized = normalize_engine_owned_config(build_config())

    assert root(normalized)["ui_log_events"] == list(ENGINE_FLOOR_UI_LOG_EVENTS)
    assert ENGINE_FLOOR_UI_LOG_EVENTS == (
        "on_agent_final_answer",
        "on_agent_reasoning",
        "on_tool_execution_success",
        "on_tool_execution_failed",
    )


@pytest.mark.parametrize("declared", [["on_agent_final_answer"], []])
def test_declared_ui_log_events_win(declared):
    config = build_config(
        {"name": "chat_agent", "type": "AgentComponent", "ui_log_events": declared}
    )

    assert root(normalize_engine_owned_config(config))["ui_log_events"] == declared


def test_require_tool_approval_defaults_to_true():
    assert (
        root(normalize_engine_owned_config(build_config()))["require_tool_approval"]
        is True
    )


def test_declared_require_tool_approval_wins():
    config = build_config(
        {"name": "chat_agent", "type": "AgentComponent", "require_tool_approval": False}
    )

    assert root(normalize_engine_owned_config(config))["require_tool_approval"] is False


def test_pre_approved_tools_is_dropped():
    config = build_config(
        {
            "name": "chat_agent",
            "type": "AgentComponent",
            "pre_approved_tools": ["read_file"],
        }
    )

    assert "pre_approved_tools" not in root(normalize_engine_owned_config(config))


def test_the_config_completes_through_to_config():
    normalized = normalize_engine_owned_config(build_config())

    assert type(normalized) is FlowConfig
    assert normalized.routers == [{"from": "chat_agent", "to": "end"}]


def test_input_config_is_not_mutated():
    component = {
        "name": "chat_agent",
        "type": "AgentComponent",
        "pre_approved_tools": [],
    }
    config = build_config(dict(component))

    normalized = normalize_engine_owned_config(config)

    assert normalized is not config
    assert config.components == [component]
    assert config.routers is None
    assert config.flow is None


def test_non_agent_components_are_untouched():
    component = {"name": "gate", "type": "HumanInputComponent"}

    assert root(normalize_engine_owned_config(build_config(component))) == component
