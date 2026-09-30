from duo_workflow_service.agent_platform.v1.chat_engine import (
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


def test_the_config_completes_through_to_config():
    normalized = normalize_engine_owned_config(build_config())

    assert type(normalized) is FlowConfig
    assert normalized.routers == [{"from": "chat_agent", "to": "end"}]


def test_components_pass_through_untouched():
    component = {
        "name": "chat_agent",
        "type": "AgentComponent",
        "toolset": [],
        "pre_approved_tools": ["read_file"],
    }

    normalized = normalize_engine_owned_config(build_config(dict(component)))

    assert normalized.components == [component]


def test_input_config_is_not_mutated():
    config = build_config()

    normalized = normalize_engine_owned_config(config)

    assert normalized is not config
    assert config.routers is None
    assert config.flow is None
