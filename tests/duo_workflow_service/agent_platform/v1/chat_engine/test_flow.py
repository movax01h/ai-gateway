from functools import partial
from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.v1.chat_engine import ChatFlow
from duo_workflow_service.agent_platform.v1.chat_engine.graph_builder import (
    ChatGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    PartialFlowConfig,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from lib.events import GLReportingEventContext


def _chat_flow(user) -> tuple[ChatFlow, FlowConfig]:
    config = PartialFlowConfig(
        version="v1",
        environment="chat-partial",
        components=[{"name": "chat_agent", "type": "AgentComponent", "toolset": []}],
    ).to_config()
    factory = partial(ChatFlow, config=config)

    flow = factory(
        workflow_id="workflow-1",
        workflow_metadata={
            "git_url": "https://gitlab.com/test/project",
            "git_sha": "abc",
        },
        workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
        user=user,
    )
    return flow, config


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
def test_chat_flow_builds_through_the_registry_factory(user):
    flow, config = _chat_flow(user)

    assert isinstance(flow, Flow)
    assert flow._config is config


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
def test_chat_flow_builds_its_graph_with_the_chat_graph_builder(user):
    flow, _ = _chat_flow(user)

    builder = flow._graph_builder(Mock(spec=ToolsRegistry))

    assert isinstance(builder, ChatGraphBuilder)
    assert builder._workflow_id == "workflow-1"
