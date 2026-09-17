from functools import partial

import pytest

from duo_workflow_service.agent_platform.v1.chat_engine import (
    ChatFlow,
    normalize_engine_owned_config,
)
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import PartialFlowConfig
from lib.events import GLReportingEventContext


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
def test_chat_flow_builds_through_the_registry_factory(user):
    config = normalize_engine_owned_config(
        PartialFlowConfig(
            version="v1",
            environment="chat-partial",
            components=[
                {"name": "chat_agent", "type": "AgentComponent", "toolset": []}
            ],
        )
    )
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

    assert isinstance(flow, Flow)
    assert flow._config is config
