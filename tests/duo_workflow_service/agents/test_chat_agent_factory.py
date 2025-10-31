from unittest.mock import Mock

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.prompts import Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.chat_agent_factory import create_agent
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_user")
def mock_user_fixture():
    return CloudConnectorUser(True, claims=UserClaims(gitlab_realm="test-realm"))


@pytest.fixture(name="mock_tools_registry")
def mock_tools_registry_fixture():
    return Mock(spec=ToolsRegistry)


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock_toolset = Mock(spec=Toolset)
    mock_toolset.bindable = []
    return mock_toolset


@pytest.fixture(name="mock_prompt")
def mock_prompt_fixture():
    mock_prompt = Mock(spec=Prompt)
    mock_prompt.name = "test_agent"
    mock_prompt.prompt_tpl = Mock()
    return mock_prompt


@pytest.fixture(name="mock_local_prompt_registry")
def mock_local_prompt_registry_fixture(mock_prompt):
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_registry.get_on_behalf.return_value = mock_prompt
    return mock_registry


class TestCreateAgent:
    def test_create_agent_with_prompt_registry(
        self,
        mock_user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        agent = create_agent(
            user=mock_user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override="test_system_template",
        )

        assert isinstance(agent, ChatAgent)
        assert agent.name == "test_agent"
        assert agent.system_template_override == "test_system_template"

        mock_local_prompt_registry.get_on_behalf.assert_called_once_with(
            user=mock_user,
            prompt_id="chat/agent",
            prompt_version="^1.0.0",
            internal_event_category="test_category",
            tools=[],
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
        )
