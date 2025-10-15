from unittest.mock import Mock

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.prompts import InMemoryPromptRegistry, Prompt
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


@pytest.fixture(name="mock_in_memory_prompt_registry")
def mock_in_memory_prompt_registry_fixture(mock_prompt):
    mock_registry = Mock(spec=InMemoryPromptRegistry)
    mock_registry.get_on_behalf.return_value = mock_prompt
    return mock_registry


class TestCreateAgent:
    @pytest.mark.parametrize(
        "prompt_id,prompt_version,workflow_id,registry_fixture",
        [
            ("chat/agent", "^1.0.0", "workflow_123", "mock_local_prompt_registry"),
            ("custom/prompt", None, "workflow_456", "mock_in_memory_prompt_registry"),
        ],
        ids=["with_local_prompt_registry", "with_in_memory_prompt_registry"],
    )
    def test_create_agent_with_prompt_registry(
        self,
        mock_user,
        mock_tools_registry,
        mock_toolset,
        prompt_id,
        prompt_version,
        workflow_id,
        registry_fixture,
        request,
    ):
        prompt_registry = request.getfixturevalue(registry_fixture)

        agent = create_agent(
            user=mock_user,
            tools_registry=mock_tools_registry,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            model_metadata=None,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=prompt_registry,
            workflow_id=workflow_id,
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
        )

        assert isinstance(agent, ChatAgent)
        assert agent.name == "test_agent"

        prompt_registry.get_on_behalf.assert_called_once_with(
            user=mock_user,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            model_metadata=None,
            internal_event_category="test_category",
            tools=[],
            workflow_id=workflow_id,
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
        )
