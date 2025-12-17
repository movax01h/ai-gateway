from unittest.mock import Mock

import pytest

from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.chat_agent_factory import create_agent
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_tools_registry")
def mock_tools_registry_fixture():
    return Mock(spec=ToolsRegistry)


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock_toolset = Mock(spec=Toolset)
    mock_toolset.bindable = []
    return mock_toolset


class TestCreateAgent:
    def test_create_agent_with_prompt_registry(
        self,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        prompt,
    ):
        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override="test_system_template",
        )

        assert isinstance(agent, ChatAgent)
        assert agent.name == prompt.name
        assert agent.system_template_override == "test_system_template"

        mock_local_prompt_registry.get_on_behalf.assert_called_once_with(
            user=user,
            prompt_id="chat/agent",
            prompt_version="^1.0.0",
            internal_event_category="test_category",
            tools=[],
            internal_event_extra={
                "agent_name": "chat",
                "workflow_id": "workflow_123",
                "workflow_type": CategoryEnum.WORKFLOW_CHAT,
            },
        )

    def test_create_agent_with_agent_name_override(
        self,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        prompt,
    ):
        """Test that agent_name_override is used for chat-partial foundational agents."""
        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.AI_CATALOG_AGENT,
            system_template_override=None,
            agent_name_override="348/0",
        )

        assert isinstance(agent, ChatAgent)
        assert agent.name == prompt.name

        # Verify that the agent_name in internal_event_extra uses the override
        mock_local_prompt_registry.get_on_behalf.assert_called_once_with(
            user=user,
            prompt_id="chat/agent",
            prompt_version="^1.0.0",
            internal_event_category="test_category",
            tools=[],
            internal_event_extra={
                "agent_name": "348/0",  # Should use the override
                "workflow_id": "workflow_123",
                "workflow_type": CategoryEnum.AI_CATALOG_AGENT,
            },
        )

    def test_create_agent_without_override_defaults_to_chat(
        self,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        """Test that agent_name defaults to 'chat' when no override is provided."""
        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override=None,
            agent_name_override=None,
        )

        assert isinstance(agent, ChatAgent)

        # Verify that the agent_name defaults to "chat"
        call_kwargs = mock_local_prompt_registry.get_on_behalf.call_args.kwargs
        assert call_kwargs["internal_event_extra"]["agent_name"] == "chat"
