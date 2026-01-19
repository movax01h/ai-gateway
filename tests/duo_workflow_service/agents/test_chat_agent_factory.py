from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.chat_agent_factory import create_agent
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.tools.toolset import Toolset
from lib.feature_flags.context import FeatureFlag
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
    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_with_prompt_registry(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        prompt,
    ):
        # Mock feature flag and client capability to return False by default
        mock_is_feature_enabled.return_value = False
        mock_is_client_capable.return_value = False

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
            bind_tools_params={},
            internal_event_extra={
                "agent_name": "chat",
                "workflow_id": "workflow_123",
                "workflow_type": CategoryEnum.WORKFLOW_CHAT,
            },
        )

    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_with_agent_name_override(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        prompt,
    ):
        """Test that agent_name_override is used for chat-partial foundational agents."""
        # Mock feature flag and client capability to return False by default
        mock_is_feature_enabled.return_value = False
        mock_is_client_capable.return_value = False

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
            bind_tools_params={},
            internal_event_extra={
                "agent_name": "348/0",  # Should use the override
                "workflow_id": "workflow_123",
                "workflow_type": CategoryEnum.AI_CATALOG_AGENT,
            },
        )

    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_without_override_defaults_to_chat(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        """Test that agent_name defaults to 'chat' when no override is provided."""
        # Mock feature flag and client capability to return False by default
        mock_is_feature_enabled.return_value = False
        mock_is_client_capable.return_value = False

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
        assert call_kwargs["bind_tools_params"] == {}

    @pytest.mark.parametrize(
        "feature_enabled,client_capable,expected_params,should_check_capability,test_description",
        [
            (
                True,
                True,
                {"web_search_options": {}},
                True,
                "both feature flag and client capability enabled",
            ),
            (
                False,
                True,
                {},
                False,
                "feature flag disabled but client capable",
            ),
            (
                True,
                False,
                {},
                True,
                "feature flag enabled but client not capable",
            ),
            (
                False,
                False,
                {},
                False,
                "both feature flag and client capability disabled",
            ),
        ],
    )
    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_web_search_options_conditional(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        feature_enabled,
        client_capable,
        expected_params,
        should_check_capability,
        test_description,
    ):
        """Test that web_search_options is conditionally included based on feature flag and client capability."""
        mock_is_feature_enabled.return_value = feature_enabled
        mock_is_client_capable.return_value = client_capable

        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override=None,
        )

        assert isinstance(agent, ChatAgent)

        # Verify bind_tools_params matches expected
        call_kwargs = mock_local_prompt_registry.get_on_behalf.call_args.kwargs
        assert (
            call_kwargs["bind_tools_params"] == expected_params
        ), f"Failed for scenario: {test_description}"

        # Verify the feature flag was always checked
        mock_is_feature_enabled.assert_called_once_with(FeatureFlag.DAP_WEB_SEARCH)

        # Verify client capability was checked only when feature flag is enabled
        # (due to short-circuit evaluation of the 'and' operator)
        if should_check_capability:
            mock_is_client_capable.assert_called_once_with("web_search")
        else:
            mock_is_client_capable.assert_not_called()
