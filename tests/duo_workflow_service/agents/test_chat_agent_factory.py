from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.chat_agent_factory import (
    _extract_manual_compactor,
    create_agent,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.legacy_trim import (
    LegacyTrimOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import CompactionConfig
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
                "agent_name": "348/0",
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

        call_kwargs = mock_local_prompt_registry.get_on_behalf.call_args.kwargs
        assert call_kwargs["bind_tools_params"] == expected_params, (
            f"Failed for scenario: {test_description}"
        )

        assert mock_is_feature_enabled.call_count == 1
        mock_is_feature_enabled.assert_any_call(FeatureFlag.DAP_WEB_SEARCH)

        if should_check_capability:
            mock_is_client_capable.assert_called_once_with("web_search")
        else:
            mock_is_client_capable.assert_not_called()

    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_with_compaction_default_config(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        """When compaction=CompactionConfig() is passed, the pipeline uses a CompactionOptimizer and the manual
        compactor is reused from the pipeline (same object)."""
        mock_is_feature_enabled.return_value = False
        mock_is_client_capable.return_value = False

        cfg = CompactionConfig()
        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override=None,
            compaction=cfg,
        )

        assert isinstance(agent, ChatAgent)
        optimizers = agent._optimizer_pipeline.optimizers
        assert len(optimizers) == 1
        assert isinstance(optimizers[0], CompactionOptimizer)
        assert optimizers[0]._config is cfg

        # The manual compactor is the same object as the pipeline optimizer — no duplicate.
        assert agent._manual_compactor is optimizers[0]

    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_with_custom_compaction_config(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        """Custom CompactionConfig flows through to the pipeline optimizer; the manual compactor is reused from the
        pipeline (same object)."""
        mock_is_feature_enabled.return_value = False
        mock_is_client_capable.return_value = False

        custom_config = CompactionConfig(
            max_recent_messages=15,
            trim_threshold=0.6,
        )

        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=CategoryEnum.WORKFLOW_CHAT,
            system_template_override=None,
            compaction=custom_config,
        )

        assert isinstance(agent, ChatAgent)
        optimizers = agent._optimizer_pipeline.optimizers
        assert isinstance(optimizers[0], CompactionOptimizer)
        assert optimizers[0]._config is custom_config
        # The manual compactor is the same object as the pipeline optimizer — no duplicate.
        assert agent._manual_compactor is optimizers[0]

    @patch("duo_workflow_service.agents.chat_agent_factory.is_feature_enabled")
    @patch("duo_workflow_service.agents.chat_agent_factory.is_client_capable")
    def test_create_agent_without_compaction(
        self,
        mock_is_client_capable,
        mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
    ):
        """When compaction is None, the pipeline falls back to legacy trim and manual compactor is None."""
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
            compaction=None,
        )

        assert isinstance(agent, ChatAgent)
        optimizers = agent._optimizer_pipeline.optimizers
        assert len(optimizers) == 1
        assert isinstance(optimizers[0], LegacyTrimOptimizer)
        assert agent._manual_compactor is None


class TestExtractManualCompactor:
    """Unit tests for _extract_manual_compactor."""

    def test_returns_none_when_compaction_is_none(self):
        """When compaction config is None, always return None regardless of pipeline contents."""
        compactor = Mock(spec=CompactionOptimizer)
        pipeline = HistoryOptimizerPipeline([compactor])
        assert _extract_manual_compactor(pipeline, None) is None

    def test_returns_none_when_pipeline_is_empty(self):
        """When the pipeline has no optimizers, return None even if compaction is configured."""
        pipeline = HistoryOptimizerPipeline([])
        assert _extract_manual_compactor(pipeline, CompactionConfig()) is None

    def test_returns_compaction_optimizer_at_index_zero(self):
        """Returns the CompactionOptimizer when it is the first (and only) optimizer."""
        compactor = Mock(spec=CompactionOptimizer)
        pipeline = HistoryOptimizerPipeline([compactor])
        result = _extract_manual_compactor(pipeline, CompactionConfig())
        assert result is compactor

    def test_returns_compaction_optimizer_not_at_index_zero(self):
        """Returns the CompactionOptimizer even when it is not the first entry in the pipeline.

        This covers the case where pre-main optimizers (e.g. ToolResultPruner) are inserted ahead of the
        CompactionOptimizer as the pipeline grows.
        """
        pre_main = Mock(spec=LegacyTrimOptimizer)
        compactor = Mock(spec=CompactionOptimizer)
        pipeline = HistoryOptimizerPipeline([pre_main, compactor])
        result = _extract_manual_compactor(pipeline, CompactionConfig())
        assert result is compactor

    def test_returns_none_when_no_compaction_optimizer_in_pipeline(self):
        """Returns None when the pipeline contains no CompactionOptimizer instance."""
        trim_optimizer = Mock(spec=LegacyTrimOptimizer)
        pipeline = HistoryOptimizerPipeline([trim_optimizer])
        assert _extract_manual_compactor(pipeline, CompactionConfig()) is None
