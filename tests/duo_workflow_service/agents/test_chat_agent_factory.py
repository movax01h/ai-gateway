from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
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
from duo_workflow_service.tools.web_search import WebSearchState
from lib.events import GLReportingEventContext


@pytest.fixture(name="mock_tools_registry")
def mock_tools_registry_fixture():
    return Mock(spec=ToolsRegistry)


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock_toolset = Mock(spec=Toolset)
    mock_toolset.bindable = []
    return mock_toolset


class TestCreateAgent:
    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
        tracker = Mock(spec=ToolEventTracker)

        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
            system_template_override="test_system_template",
            tracker=tracker,
        )

        assert isinstance(agent, ChatAgent)
        assert agent.name == prompt.name
        assert agent.system_template_override == "test_system_template"
        assert agent._tracker is tracker

        mock_local_prompt_registry.get_on_behalf.assert_called_once_with(
            user=user,
            prompt_id="chat/agent",
            prompt_version="^1.0.0",
            internal_event_category="test_category",
            tools=[],
            internal_event_extra={
                "agent_name": "chat",
                "workflow_id": "workflow_123",
                "workflow_type": "chat",
            },
        )

    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
            workflow_type=GLReportingEventContext.from_workflow_definition(
                "ai_catalog_agent"
            ),
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
                "agent_name": "348/0",
                "workflow_id": "workflow_123",
                "workflow_type": "ai_catalog_agent",
            },
        )

    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
            system_template_override=None,
            agent_name_override=None,
        )

        assert isinstance(agent, ChatAgent)

        call_kwargs = mock_local_prompt_registry.get_on_behalf.call_args.kwargs
        assert call_kwargs["internal_event_extra"]["agent_name"] == "chat"
        assert "bind_tools_params" not in call_kwargs

    @pytest.mark.parametrize(
        ("session_allows", "toggle", "binds_web_search", "expected_state"),
        [
            (True, True, True, WebSearchState(supported=True, active=True)),
            # Toggled on, but the binding found no route (no native search and no gateway, or an
            # admin deny): the toggle can't help, so the prompt mustn't mention web search.
            (True, True, False, WebSearchState(supported=False, active=False)),
            # Toggled off: the prompt says it's off and that the toggle turns it back on.
            (True, False, False, WebSearchState(supported=True, active=False)),
            (False, True, False, WebSearchState(supported=False, active=False)),
        ],
        ids=["bound", "toggled_on_no_route", "toggled_off", "not_allowed"],
    )
    @patch(
        "duo_workflow_service.tools.web_search.is_feature_enabled", return_value=True
    )
    @patch("duo_workflow_service.tools.web_search.is_client_capable", return_value=True)
    def test_prompt_says_search_is_on_only_when_bound(
        self,
        _mock_is_client_capable,
        _mock_is_feature_enabled,
        user,
        mock_tools_registry,
        mock_toolset,
        mock_local_prompt_registry,
        prompt,
        session_allows,
        toggle,
        binds_web_search,
        expected_state,
    ):
        """Which route the model gets is the binding's call (`bind_web_search`); the factory adds none."""
        prompt.binds_web_search = binds_web_search

        agent = create_agent(
            user=user,
            tools_registry=mock_tools_registry,
            internal_event_category="test_category",
            tools=mock_toolset,
            prompt_registry=mock_local_prompt_registry,
            workflow_id="workflow_123",
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
            system_template_override=None,
            web_search_enabled=toggle,
            web_search_allowed_for_group=session_allows,
        )

        assert (
            "bind_tools_params"
            not in mock_local_prompt_registry.get_on_behalf.call_args.kwargs
        )
        assert agent.web_search == expected_state

    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
            system_template_override=None,
            compaction=cfg,
        )

        assert isinstance(agent, ChatAgent)
        optimizers = agent._optimizer_pipeline.optimizers
        assert len(optimizers) == 2
        assert isinstance(optimizers[0], CompactionOptimizer)
        assert optimizers[0]._config is cfg
        assert isinstance(optimizers[1], LegacyTrimOptimizer)

        # The manual compactor is the same object as the pipeline optimizer — no duplicate.
        assert agent._manual_compactor is optimizers[0]

    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
            system_template_override=None,
            compaction=custom_config,
        )

        assert isinstance(agent, ChatAgent)
        optimizers = agent._optimizer_pipeline.optimizers
        assert isinstance(optimizers[0], CompactionOptimizer)
        assert optimizers[0]._config is custom_config
        # The manual compactor is the same object as the pipeline optimizer — no duplicate.
        assert agent._manual_compactor is optimizers[0]

    @patch("duo_workflow_service.tools.web_search.is_feature_enabled")
    @patch("duo_workflow_service.tools.web_search.is_client_capable")
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
            workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
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
