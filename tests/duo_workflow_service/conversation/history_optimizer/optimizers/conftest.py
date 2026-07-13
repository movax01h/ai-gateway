from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)


@pytest.fixture(name="compaction_config")
def compaction_config_fixture():
    return CompactionConfig()


@pytest.fixture(name="mock_prompt")
def mock_prompt_fixture():
    """Mock Prompt with ainvoke and prompt_tpl stubbed."""
    mock = AsyncMock()
    mock.prompt_tpl = MagicMock()
    mock.prompt_tpl.format_messages.return_value = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user prompt"),
    ]
    mock.operation_type = "compaction_auto"
    return mock


@pytest.fixture(name="mock_prompt_manual")
def mock_prompt_manual_fixture():
    """Mock Prompt for the manual compaction prompt."""
    mock = AsyncMock()
    mock.prompt_tpl = MagicMock()
    mock.prompt_tpl.format_messages.return_value = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user prompt"),
    ]
    mock.operation_type = "compaction_manual"
    return mock


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture(mock_prompt):
    """Mock BasePromptRegistry returning mock_prompt from get_on_behalf."""
    mock_registry = MagicMock()
    mock_registry.get_on_behalf.return_value = mock_prompt
    return mock_registry


@pytest.fixture(name="mock_internal_events_client")
def mock_internal_events_client_fixture():
    """Mock InternalEventsClient for testing Snowplow event firing."""
    return MagicMock()


@pytest.fixture(name="compactor")
def compactor_fixture(compaction_config, mock_prompt_registry, user):
    """Create a ConversationCompactor via the factory, using mock registry.

    Patches get_model_metadata so the compactor doesn't depend on the gRPC model metadata context variable during tests.
    The patch stays active for the lifetime of the test because the compactor now reads model metadata lazily inside
    compact().
    """
    with patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers.compaction.get_model_metadata",
        return_value=None,
    ):
        yield create_conversation_compactor(
            config=compaction_config,
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )


@pytest.fixture(name="compactor_with_events")
def compactor_with_events_fixture(
    compaction_config, mock_prompt_registry, user, mock_internal_events_client
):
    """Create a ConversationCompactor with an InternalEventsClient for event testing.

    Patches get_model_metadata so the compactor doesn't depend on the gRPC model metadata context variable during tests.
    The patch stays active for the lifetime of the test because the compactor now reads model metadata lazily inside
    compact().
    """
    with patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers.compaction.get_model_metadata",
        return_value=None,
    ):
        yield create_conversation_compactor(
            config=compaction_config,
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
            internal_events_client=mock_internal_events_client,
        )
