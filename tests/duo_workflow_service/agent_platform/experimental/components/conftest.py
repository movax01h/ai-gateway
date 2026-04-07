from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import BaseTool

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    """Fixture for mock prompt registry."""
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_prompt = Mock()
    mock_prompt.model = Mock()
    mock_prompt.model.model_name = "claude-3-sonnet"
    mock_registry.get_on_behalf.return_value = mock_prompt
    return mock_registry


@pytest.fixture(name="flow_id")
def flow_id_fixture():
    """Fixture for flow ID."""
    return "test_flow_id"


@pytest.fixture(name="flow_type")
def flow_type_fixture() -> GLReportingEventContext:
    """Fixture for flow type."""
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="mock_internal_event_client")
def mock_internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    """Fixture for mock tool."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.ainvoke = AsyncMock(return_value="Tool execution result")
    return mock_tool


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    """Fixture for mock toolset."""
    mock_toolset = Mock(spec=Toolset)
    mock_toolset.__contains__ = Mock(return_value=True)
    mock_toolset.__getitem__ = Mock(return_value=mock_tool)
    mock_toolset.bindable = [mock_tool]
    return mock_toolset
