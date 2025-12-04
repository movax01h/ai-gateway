from unittest.mock import MagicMock, Mock

import pytest

from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.tools import DuoBaseTool


@pytest.fixture(name="graph_config")
def graph_config_fixture():
    return {"configurable": {"thread_id": "test-workflow"}}


@pytest.fixture(name="mock_tool")
def mock_tool_fixture() -> Mock:
    mock = MagicMock(DuoBaseTool)
    mock.args_schema = None
    mock.name = "test_tool"
    mock.description = "Test description"
    return mock


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock = MagicMock()
    mock.bindable = []
    return mock


@pytest.fixture(name="mock_tool_registry")
def mock_tool_registry_fixture(mock_tool):
    """Create a mock tool registry."""
    registry = MagicMock(ToolsRegistry)
    registry.get.return_value = mock_tool
    return registry
