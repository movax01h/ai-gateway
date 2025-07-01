from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.components.tools_registry import ToolsRegistry


@pytest.fixture
def tool_approval_required():
    return False


@pytest.fixture
def mock_tools_registry(tool_approval_required):
    mock = MagicMock(spec=ToolsRegistry)

    mock.approval_required.return_value = tool_approval_required

    return mock


@pytest.fixture
def mock_tools_registry_cls(mock_tools_registry):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True
    ) as mock:
        mock.configure = AsyncMock(return_value=mock_tools_registry)
        yield mock


@pytest.fixture
def mock_gitlab_workflow():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True
    ) as mock:
        yield mock


@pytest.fixture
def mock_checkpoint_notifier():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True
    ) as mock:
        yield mock


@pytest.fixture
def offline_mode():
    return False


@pytest.fixture
def mock_git_lab_workflow_instance(mock_gitlab_workflow, offline_mode):
    mock = mock_gitlab_workflow.return_value
    mock.__aenter__.return_value = mock
    mock.__aexit__.return_value = None
    mock._offline_mode = offline_mode
    mock.aget_tuple = AsyncMock(return_value=None)
    mock.alist = AsyncMock(return_value=[])
    mock.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock.get_next_version = MagicMock(return_value=1)

    return mock


@pytest.fixture
def workflow_config() -> dict[str, Any]:
    return {"project_id": 1}


@pytest.fixture
def mock_fetch_project_data_with_workflow_id(workflow_config: dict[str, Any]):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
    ) as mock:
        mock.return_value = (
            {
                "id": 1,
                "name": "test-project",
                "description": "This is a test project",
                "http_url_to_repo": "https://example.com/project",
                "web_url": "https://example.com/project",
            },
            workflow_config,
        )
        yield mock
