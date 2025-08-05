from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointTuple

from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import Plan, WorkflowStatusEnum


@pytest.fixture(name="tool_approval_required")
def tool_approval_required_fixture():
    return False


@pytest.fixture(name="mock_tools_registry")
def mock_tools_registry_fixture(tool_approval_required):
    mock = MagicMock(spec=ToolsRegistry)

    mock.approval_required.return_value = tool_approval_required

    return mock


@pytest.fixture(name="mock_tools_registry_cls")
def mock_tools_registry_cls_fixture(mock_tools_registry):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True
    ) as mock:
        mock.configure = AsyncMock(return_value=mock_tools_registry)
        yield mock


@pytest.fixture(name="mock_gitlab_workflow")
def mock_gitlab_workflow_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True
    ) as mock:
        yield mock


@pytest.fixture(name="mock_checkpoint_notifier")
def mock_checkpoint_notifier_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True
    ) as mock:
        yield mock


@pytest.fixture(name="offline_mode")
def offline_mode_fixture():
    return False


@pytest.fixture(name="mock_git_lab_workflow_instance")
def mock_git_lab_workflow_instance_fixture(mock_gitlab_workflow, offline_mode):
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


@pytest.fixture(name="workflow_config")
def workflow_config_fixture() -> dict[str, Any]:
    return {"project_id": 1, "agent_privileges_names": []}


@pytest.fixture(name="mock_fetch_workflow_and_container_data")
def mock_fetch_workflow_and_container_data_fixture(workflow_config: dict[str, Any]):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
    ) as mock:
        mock.return_value = (
            {
                "id": 1,
                "name": "test-project",
                "description": "This is a test project",
                "http_url_to_repo": "https://example.com/project",
                "web_url": "https://example.com/project",
            },
            None,
            workflow_config,
        )
        yield mock


@pytest.fixture(name="checkpoint_tuple")
def checkpoint_tuple_fixture():
    return CheckpointTuple(
        config={"configurable": {"thread_id": "123", "checkpoint_id": str(uuid4())}},
        checkpoint={
            "channel_values": {"status": WorkflowStatusEnum.NOT_STARTED},
            "id": str(uuid4()),
            "channel_versions": {},
            "pending_sends": [],
            "versions_seen": {},
            "ts": "",
            "v": 0,
        },
        metadata={"step": 0},
        parent_config={"configurable": {"thread_id": "123", "checkpoint_id": None}},
    )
