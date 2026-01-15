from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointTuple

from duo_workflow_service.components.tools_registry import _AGENT_PRIVILEGES
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project


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


@pytest.fixture(name="agent_privileges_names")
def agent_privileges_names_fixture() -> list[str]:
    return list(_AGENT_PRIVILEGES.keys())


@pytest.fixture(name="workflow_config")
def workflow_config_fixture(agent_privileges_names: list[str]) -> dict[str, Any]:
    return {
        "project_id": 1,
        "agent_privileges_names": agent_privileges_names,
        "pre_approved_agent_privileges_names": [],
        "allow_agent_to_request_user": False,
        "mcp_enabled": False,
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "",
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture(name="mock_fetch_workflow_and_container_data")
def mock_fetch_workflow_and_container_data_fixture(
    project: Project, workflow_config: dict[str, Any]
):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
    ) as mock:
        mock.return_value = (project, None, workflow_config)
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


@pytest.fixture(name="scopes")
def scopes_fixture():
    return ["duo_workflow_execute_workflow"]
