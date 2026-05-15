from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import BaseCheckpointSaver

from duo_workflow_service.components.tools_registry import _AGENT_PRIVILEGES
from duo_workflow_service.entities import WorkflowStatusEnum


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


@pytest.fixture(name="mock_checkpointer")
def mock_checkpointer_fixture():
    return Mock(spec=BaseCheckpointSaver)
