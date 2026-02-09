from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.components.tools_registry import ToolMetadata, ToolsRegistry
from duo_workflow_service.entities.state import (
    Plan,
    Task,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from lib.context import gitlab_version
from lib.events import GLReportingEventContext


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {"mock_model_responses": True}


@pytest.fixture(name="plan_steps")
def plan_steps_fixture() -> list[Task]:
    return []


@pytest.fixture(name="plan")
def plan_fixture(plan_steps: list[Task]) -> Plan:
    return Plan(steps=plan_steps)


@pytest.fixture(name="mock_now")
def mock_now_fixture() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(name="gl_http_client", scope="function")
def gl_http_client_fixture():
    return AsyncMock(spec=GitlabHttpClient)


@pytest.fixture(name="project_mock", scope="function")
def project_mock_fixture():
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=None,
    )


@pytest.fixture(name="tool_metadata", scope="function")
def tool_metadata_fixture(gl_http_client, project_mock):
    return ToolMetadata(
        outbox=MagicMock(spec=Outbox),
        gitlab_client=gl_http_client,
        gitlab_host="gitlab.example.com",
        project=project_mock,
    )


@pytest.fixture(name="graph_input", scope="function")
def graph_input_fixture() -> WorkflowState:
    return WorkflowState(
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        last_human_input=None,
        handover=[],
        ui_chat_log=[],
        plan=Plan(steps=[]),
        project=None,
        goal=None,
        additional_context=None,
    )


@pytest.fixture(name="mock_gitlab_workflow")
def mock_gitlab_workflow_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True
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


@pytest.fixture(name="gl_version")
def gl_version_fixture() -> str:
    return "17.5.2"


@pytest.fixture(name="mock_gitlab_version")
def mock_gitlab_version_fixture(gl_version: str):
    # Set GitLab version in context
    gitlab_version.set(gl_version)
    yield
    gitlab_version.set(None)


@pytest.fixture(name="workflow_id")
def workflow_id_fixture():
    return "1234"


@pytest.fixture(name="workflow_type")
def workflow_type_fixture() -> GLReportingEventContext:
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="agent_responses")
def agent_responses_fixture() -> list[dict[str, Any]]:
    return []


@pytest.fixture(name="mock_agent")
def mock_agent_fixture(agent_responses: list[dict[str, Any]]):
    with patch("duo_workflow_service.agents.agent.Agent") as mock:
        mock.return_value.run.side_effect = agent_responses
        yield mock.return_value


@pytest.fixture(name="tool_approval_required")
def tool_approval_required_fixture():
    return False


@pytest.fixture(name="mock_tools_registry")
def mock_tools_registry_fixture(tool_approval_required):
    mock = MagicMock(spec=ToolsRegistry)

    mock.approval_required.return_value = tool_approval_required

    return mock


@pytest.fixture(name="system_template_override")
def system_template_override_fixture() -> str:
    return "Test system template"


@pytest.fixture(name="user")
def user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_workflow_execute_workflow"],
            issuer="gitlab-duo-workflow-service",
        ),
    )
