import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.components.tools_registry import ToolMetadata
from duo_workflow_service.entities.state import (
    Plan,
    Task,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


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
        outbox=MagicMock(spec=asyncio.Queue),
        inbox=MagicMock(spec=asyncio.Queue),
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
