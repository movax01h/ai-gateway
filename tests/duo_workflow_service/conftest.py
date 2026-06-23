from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain.messages import AIMessage

from ai_gateway.container import ContainerApplication
from duo_workflow_service.components.tools_registry import ToolMetadata, ToolsRegistry
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    Task,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES
from duo_workflow_service.workflows.type_definitions import AdditionalContext
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
def tool_metadata_fixture(gl_http_client, project_mock, workflow_id):
    return ToolMetadata(
        workflow_id=workflow_id,
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

    mock.approval_required = AsyncMock(return_value=tool_approval_required)
    mock.is_preapproved.return_value = not bool(tool_approval_required)

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


@pytest.fixture(name="namespace")
def namespace_fixture() -> Namespace | None:
    return None


@pytest.fixture(name="agent_privileges_names")
def agent_privileges_names_fixture() -> list[str]:
    return []


@pytest.fixture(name="mcp_enabled")
def mcp_enabled_fixture() -> bool:
    return False


@pytest.fixture(name="allow_agent_to_request_user")
def allow_agent_to_request_user_fixture() -> bool:
    return False


@pytest.fixture(name="first_checkpoint")
def first_checkpoint_fixture() -> dict[str, Any] | None:
    return None


@pytest.fixture(name="workflow_config")
def workflow_config_fixture(
    workflow_id: str,
    agent_privileges_names: list[str],
    allow_agent_to_request_user: bool,
    mcp_enabled: bool,
    first_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "project_id": 1,
        "agent_privileges_names": agent_privileges_names,
        "pre_approved_agent_privileges_names": [],
        "allow_agent_to_request_user": allow_agent_to_request_user,
        "mcp_enabled": mcp_enabled,
        "first_checkpoint": first_checkpoint,
        "latest_checkpoint": None,
        "workflow_status": "",
        "gitlab_host": "gitlab.com",
        "archived": False,
        "stalled": False,
    }


@pytest.fixture(name="mock_fetch_workflow_and_container_data")
def mock_fetch_workflow_and_container_data_fixture(
    project: Project, namespace: Namespace | None, workflow_config: dict[str, Any]
):
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
    ) as mock:
        mock.return_value = (project, namespace, workflow_config)
        yield mock


@pytest.fixture(name="mock_gitlab_workflow_aget_tuple")
def mock_gitlab_workflow_aget_tuple_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aget_tuple",
        return_value=None,
    ) as mock:
        yield mock


@pytest.fixture(name="mock_gitlab_workflow_aput")
def mock_gitlab_workflow_aput_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aput",
        return_value=None,
    ) as mock:
        yield mock


@pytest.fixture(name="event")
def event_fixture() -> WorkflowEvent | None:
    return None


@pytest.fixture(name="mock_get_event")
def mock_get_event_fixture(event: WorkflowEvent | None):
    with patch(
        "duo_workflow_service.agents.agent.get_event", return_value=event
    ) as mock:
        yield mock


@pytest.fixture(name="mock_ai_message")
def mock_ai_message_fixture():
    """Fixture for mock AI message."""
    mock_message = Mock(spec=AIMessage)
    mock_message.content = "Test response from agent"
    mock_message.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    mock_message.response_metadata = {
        "finish_reason": "stop"
    }  # OpenAI format used by LiteLLM
    mock_message.tool_calls = []
    mock_message.invalid_tool_calls = []
    mock_message.additional_kwargs = {}
    return mock_message


@pytest.fixture(name="mock_duo_workflow_service_container", scope="module")
def mock_duo_workflow_service_container_fixture() -> Generator[
    ContainerApplication, None, None
]:
    """Module-scoped container fixture that wires the DI container once per test module.

    wire(packages=CONTAINER_APPLICATION_PACKAGES) costs ~220ms per call because it introspects the entire
    duo_workflow_service package. Scoping this to module means it runs once per test file instead of once per test,
    saving ~220ms × N tests.

    Tests that need to change provider behavior should use override/reset_override on specific providers (e.g.
    usage_quota.service), which is cheap and safe on a shared wired container.
    """
    from ai_gateway.config import Config  # pylint: disable=import-outside-toplevel

    with (
        patch("ai_gateway.models.base.PredictionServiceAsyncClient"),
        patch("ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"),
        patch(
            "ai_gateway.models.v2.container.connect_google_gen_vertex_ai",
            return_value=None,
        ),
    ):
        config = Config(
            _env_file=None, _env_prefix="AIGW_TEST", mock_model_responses=True
        )
        container = ContainerApplication()
        container.config.from_dict(config.model_dump())
        container.wire(packages=CONTAINER_APPLICATION_PACKAGES)
        yield container


@pytest.fixture(name="ui_chat_log")
def ui_chat_log_fixture() -> list[UiChatLog]:
    return [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "This is a test message",
            "timestamp": "2025-01-08T12:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "message_sub_type": None,
            "additional_context": None,
            "message_id": None,
        }
    ]


@pytest.fixture(name="last_human_input")
def last_human_input_fixture() -> WorkflowEvent | None:
    return None


@pytest.fixture(name="workflow_state", scope="function")
def workflow_state_fixture(
    project: Project,
    goal: str | None,
    last_human_input: WorkflowEvent | None,
    ui_chat_log: list[UiChatLog],
    additional_context: list[AdditionalContext] | None,
):
    return WorkflowState(
        plan=Plan(steps=[]),
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=last_human_input,
        ui_chat_log=ui_chat_log,
        project=project,
        goal=goal,
        additional_context=additional_context,
    )
