# pylint: disable=file-naming-for-tests, unused-argument, redefined-outer-name, direct-environment-variable-reference
import asyncio
import os
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END

from ai_gateway.models import KindAnthropicModel
from contract import contract_pb2
from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolsRegistry,
)
from duo_workflow_service.entities import Plan, WorkflowStatusEnum
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from duo_workflow_service.workflows.software_development.workflow import (
    CONTEXT_BUILDER_TOOLS,
    EXECUTOR_TOOLS,
    PLANNER_TOOLS,
    Workflow,
)
from lib.internal_events.event_enum import CategoryEnum


class MockComponent:
    _mock_node_run: MagicMock
    _approved_agent_name: str

    def __init__(self, mock_node_run, approved_agent_name: str):
        self._approved_agent_name = approved_agent_name
        self._mock_node_run = mock_node_run

    def attach(
        self,
        graph,
        exit_node: str,
        back_node: str,
        next_node: str,
    ):
        node_name = f"{self._approved_agent_name}_mock_component_entry_node"

        graph.add_node(node_name, self._mock_node_run)
        graph.add_edge(node_name, exit_node)

        return node_name


@pytest.fixture(autouse=True)
def prepare_container(mock_duo_workflow_service_container):
    pass


@pytest.fixture(name="workflow")
def workflow_fixture(
    mock_duo_workflow_service_container: Mock,
    gl_http_client: GitlabHttpClient,
    user: CloudConnectorUser,
):
    """Create a software development workflow instance."""
    workflow = Workflow(
        "test",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user=user,
    )
    workflow._project = {"id": 1, "name": "test", "http_url_to_repo": "http://test"}  # type: ignore
    workflow._http_client = gl_http_client
    return workflow


@pytest.fixture(name="workflow_config")
def workflow_config_fixture():
    return {
        "first_checkpoint": None,
        "workflow_status": "created",
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "mcp_enabled": True,
        "allow_agent_to_request_user": True,
    }


@pytest.fixture(name="mock_log_exception")
def mock_log_exception_fixture():
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.log_exception"
    ) as mock:
        yield mock


@pytest.fixture(name="agent_responses")
def agent_responses_fixture() -> list[dict[str, Any]]:
    status = WorkflowStatusEnum.PLANNING
    agent_name = "context_builder"

    return [
        {
            "plan": Plan(steps=[]),
            "status": status,
            "conversation_history": {
                agent_name: [
                    SystemMessage(content="system message"),
                    HumanMessage(content="human message"),
                    AIMessage(
                        content="No tool calls in last AI message, route to the supervisor",
                    ),
                ],
            },
        },
        {
            "plan": Plan(steps=[]),
            "status": status,
            "conversation_history": {
                agent_name: [
                    SystemMessage(content="system message"),
                    HumanMessage(content="human message"),
                    AIMessage(
                        content="Tool calls are present, route to planner tools execution",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "test_tool",
                                "args": {"test": "test"},
                            }
                        ],
                    ),
                ],
            },
        },
        {
            "plan": Plan(steps=[]),
            "status": status,
            "conversation_history": {
                agent_name: [
                    SystemMessage(content="system message"),
                    HumanMessage(content="human message"),
                    AIMessage(
                        content="HandoverTool call is present, route to the next agent",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "handover_tool",
                                "args": {"summary": "done"},
                            }
                        ],
                    ),
                ],
            },
        },
    ]


@pytest.fixture(name="mock_handover_agent")
def mock_handover_agent_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.HandoverAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        }
        yield mock


@pytest.fixture(name="mock_plan_supervisor_agent")
def mock_plan_supervisor_agent_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }
        yield mock


@pytest.fixture(name="mock_tools_executor")
def mock_tools_executor_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.ToolsExecutor"
    ) as mock:
        yield mock


@pytest.fixture(name="mock_executor_component")
def mock_executor_component_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.ExecutorComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = END
        yield mock


@pytest.fixture(name="mock_planner_component")
def mock_planner_component_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.PlannerComponent",
        autospec=True,
    ) as mock:
        mock.return_value.attach.return_value = "set_status_to_execution"
        yield mock


@pytest.fixture(name="mock_tools_approval_component")
def mock_tools_approval_component_fixture():
    with patch(
        "duo_workflow_service.workflows.software_development.workflow.ToolsApprovalComponent",
        autospec=True,
    ) as mock:
        yield mock


@pytest.fixture
def mock_action():
    return contract_pb2.Action()


@pytest.mark.asyncio
async def test_workflow_initialization(workflow):
    assert isinstance(workflow._outbox, Outbox)


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aget_tuple")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aput")
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.GitLabStatusUpdater",
    autospec=True,
)
async def test_workflow_run(
    mock_status_updater,
    mock_gitlab_workflow_aput,
    mock_gitlab_workflow_aget_tuple,
    mock_checkpoint_notifier,
    mock_tools_approval_component,
    mock_planner_component,
    mock_executor_component,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    mock_tools_registry,
    checkpoint_tuple,
    workflow,
):
    mock_gitlab_workflow_aput.return_value = None
    mock_gitlab_workflow_aget_tuple.return_value = None

    mock_user_interface_instance = mock_checkpoint_notifier.return_value

    mock_tools_executor.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
        },
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
        },
    ]

    mock_tools_approval_component.return_value.attach.side_effect = [
        "build_context_tools",
        "execution_tools",
    ]

    await workflow.run("test_goal")

    assert mock_planner_component.return_value.attach.call_count == 1
    assert mock_executor_component.return_value.attach.call_count == 1
    assert (
        mock_planner_component.return_value.attach.call_args.kwargs.get(
            "approval_component"
        )
        is not None
    )
    assert (
        mock_executor_component.return_value.attach.call_args.kwargs.get(
            "approval_component"
        )
        is not None
    )

    assert mock_tools_approval_component.return_value.attach.call_count == 1

    assert mock_agent.run.call_count >= 3

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 2
    assert mock_handover_agent.return_value.run.call_count == 2

    assert mock_plan_supervisor_agent.call_count == 1
    assert mock_plan_supervisor_agent.return_value.run.call_count == 1

    assert mock_gitlab_workflow_aput.call_count >= 1
    assert mock_gitlab_workflow_aget_tuple.call_count >= 1

    mock_user_interface_instance.send_event.assert_called_with(
        type=ANY, state=ANY, stream=False
    )
    assert mock_user_interface_instance.send_event.call_count >= 2

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.parametrize("offline_mode", [True])
async def test_workflow_run_with_memory_saver(
    mock_checkpoint_notifier,
    mock_executor_component,
    mock_planner_component,
    mock_gitlab_workflow,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_git_lab_workflow_instance,
    mock_tools_registry_cls,
    workflow,
):
    mock_git_lab_workflow_instance.__aenter__.return_value = MemorySaver()

    mock_tools_executor.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
        },
    ]

    await workflow.run("test_goal")

    assert mock_agent.run.call_count == 3

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 2
    assert mock_handover_agent.return_value.run.call_count >= 1

    assert mock_plan_supervisor_agent.call_count == 1
    assert mock_plan_supervisor_agent.return_value.run.call_count == 1

    assert mock_git_lab_workflow_instance.aput.call_count == 0
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 0

    assert mock_planner_component.return_value.attach.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
async def test_workflow_run_when_exception(
    mock_log_exception,
    mock_planner_component,
    mock_executor_component,
    mock_fetch_workflow_and_container_data,
    mock_gitlab_workflow,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    workflow,
):
    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    with patch(
        "duo_workflow_service.workflows.software_development.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.aget_state = AsyncMock(return_value=None)
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("test_goal")

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aget_tuple")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow.aput")
@patch(
    "duo_workflow_service.checkpointer.gitlab_workflow.GitLabStatusUpdater",
    autospec=True,
)
async def test_workflow_run_with_error_state(
    mock_status_updater,
    mock_gitlab_workflow_aput,
    mock_gitlab_workflow_aget_tuple,
    mock_checkpoint_notifier,
    mock_planner_component,
    mock_tools_approval_component,
    mock_executor_component,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_agent,
    workflow,
):
    mock_gitlab_workflow_aput.return_value = None
    mock_gitlab_workflow_aget_tuple.return_value = None

    mock_tools_executor.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.ERROR,
            "conversation_history": {},
        }
    ]

    mock_tools_approval_component.return_value.attach.return_value = (
        "build_context_tools"
    )

    await workflow.run("test_goal")

    assert mock_agent.run.call_count == 2

    assert mock_tools_executor.call_count == 1
    assert mock_tools_executor.return_value.run.call_count == 1

    assert mock_planner_component.return_value.attach.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
async def test_workflow_run_with_tools_registry(
    mock_log_exception,
    mock_executor_component,
    mock_planner_component,
    mock_gitlab_workflow,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_git_lab_workflow_instance,
    mock_tools_registry_cls,
    mock_tools_registry,
    checkpoint_tuple,
    workflow,
):
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])

    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    with patch(
        "duo_workflow_service.workflows.software_development.workflow.StateGraph"
    ) as graph_cls:
        compiled_graph = MagicMock()
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph_cls.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("test_goal")

    mock_tools_registry.toolset.assert_has_calls(
        [
            call(EXECUTOR_TOOLS),
            call(CONTEXT_BUILDER_TOOLS),
            call(PLANNER_TOOLS),
        ],
        any_order=True,
    )


@pytest.fixture(name="tools_registry")
def tools_registry_fixture(tool_metadata):
    """Create a tools registry with all privileges enabled."""
    return ToolsRegistry(
        enabled_tools=list(_AGENT_PRIVILEGES.keys()),
        preapproved_tools=list(_AGENT_PRIVILEGES.keys()),
        tool_metadata=tool_metadata,
    )


def assert_tools_in_tools_registry(tools_registry, tools):
    missing_tools = []
    for tool_name in tools:
        if tools_registry.get(tool_name) is None:
            missing_tools.append(tool_name)

    assert (
        not missing_tools
    ), f"The following tools are missing from the tools registry: {missing_tools}"

    assert tools, "No tools were captured"


# Above, test_workflow_run_with_tools_registry checks that the tools listed in the test match the tool registry
# calls made when the workflow is run.
# The next test check that the tools defined in the agent setup methods in the workflow are actually in
# the registry and match the list in the test.
def test_context_builder_tools(tools_registry, workflow):
    """Test that all tools used by the context builder agent are available in the tools registry."""
    agent_components = workflow._setup_context_builder(tools_registry)
    assert agent_components["toolset"] == tools_registry.toolset(CONTEXT_BUILDER_TOOLS)
    assert_tools_in_tools_registry(tools_registry, agent_components["toolset"])


@pytest.mark.asyncio
async def test_workflow_run_with_setup_error(
    mock_executor_component,
    mock_planner_component,
    mock_fetch_workflow_and_container_data,
    mock_gitlab_workflow,
    mock_git_lab_workflow_instance,
    mock_tools_registry_cls,
    checkpoint_tuple,
    workflow,
):
    mock_tools_registry_cls.configure = AsyncMock(
        side_effect=Exception("Failed to configure tools")
    )

    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])

    await workflow.run("test_goal")

    assert workflow.is_done


@pytest.mark.asyncio
async def test_workflow_run_with_missing_web_url(
    mock_fetch_workflow_and_container_data,
    mock_gitlab_workflow,
    mock_git_lab_workflow_instance,
    checkpoint_tuple,
    workflow,
):
    mock_fetch_workflow_and_container_data.return_value = (
        {
            "id": 1,
            "name": "test-project",
            "description": "This is a test project",
            "http_url_to_repo": "https://example.com/project",
            # web_url is missing
        },
        None,
        {"project_id": 1},
    )

    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])

    await workflow.run("test_goal")
    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.gitlab.gitlab_api.GitLabUrlParser", autospec=True)
async def test_workflow_run_with_invalid_web_url(
    mock_gitlab_url_parser,
    mock_fetch_workflow_and_container_data,
    mock_gitlab_workflow,
    mock_git_lab_workflow_instance,
    checkpoint_tuple,
    workflow,
):
    # Test case for invalid web_url (cannot extract gitlab_host)
    mock_fetch_workflow_and_container_data.return_value = (
        {
            "id": 1,
            "name": "test-project",
            "description": "This is a test project",
            "http_url_to_repo": "https://example.com/project",
            "web_url": "invalid-url",  # Invalid URL format
        },
        None,
        {"project_id": 1},
    )

    mock_gitlab_url_parser.extract_host_from_url.return_value = None

    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])

    await workflow.run("test_goal")
    assert workflow.is_done


@pytest.mark.asyncio
async def test_workflow_run_with_retry(
    mock_log_exception,
    mock_executor_component,
    mock_planner_component,
    mock_fetch_workflow_and_container_data,
    mock_gitlab_workflow,
    mock_git_lab_workflow_instance,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    workflow,
):
    # Setup AsyncIterator for workflow steps
    class AsyncIterator:
        def __init__(self):
            self.call_count = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.call_count += 1
            if self.call_count > 3:
                raise StopAsyncIteration
            if self.call_count == 1:
                raise asyncio.CancelledError()
            return {"build_context": {}}

    async_iterator = AsyncIterator()

    with patch(
        "duo_workflow_service.workflows.software_development.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.astream.return_value = async_iterator
        compiled_graph.aget_state = AsyncMock(return_value=None)
        graph.return_value.compile.return_value = compiled_graph

        await workflow.run("test_goal")
        assert workflow.is_done

    mock_checkpoint = {
        "id": "checkpoint1",
        "channel_values": {
            "agent": "build_context",
            "conversation_history": {"build_context": [{}]},
        },
    }
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(
        return_value=CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "123",
                    "checkpoint_id": "checkpoint1",
                }
            },
            checkpoint=mock_checkpoint,
            metadata={},
            parent_config={},
        )
    )

    # re-run the workflow
    async_iterator = AsyncIterator()

    with patch(
        "duo_workflow_service.workflows.software_development.workflow.StateGraph"
    ) as graph:
        compiled_graph = MagicMock()
        compiled_graph.astream.return_value = async_iterator
        compiled_graph.aget_state = AsyncMock(return_value=None)
        compiled_graph.aupdate_state = AsyncMock(return_value=None)
        graph.return_value.compile.return_value = compiled_graph

        await workflow.run("test_goal")
        assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_responses",
    [
        [
            {
                "plan": Plan(steps=[]),
                "status": WorkflowStatusEnum.PLANNING,
                "conversation_history": {
                    "context_builder": [
                        SystemMessage(content="system message"),
                        HumanMessage(content="human message"),
                        AIMessage(
                            content="Tool calls are present, route to build context tools execution",
                            tool_calls=[
                                {
                                    "id": "1",
                                    "name": "run_command",
                                    "args": {"test": "test"},
                                }
                            ],
                        ),
                    ],
                },
            },
            {
                "plan": Plan(steps=[]),
                "status": WorkflowStatusEnum.EXECUTION,
                "conversation_history": {
                    "context_builder": [
                        AIMessage(
                            content="Tool calls are present, route to build executor tools execution",
                            tool_calls=[
                                {
                                    "id": "1",
                                    "name": "run_command",
                                    "args": {"test": "test"},
                                }
                            ],
                        ),
                    ],
                },
            },
        ]
    ],
)
@pytest.mark.parametrize("tool_approval_required", [[True, False, False]])
async def test_workflow_run_with_tool_approvals(
    mock_checkpoint_notifier,
    mock_executor_component,
    mock_tools_approval_component,
    mock_gitlab_workflow,
    mock_git_lab_workflow_instance,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_planner_component,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    workflow,
):
    mock_tools_approval_execution = MagicMock()
    mock_tools_approval_execution.return_value = {"status": WorkflowStatusEnum.PLANNING}
    mock_tools_approval_component.return_value = MockComponent(
        mock_node_run=mock_tools_approval_execution,
        approved_agent_name="context_builder",
    )

    await workflow.run("test_goal")

    assert mock_tools_approval_execution.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_config",
    [
        {
            "project_id": 1,
            "allow_agent_to_request_user": False,
            "agent_privileges_names": [],
            "pre_approved_agent_privileges_names": [],
            "mcp_enabled": False,
            "first_checkpoint": None,
            "workflow_status": "",
            "gitlab_host": "gitlab.com",
        }
    ],
)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanApprovalComponent",
    autospec=True,
)
async def test_workflow_run_without_plan_approval_component(
    mock_plan_approval_component,
    mock_executor_component,
    mock_tools_approval_component,
    mock_gitlab_workflow,
    mock_fetch_workflow_and_container_data,
    mock_tools_executor,
    mock_planner_component,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
    workflow,
):
    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    with patch(
        "duo_workflow_service.workflows.software_development.workflow.StateGraph"
    ) as graph_cls:
        compiled_graph = MagicMock()
        compiled_graph.astream.return_value = AsyncIterator()
        instance = graph_cls.return_value
        instance.compile.return_value = compiled_graph
        await workflow.run("test_goal")

    assert mock_planner_component.return_value.attach.call_count == 1
    assert mock_executor_component.return_value.attach.call_count == 1

    mock_planner_component.return_value.attach.assert_called_with(
        graph=ANY,
        next_node="set_status_to_execution",
        exit_node="plan_terminator",
        approval_component=None,
    )
    mock_executor_component.return_value.attach.assert_called_with(
        graph=ANY, next_node=END, exit_node="plan_terminator", approval_component=ANY
    )
    mock_plan_approval_component.assert_not_called()

    assert workflow.is_done


@pytest.mark.asyncio
async def test_workflow_cleanup(workflow, mock_action):
    assert workflow._outbox._queue.empty()

    workflow._outbox.put_action(mock_action)

    assert workflow._outbox._queue.qsize() == 1
    assert not workflow.is_done

    await workflow.cleanup("123")

    assert workflow.is_done
    assert workflow._outbox._queue.qsize() == 0


@pytest.mark.parametrize(
    "env_vars,expected_config_type,expected_model",
    [
        # Vertex (falls back to parent's hardcoded model)
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-project",
                "DUO_WORKFLOW__VERTEX_LOCATION": "us-central1",
            },
            VertexConfig,
            KindAnthropicModel.CLAUDE_SONNET_4_VERTEX.value,
        ),
        # Anthropic API (falls back to parent's hardcoded model)
        (
            {"ANTHROPIC_API_KEY": "test-key"},
            AnthropicConfig,
            KindAnthropicModel.CLAUDE_SONNET_4.value,
        ),
    ],
)
def test_software_development_workflow_model_config(
    env_vars,
    expected_config_type,
    expected_model,
    workflow,
):
    """Test that software development workflow uses correct model based on feature flags."""
    with patch.dict(os.environ, env_vars, clear=True):
        config = workflow._get_model_config()

        assert isinstance(config, expected_config_type)
        assert config.model_name == expected_model
