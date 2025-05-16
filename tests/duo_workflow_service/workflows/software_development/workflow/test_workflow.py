import asyncio
import os
from collections import namedtuple
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver

from contract import contract_pb2
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolsRegistry,
)
from duo_workflow_service.entities import Plan, WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.tools.toolset import Toolset
from duo_workflow_service.workflows.software_development import Workflow
from duo_workflow_service.workflows.software_development.workflow import (
    CONTEXT_BUILDER_TOOLS,
    EXECUTOR_TOOLS,
    PLANNER_TOOLS,
    Workflow,
)


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


@pytest.fixture
def checkpoint_tuple():
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


@pytest.mark.asyncio
async def test_workflow_initialization():
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    assert isinstance(workflow._outbox, asyncio.Queue)
    assert isinstance(workflow._inbox, asyncio.Queue)


def _agent_responses(status: WorkflowStatusEnum, agent_name: str):
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
                                "name": "update_plan",
                                "args": {"summary": "done"},
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


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run(
    mock_checkpoint_notifier,
    mock_goal_disambiguator_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
):
    mock_user_interface_instance = mock_checkpoint_notifier.return_value
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_tools_registry.approval_required.return_value = False
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

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
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        },
    ]

    mock_handover_agent.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
    ]

    mock_agent.return_value.run.side_effect = [
        *_agent_responses(
            WorkflowStatusEnum.PLANNING, "context_builder"
        ),  # context builder responses
        *_agent_responses(WorkflowStatusEnum.PLANNING, "planner"),  # planner responses
        *_agent_responses(
            WorkflowStatusEnum.EXECUTION, "executor"
        ),  # executor responses
    ]

    mock_plan_supervisor_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    await workflow.run("test_goal")

    assert mock_goal_disambiguator_component.return_value.attach.call_count == 1

    assert mock_agent.call_count == 3
    assert mock_agent.return_value.run.call_count >= 5

    assert mock_tools_executor.call_count == 3
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 3
    assert mock_handover_agent.return_value.run.call_count >= 1

    assert mock_plan_supervisor_agent.call_count == 3
    assert mock_plan_supervisor_agent.return_value.run.call_count >= 2

    assert mock_git_lab_workflow_instance.aput.call_count >= 1
    assert mock_git_lab_workflow_instance.aget_tuple.call_count >= 1

    mock_user_interface_instance.send_event.assert_called_with(
        type=ANY, state=ANY, stream=False
    )
    assert mock_user_interface_instance.send_event.call_count >= 2

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_memory_saver(
    mock_goal_disambiguator_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
):

    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_tools_registry.approval_required.return_value = False
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = MemorySaver()
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = True
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )

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
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        },
    ]

    mock_handover_agent.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
    ]

    mock_agent.return_value.run.side_effect = [
        *_agent_responses(
            WorkflowStatusEnum.PLANNING, "context_builder"
        ),  # context builder responses
        *_agent_responses(WorkflowStatusEnum.PLANNING, "planner"),  # planner responses
        *_agent_responses(
            WorkflowStatusEnum.EXECUTION, "executor"
        ),  # executor responses
    ]

    mock_plan_supervisor_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    await workflow.run("test_goal")

    assert mock_agent.call_count == 3
    assert mock_agent.return_value.run.call_count >= 5

    assert mock_tools_executor.call_count == 3
    assert mock_tools_executor.return_value.run.call_count >= 1

    assert mock_handover_agent.call_count == 3
    assert mock_handover_agent.return_value.run.call_count >= 1

    assert mock_plan_supervisor_agent.call_count == 3
    assert mock_plan_supervisor_agent.return_value.run.call_count >= 2

    assert mock_git_lab_workflow_instance.aput.call_count == 0
    assert mock_git_lab_workflow_instance.aget_tuple.call_count == 0

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_when_exception(
    mock_goal_disambiguator_component,
    chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_gitlab_workflow,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry,
):
    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry.configure = AsyncMock(
        return_value=MagicMock(spec=ToolsRegistry)
    )

    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        def __anext__(self):
            raise asyncio.CancelledError()

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
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
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.new_chat_client",
    autospec=True,
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_error_state(
    mock_goal_disambiguator_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
):
    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_tools_registry.approval_required.return_value = False

    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    mock_tools_executor.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.ERROR,
            "conversation_history": {},
        }
    ]

    mock_agent.return_value.run.side_effect = [
        *_agent_responses(WorkflowStatusEnum.PLANNING, "context_builder")
    ]

    mock_plan_supervisor_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    await workflow.run("test_goal")

    assert mock_agent.call_count == 3
    assert mock_agent.return_value.run.call_count == 2

    assert mock_tools_executor.call_count == 3
    assert mock_tools_executor.return_value.run.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry")
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
async def test_workflow_run_with_tools_registry(
    mock_goal_disambiguator_component,
    mock_gitlab_workflow,
    chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
):
    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.return_value = mock_tools_registry
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)

    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    class AsyncIterator:
        def __init__(self):
            pass

        def __aiter__(self):
            return self

        def __anext__(self):
            raise asyncio.CancelledError()

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
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
    mock_tools_registry.get.assert_has_calls(
        [
            call("get_plan"),
            call("add_new_task"),
            call("remove_task"),
            call("update_task_description"),
        ],
        any_order=True,
    )
    # Verify get_plan is called once in executor setup and twice in planner setup
    assert mock_tools_registry.get.call_args_list.count(call("get_plan")) == 3


@pytest.fixture
def tools_registry(tool_metadata):
    """Create a tools registry with all privileges enabled."""
    return ToolsRegistry(
        enabled_tools=list(_AGENT_PRIVILEGES.keys()),
        preapproved_tools=list(_AGENT_PRIVILEGES.keys()),
        tool_metadata=tool_metadata,
    )


@pytest.fixture
def software_development_workflow():
    """Create a software development workflow instance."""
    workflow = Workflow(
        "test",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    workflow._project = {"id": 1, "name": "test", "http_url_to_repo": "http://test"}  # type: ignore
    workflow._http_client = MagicMock()
    return workflow


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
# The next three tests check that the tools defined in the agent setup methods in the workflow are actually in
# the registry and match the list in the test.


@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
def test_executor_tools(tools_registry, software_development_workflow):
    agent_components = software_development_workflow._setup_executor(
        "test goal", tools_registry, MagicMock()
    )
    assert agent_components["toolset"] == tools_registry.toolset(EXECUTOR_TOOLS)
    assert_tools_in_tools_registry(tools_registry, agent_components["toolset"])


@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
def test_planner_tools(tools_registry, software_development_workflow):
    agent_components = software_development_workflow._setup_planner(
        "test goal", tools_registry, MagicMock(), MagicMock(spec=Toolset)
    )
    assert agent_components["toolset"] == tools_registry.toolset(PLANNER_TOOLS)
    assert_tools_in_tools_registry(tools_registry, agent_components["toolset"])


@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
def test_context_builder_tools(tools_registry, software_development_workflow):
    """Test that all tools used by the context builder agent are available in the tools registry."""
    agent_components = software_development_workflow._setup_context_builder(
        "test goal", tools_registry
    )
    assert agent_components["toolset"] == tools_registry.toolset(CONTEXT_BUILDER_TOOLS)
    assert_tools_in_tools_registry(tools_registry, agent_components["toolset"])


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_setup_error(
    mock_goal_disambiguator_component,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_gitlab_workflow,
    mock_tools_registry,
):
    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry.configure = AsyncMock(
        side_effect=Exception("Failed to configure tools")
    )

    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    await workflow.run("test_goal")

    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_missing_web_url(
    mock_fetch_project_data_with_workflow_id,
    mock_gitlab_workflow,
    checkpoint_tuple,
):
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        # web_url is missing
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    await workflow.run("test_goal")
    assert workflow.is_done


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.workflows.abstract_workflow.GitLabUrlParser", autospec=True
)
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_invalid_web_url(
    mock_fetch_project_data_with_workflow_id,
    mock_gitlab_workflow,
    mock_gitlab_url_parser,
    checkpoint_tuple,
):
    # Test case for invalid web_url (cannot extract gitlab_host)
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "invalid-url",  # Invalid URL format
    }

    mock_gitlab_url_parser.extract_host_from_url.return_value = None

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=checkpoint_tuple)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[checkpoint_tuple])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    await workflow.run("test_goal")
    assert workflow.is_done


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.GoalDisambiguationComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_retry(
    mock_goal_disambiguator_component,
    chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_gitlab_workflow,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry,
    checkpoint_tuple,
):
    mock_goal_disambiguator_component.return_value.attach.return_value = "planning"
    mock_tools_registry.configure = AsyncMock(
        return_value=MagicMock(spec=ToolsRegistry)
    )

    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )

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

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
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
            checkpoint=mock_checkpoint,  # type:ignore
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
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface", autospec=True)
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry", autospec=True)
@patch("duo_workflow_service.workflows.software_development.workflow.Agent")
@patch("duo_workflow_service.workflows.software_development.workflow.HandoverAgent")
@patch(
    "duo_workflow_service.workflows.software_development.workflow.PlanSupervisorAgent"
)
@patch("duo_workflow_service.workflows.software_development.workflow.ToolsExecutor")
@patch(
    "duo_workflow_service.workflows.abstract_workflow.fetch_project_data_with_workflow_id"
)
@patch("duo_workflow_service.workflows.abstract_workflow.fetch_workflow_config")
@patch("duo_workflow_service.workflows.software_development.workflow.new_chat_client")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow", autospec=True)
@patch(
    "duo_workflow_service.workflows.software_development.workflow.ToolsApprovalComponent",
    autospec=True,
)
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_workflow_run_with_tool_approvals(
    mock_tools_approval_component,
    mock_gitlab_workflow,
    mock_chat_client,
    mock_fetch_workflow_config,
    mock_fetch_project_data_with_workflow_id,
    mock_tools_executor,
    mock_plan_supervisor_agent,
    mock_handover_agent,
    mock_agent,
    mock_tools_registry_cls,
    checkpoint_tuple,
):
    mock_tools_registry = MagicMock(spec=ToolsRegistry)
    mock_tools_registry_cls.configure = AsyncMock(return_value=mock_tools_registry)
    mock_fetch_project_data_with_workflow_id.return_value = {
        "id": 1,
        "name": "test-project",
        "description": "This is a test project",
        "http_url_to_repo": "https://example.com/project",
        "web_url": "https://example.com/project",
    }
    mock_fetch_workflow_config.return_value = {
        "id": 1,
        "project_id": 1,
    }

    mock_git_lab_workflow_instance = mock_gitlab_workflow.return_value
    mock_git_lab_workflow_instance.__aenter__.return_value = (
        mock_git_lab_workflow_instance
    )
    mock_git_lab_workflow_instance.__aexit__.return_value = None
    mock_git_lab_workflow_instance._offline_mode = False
    mock_git_lab_workflow_instance.aget_tuple = AsyncMock(return_value=None)
    mock_git_lab_workflow_instance.alist = AsyncMock(return_value=[])
    mock_git_lab_workflow_instance.aput = AsyncMock(
        return_value={
            "configurable": {"thread_id": "123", "checkpoint_id": "checkpoint1"}
        }
    )
    mock_git_lab_workflow_instance.get_next_version = MagicMock(return_value=1)

    mock_handover_agent.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.COMPLETED,
            "conversation_history": {},
        },
    ]

    mock_tools_registry.approval_required.return_value = [True, False, False]

    mock_agent.return_value.run.side_effect = [
        {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {
                "context_builder": [
                    SystemMessage(content="system message"),
                    HumanMessage(content="human message"),
                    AIMessage(
                        content="Tool calls are present, route to planner tools execution",
                        tool_calls=[
                            {
                                "id": "1",
                                "name": "update_plan",
                                "args": {"summary": "done"},
                            }
                        ],
                    ),
                ],
            },
        },
    ]

    mock_plan_supervisor_agent.return_value.run.return_value = {
        "plan": Plan(steps=[]),
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
    }

    mock_tools_aprroval_execution = MagicMock()
    mock_tools_aprroval_execution.return_value = {"status": WorkflowStatusEnum.PLANNING}
    mock_tools_approval_component.side_effect = [
        MockComponent(
            mock_node_run=mock_tools_aprroval_execution,
            approved_agent_name="context_builder",
        ),
        MockComponent(
            mock_node_run=mock_tools_aprroval_execution, approved_agent_name="executor"
        ),
    ]

    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    await workflow.run("test_goal")

    assert mock_agent.call_count == 3
    assert mock_tools_aprroval_execution.call_count == 1

    assert workflow.is_done


@pytest.mark.asyncio
async def test_get_from_outbox():
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    workflow._outbox.put_nowait("test_item")
    item = await workflow.get_from_outbox()
    assert item == "test_item"


def test_add_to_inbox():
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    event = contract_pb2.ClientEvent()
    workflow.add_to_inbox(event)
    assert workflow._inbox.qsize() == 1
    assert workflow._inbox.get_nowait() == event


@pytest.mark.asyncio
async def test_workflow_cleanup():
    workflow = Workflow(
        "123",
        {},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    assert workflow._outbox.empty()
    assert workflow._inbox.empty()

    workflow._outbox.put_nowait("test_outbox_item_1")
    workflow._inbox.put_nowait("test_inbox_item_1")

    assert workflow._outbox.qsize() == 1
    assert workflow._inbox.qsize() == 1
    assert not workflow.is_done

    await workflow.cleanup("123")

    assert workflow.is_done
    assert workflow._outbox.qsize() == 0
    assert workflow._inbox.qsize() == 0
