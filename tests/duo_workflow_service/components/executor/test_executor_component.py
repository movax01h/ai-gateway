from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.executor.component import ExecutorComponent, Routes
from duo_workflow_service.entities import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_build_agent")
def mock_build_agent():
    with patch(
        "duo_workflow_service.components.executor.component.build_agent"
    ) as mock:
        yield mock


@pytest.fixture(name="approval_component")
def approval_component_fixture(mock_toolset):
    mock = MagicMock(
        spec=ToolsApprovalComponent,
        toolset=mock_toolset,
    )
    mock.attach.return_value = "execution_tools"

    return mock


@pytest.fixture(name="compiled_graph")
def compiled_graph_fixture(executor_component, approval_component):
    graph = StateGraph(WorkflowState)

    entry_point = executor_component.attach(
        graph=graph,
        exit_node=END,
        next_node=END,
        approval_component=approval_component,
    )
    graph.set_entry_point(entry_point)

    return graph.compile()


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    mock = MagicMock(DuoBaseTool)
    mock.args_schema = None
    return mock


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    mock = MagicMock()
    mock.__getitem__.return_value = mock_tool
    mock.approved.return_value = True
    return mock


@pytest.fixture(name="mock_tool_registry")
def mock_tool_registry_fixture():
    """Create a mock tool registry."""
    registry = MagicMock(ToolsRegistry)
    registry.get.return_value = MagicMock(name="test_tool")
    return registry


@pytest.fixture(name="mock_tools_executor")
def mock_tools_executor_fixture():
    with patch(
        "duo_workflow_service.components.executor.component.ToolsExecutor"
    ) as mock:
        yield mock


@pytest.fixture(name="mock_supervisor_agent")
def mock_supervisor_agent_fixture():
    with patch(
        "duo_workflow_service.components.executor.component.PlanSupervisorAgent"
    ) as mock:
        yield mock


@pytest.fixture(name="mock_handover_agent")
def mock_handover_agent_fixture():
    with patch(
        "duo_workflow_service.components.executor.component.HandoverAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "status": WorkflowStatusEnum.COMPLETED,
            "handover": [AIMessage(content="This is a summary")],
        }
        yield mock


@pytest.mark.usefixtures("mock_container")
class TestExecutorComponent:
    @pytest.fixture(name="goal")
    def goal_fixture(self) -> str:
        return "Test goal"

    @pytest.fixture(name="mock_dependencies")
    def mock_dependencies_fixture(
        self,
        mock_toolset,
        mock_tool_registry,
        gl_http_client,
        additional_context,
        workflow_type,
        goal,
        project,
        user,
    ):
        return {
            "workflow_id": "test-workflow-123",
            "workflow_type": workflow_type,
            "goal": goal,
            "executor_toolset": mock_toolset,
            "tools_registry": mock_tool_registry,
            "project": project,
            "http_client": gl_http_client,
            "additional_context": additional_context,
            "user": user,
        }

    @pytest.fixture(name="executor_component")
    def executor_component_fixture(
        self, mock_dependencies, mock_duo_workflow_service_container
    ):
        return ExecutorComponent(**mock_dependencies)

    def test_init(self, mock_dependencies, workflow_type, goal):
        """Test executorComponent initialization."""
        component = ExecutorComponent(**mock_dependencies)

        assert component.workflow_id == "test-workflow-123"
        assert component.workflow_type == workflow_type
        assert component.goal == goal
        assert component.executor_toolset == mock_dependencies["executor_toolset"]
        assert component.tools_registry == mock_dependencies["tools_registry"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    def test_attach_creates_nodes_and_edges(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        executor_component,
    ):
        """Test that attach method creates all necessary nodes and edges."""
        # Setup mocks
        mock_graph = Mock(spec=StateGraph)

        # Attach component
        entry_node = executor_component.attach(
            mock_graph, "exit_node", "next_node", None
        )

        # Verify nodes are added
        expected_calls = [
            call("execution", mock_agent.run),
            call("execution_tools", mock_tools_executor.return_value.run),
            call("execution_supervisor", mock_supervisor_agent.return_value.run),
            call("execution_handover", mock_handover_agent.return_value.run),
        ]
        mock_graph.add_node.assert_has_calls(expected_calls)

        # Verify edges are added
        mock_graph.add_edge.assert_has_calls(
            [
                call("execution_supervisor", "execution"),
                call("execution_tools", "execution"),
                call("execution_handover", "next_node"),
            ]
        )

        # Verify conditional edges routing
        call_args = mock_graph.add_conditional_edges.call_args
        routing_dict = call_args[0][2]

        assert routing_dict[Routes.CALL_TOOL] == "execution_tools"
        assert routing_dict[Routes.SUPERVISOR] == "execution_supervisor"
        assert routing_dict[Routes.HANDOVER] == "execution_handover"
        assert routing_dict[Routes.STOP] == "exit_node"

        # Verify return value
        assert entry_node == "execution"

    def test_attach_creates_agent_with_correct_parameters(
        self,
        mock_build_agent: Mock,
        executor_component: ExecutorComponent,
        workflow_type: CategoryEnum,
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)

        executor_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        mock_build_agent.assert_called_once_with(
            "executor",
            executor_component.prompt_registry,
            executor_component.user,
            "workflow/executor",
            "^2.0.0",
            tools=executor_component.executor_toolset.bindable,
            workflow_id="test-workflow-123",
            workflow_type=workflow_type,
            http_client=executor_component.http_client,
            prompt_template_inputs={
                "set_task_status_tool_name": "set_task_status",
                "get_plan_tool_name": "get_plan",
                "agent_user_environment": {},
                "agents_dot_md": None,
            },
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to executor tools execution",
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
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            AIMessage(
                                content="No tool calls, route to execution supervisor",
                            ),
                        ],
                    },
                },
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            AIMessage(
                                content="Done with the execution, over to handover agent",
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
        ],
    )
    async def test_component_run_with_no_approval_component(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
        mock_tool_registry.approval_required.return_value = False

        mock_supervisor_agent.return_value.run.return_value = {
            "conversation_history": {
                "executor": [
                    HumanMessage(
                        content="What is the next task? Call handover agent if task is complete"
                    )
                ]
            }
        }

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_called_once()
        mock_handover_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.run.call_count == 3

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["handover"]) == 1
        assert response["handover"][-1] == AIMessage(content="This is a summary")
        assert len(response["conversation_history"]["executor"]) == 6

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to executor tools execution.",
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
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            AIMessage(
                                content="Done with the execution, over to handover agent",
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
        ],
    )
    async def test_component_run_with_approval_component(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        approval_component,
        executor_component,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_not_called()
        mock_handover_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["handover"]) == 1
        assert response["handover"][-1] == AIMessage(content="This is a summary")
        assert len(response["conversation_history"]["executor"]) == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.EXECUTION,
                    "conversation_history": {
                        "executor": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to executor tools execution",
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
                    "status": WorkflowStatusEnum.ERROR,
                    "conversation_history": {
                        "executor": [
                            AIMessage(content="Failed, exiting workflow"),
                        ],
                    },
                },
            ]
        ],
    )
    async def test_component_run_with_error(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        executor_component,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
        mock_tool_registry.approval_required.return_value = False

        mock_tools_executor.return_value.run.return_value = {
            "status": WorkflowStatusEnum.ERROR,
            "conversation_history": {},
        }

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_not_called()
        mock_handover_agent.return_value.run.assert_not_called()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.ERROR
        assert len(response["handover"]) == 0
        assert len(response["conversation_history"]["executor"]) == 4
