import re
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.components import PlanApprovalComponent, ToolsRegistry
from duo_workflow_service.components.planner.component import PlannerComponent, Routes
from duo_workflow_service.entities import Plan, Task, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="approval_component")
def approval_component_fixture(mock_toolset):
    mock = MagicMock(
        spec=PlanApprovalComponent,
        toolset=mock_toolset,
    )
    mock.attach.return_value = END
    return mock


@pytest.fixture(name="mock_build_agent")
def mock_build_agent():
    with patch("duo_workflow_service.components.planner.component.build_agent") as mock:
        yield mock


@pytest.fixture(name="mock_tool")
def mock_tool_fixture() -> Mock:
    mock = MagicMock(DuoBaseTool)
    mock.args_schema = None
    mock.name = "test_tool"
    mock.description = "Test description"
    return mock


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    mock = MagicMock()
    mock.__getitem__.return_value = mock_tool
    return mock


@pytest.fixture(name="mock_tool_registry")
def mock_tool_registry_fixture(mock_tool):
    """Create a mock tool registry."""
    registry = MagicMock(ToolsRegistry)
    registry.get.return_value = mock_tool
    return registry


@pytest.fixture(name="mock_tools_executor")
def mock_tools_executor_fixture():
    with patch(
        "duo_workflow_service.components.planner.component.ToolsExecutor"
    ) as mock:
        mock.return_value.run.return_value = {
            "plan": Plan(steps=[MagicMock(spec=Task)]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

        yield mock


@pytest.fixture(name="mock_supervisor_agent")
def mock_supervisor_agent_fixture():
    with patch(
        "duo_workflow_service.components.planner.component.PlanSupervisorAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "conversation_history": {
                "planner": [
                    HumanMessage(
                        content="What is the next task? Call handover agent if task is complete"
                    )
                ]
            }
        }
        yield mock


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestPlannerComponent:
    @pytest.fixture(name="goal")
    def goal_fixture(self) -> str:
        return "Test goal"

    @pytest.fixture(name="mock_dependencies")
    def mock_dependencies_fixture(
        self,
        workflow_type,
        goal,
        mock_toolset,
        mock_tool,
        mock_tool_registry,
        gl_http_client,
        project,
        user,
    ):
        return {
            "workflow_id": "test-workflow-123",
            "workflow_type": workflow_type,
            "goal": goal,
            "planner_toolset": mock_toolset,
            "executor_toolset": {"test_tool": mock_tool},
            "tools_registry": mock_tool_registry,
            "project": project,
            "http_client": gl_http_client,
            "user": user,
        }

    @pytest.fixture(name="planner_component")
    def planner_component_fixture(self, mock_dependencies) -> PlannerComponent:
        return PlannerComponent(**mock_dependencies)

    @pytest.fixture(name="compiled_graph")
    def compiled_graph_fixture(self, planner_component, approval_component):
        graph = StateGraph(WorkflowState)

        entry_point = planner_component.attach(
            graph=graph,
            exit_node=END,
            next_node=END,
            approval_component=approval_component,
        )
        graph.set_entry_point(entry_point)

        return graph.compile()

    def test_init(self, mock_dependencies, workflow_type, goal):
        """Test PlannerComponent initialization."""
        component = PlannerComponent(**mock_dependencies)

        assert component.workflow_id == "test-workflow-123"
        assert component.workflow_type == workflow_type
        assert component.goal == goal
        assert component.planner_toolset == mock_dependencies["planner_toolset"]
        assert component.executor_toolset == mock_dependencies["executor_toolset"]
        assert component.tools_registry == mock_dependencies["tools_registry"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    def test_attach_creates_nodes_and_edges(
        self,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        planner_component,
        compiled_graph,
    ):
        """Test that attach method creates all necessary nodes and edges."""
        # Setup mocks
        mock_graph = Mock(spec=StateGraph)

        # Execute
        entry_node = planner_component.attach(
            mock_graph, "exit_node", "next_node", None
        )

        # Verify nodes are added
        expected_calls = [
            call("planning", mock_agent.run),
            call("update_plan", mock_tools_executor.return_value.run),
            call("planning_supervisor", mock_supervisor_agent.return_value.run),
        ]
        mock_graph.add_node.assert_has_calls(expected_calls)

        # Verify edges are added
        mock_graph.add_conditional_edges.assert_called_once()
        mock_graph.add_edge.assert_has_calls(
            [call("update_plan", "planning"), call("planning_supervisor", "planning")]
        )

        # Verify conditional edges routing
        call_args = mock_graph.add_conditional_edges.call_args
        routing_dict = call_args[0][2]

        assert routing_dict[Routes.CALL_TOOL] == "update_plan"
        assert routing_dict[Routes.SUPERVISOR] == "planning_supervisor"
        assert routing_dict[Routes.HANDOVER] == "next_node"
        assert routing_dict[Routes.STOP] == "exit_node"

        # Verify return value
        assert entry_node == "planning"

    def test_attach_creates_agent_with_correct_parameters(
        self,
        mock_build_agent: Mock,
        planner_component: PlannerComponent,
        workflow_type: CategoryEnum,
        mock_tool: Mock,
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)

        planner_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        mock_build_agent.assert_called_once_with(
            "planner",
            planner_component.prompt_registry,
            planner_component.user,
            "workflow/planner",
            "^1.0.0",
            tools=planner_component.planner_toolset.bindable,
            workflow_id="test-workflow-123",
            workflow_type=workflow_type,
            http_client=planner_component.http_client,
            prompt_template_inputs={
                "executor_agent_tools": f"{mock_tool.name}: {mock_tool.description}",
                "create_plan_tool_name": "test_tool",
                "get_plan_tool_name": "test_tool",
                "add_new_task_tool_name": "test_tool",
                "remove_task_tool_name": "test_tool",
                "update_task_description_tool_name": "test_tool",
                "agent_user_environment": {},
            },
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {
                        "planner": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to update_plan",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "create_plan",
                                        "args": {"task-0": "test"},
                                    }
                                ],
                            ),
                        ],
                    },
                },
                {
                    "plan": Plan(steps=[MagicMock(spec=Task)]),
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {
                        "planner": [
                            AIMessage(
                                content="No tool calls, route to planning supervisor",
                            ),
                        ],
                    },
                },
                {
                    "status": WorkflowStatusEnum.COMPLETED,
                    "conversation_history": {
                        "planner": [
                            AIMessage(
                                content="Done with the planning",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "handover_tool",
                                        "args": {
                                            "summary": "created plan with x steps"
                                        },
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
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        planner_component,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.run.call_count == 3

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["conversation_history"]["planner"]) == 6

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "plan": Plan(steps=[]),
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {
                        "planner": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to update_plan",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "create_plan",
                                        "args": {"task-0": "test"},
                                    }
                                ],
                            ),
                        ],
                    },
                },
                {
                    "status": WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
                    "conversation_history": {
                        "planner": [
                            AIMessage(
                                content="Done with the planning, route to plan approval",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "handover_tool",
                                        "args": {
                                            "summary": "created plan with x steps"
                                        },
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
        mock_tools_executor,
        mock_agent,
        approval_component,
        planner_component,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)
        mock_tools_executor.return_value.run.assert_called_once()

        assert mock_agent.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED
        assert len(response["conversation_history"]["planner"]) == 4
        approval_component.attach.assert_called_once()
