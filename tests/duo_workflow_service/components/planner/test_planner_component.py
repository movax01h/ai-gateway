from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.components import PlanApprovalComponent, ToolsRegistry
from duo_workflow_service.components.planner.component import PlannerComponent, Routes
from duo_workflow_service.entities import Plan, Task, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool


@pytest.fixture
def approval_component(mock_toolset):
    return MagicMock(
        spec=PlanApprovalComponent,
        toolset=mock_toolset,
    )


@pytest.fixture
def mock_tool():
    mock = MagicMock(DuoBaseTool)
    mock.args_schema = None
    return mock


@pytest.fixture
def mock_toolset(mock_tool):
    mock = MagicMock()
    mock.__getitem__.return_value = mock_tool
    return mock


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock(ToolsRegistry)
    registry.get.return_value = MagicMock(name="test_tool")
    return registry


class TestPlannerComponent:
    @pytest.fixture
    def mock_dependencies(self, mock_toolset, mock_tool_registry, gl_http_client):
        return {
            "workflow_id": "test-workflow-123",
            "workflow_type": "test-workflow-type",
            "goal": "Test goal",
            "planner_toolset": mock_toolset,
            "executor_toolset": mock_toolset,
            "tools_registry": mock_tool_registry,
            "model_config": "",
            "project": {
                "id": 123,
                "name": "test-project",
                "http_url_to_repo": "https://gitlab.com/test/repo",
            },
            "http_client": gl_http_client,
        }

    @pytest.fixture
    def planner_component(self, mock_dependencies):
        return PlannerComponent(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test PlannerComponent initialization."""
        component = PlannerComponent(**mock_dependencies)

        assert component.workflow_id == "test-workflow-123"
        assert component.workflow_type == "test-workflow-type"
        assert component.goal == "Test goal"
        assert component.planner_toolset == mock_dependencies["planner_toolset"]
        assert component.executor_toolset == mock_dependencies["executor_toolset"]
        assert component.tools_registry == mock_dependencies["tools_registry"]
        assert component.model_config == mock_dependencies["model_config"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    @patch("duo_workflow_service.components.planner.component.create_chat_model")
    @patch("duo_workflow_service.components.planner.component.Agent")
    @patch("duo_workflow_service.components.planner.component.ToolsExecutor")
    @patch("duo_workflow_service.components.planner.component.PlanSupervisorAgent")
    def test_attach_creates_nodes_and_edges(
        self,
        mock_supervisor,
        mock_executor,
        mock_agent,
        mock_create_model,
        planner_component,
    ):
        """Test that attach method creates all necessary nodes and edges."""
        # Setup mocks
        mock_graph = Mock(spec=StateGraph)
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        planner_component.tools_registry.get.return_value = mock_tool

        # Execute
        entry_node = planner_component.attach(
            mock_graph, "exit_node", "next_node", None
        )

        # Verify nodes are added
        expected_calls = [
            call("planning", mock_agent.return_value.run),
            call("update_plan", mock_executor.return_value.run),
            call("planning_supervisor", mock_supervisor.return_value.run),
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

    @patch("duo_workflow_service.components.planner.component.create_chat_model")
    @patch("duo_workflow_service.components.planner.component.Agent")
    def test_attach_creates_agent_with_correct_parameters(
        self, mock_agent, mock_create_model, planner_component
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        planner_component.tools_registry.get.return_value = mock_tool

        planner_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args

        assert call_args[1]["name"] == "planner"
        assert call_args[1]["workflow_id"] == "test-workflow-123"
        assert call_args[1]["toolset"] == planner_component.planner_toolset
        assert call_args[1]["workflow_type"] == "test-workflow-type"

    @pytest.mark.asyncio
    @patch("duo_workflow_service.components.planner.component.create_chat_model")
    @patch("duo_workflow_service.components.planner.component.Agent")
    @patch("duo_workflow_service.components.planner.component.ToolsExecutor")
    @patch("duo_workflow_service.components.planner.component.PlanSupervisorAgent")
    async def test_component_run_with_no_approval_component(
        self,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        planner_component,
        graph_input,
        graph_config,
        mock_tool_registry,
    ):
        graph = StateGraph(WorkflowState)

        mock_agent.return_value.run.side_effect = [
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
                                    "args": {"summary": "created plan with x steps"},
                                }
                            ],
                        ),
                    ],
                },
            },
        ]
        mock_supervisor_agent.return_value.run.return_value = {
            "conversation_history": {
                "planner": [
                    HumanMessage(
                        content="What is the next task? Call handover agent if task is complete"
                    )
                ]
            }
        }

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[MagicMock(spec=Task)]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

        entry_point = planner_component.attach(
            graph=graph, exit_node=END, next_node=END, approval_component=None
        )
        graph.set_entry_point(entry_point)
        compiled_graph = graph.compile()

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 3

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["conversation_history"]["planner"]) == 6

    @pytest.mark.asyncio
    @patch("duo_workflow_service.components.planner.component.create_chat_model")
    @patch("duo_workflow_service.components.planner.component.Agent")
    @patch("duo_workflow_service.components.planner.component.ToolsExecutor")
    async def test_component_run_with_approval_component(
        self,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        approval_component,
        planner_component,
        graph_input,
        graph_config,
        mock_tool_registry,
    ):
        graph = StateGraph(WorkflowState)

        mock_agent.return_value.run.side_effect = [
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
                                    "args": {"summary": "created plan with x steps"},
                                }
                            ],
                        ),
                    ],
                },
            },
        ]

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[MagicMock(spec=Task)]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }
        approval_component_instance = approval_component.return_value
        approval_component_instance.attach.return_value = END

        entry_point = planner_component.attach(
            graph=graph,
            exit_node=END,
            next_node=END,
            approval_component=approval_component_instance,
        )
        graph.set_entry_point(entry_point)
        compiled_graph = graph.compile()

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)
        mock_tools_executor.return_value.run.assert_called_once()

        assert mock_agent.return_value.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED
        assert len(response["conversation_history"]["planner"]) == 4
        approval_component_instance.attach.assert_called_once()
