from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.executor.component import ExecutorComponent, Routes
from duo_workflow_service.entities import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool
from duo_workflow_service.workflows.type_definitions import AdditionalContext


@pytest.fixture
def approval_component(mock_toolset):
    return MagicMock(
        spec=ToolsApprovalComponent,
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
    mock.approved.return_value = True
    return mock


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry."""
    registry = MagicMock(ToolsRegistry)
    registry.get.return_value = MagicMock(name="test_tool")
    return registry


class TestExecutorComponent:

    @pytest.fixture
    def mock_dependencies(self, mock_toolset, mock_tool_registry, gl_http_client):
        return {
            "workflow_id": "test-workflow-123",
            "workflow_type": "test-workflow-type",
            "goal": "Test goal",
            "executor_toolset": mock_toolset,
            "tools_registry": mock_tool_registry,
            "model_config": "",
            "project": {
                "id": 123,
                "name": "test-project",
                "http_url_to_repo": "https://gitlab.com/test/repo",
            },
            "http_client": gl_http_client,
            "additional_context": None,
        }

    @pytest.fixture
    def executor_component(self, mock_dependencies):
        return ExecutorComponent(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test executorComponent initialization."""
        component = ExecutorComponent(**mock_dependencies)

        assert component.workflow_id == "test-workflow-123"
        assert component.workflow_type == "test-workflow-type"
        assert component.goal == "Test goal"
        assert component.executor_toolset == mock_dependencies["executor_toolset"]
        assert component.tools_registry == mock_dependencies["tools_registry"]
        assert component.model_config == mock_dependencies["model_config"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    @patch("duo_workflow_service.components.executor.component.create_chat_model")
    @patch("duo_workflow_service.components.executor.component.Agent")
    @patch("duo_workflow_service.components.executor.component.ToolsExecutor")
    @patch("duo_workflow_service.components.executor.component.PlanSupervisorAgent")
    @patch("duo_workflow_service.components.executor.component.HandoverAgent")
    def test_attach_creates_nodes_and_edges(
        self,
        mock_handover,
        mock_supervisor,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        executor_component,
        approval_component,
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
            call("execution", mock_agent.return_value.run),
            call("execution_tools", mock_tools_executor.return_value.run),
            call("execution_supervisor", mock_supervisor.return_value.run),
            call("execution_handover", mock_handover.return_value.run),
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

    @patch("duo_workflow_service.components.executor.component.create_chat_model")
    @patch("duo_workflow_service.components.executor.component.Agent")
    def test_attach_creates_agent_with_correct_parameters(
        self, mock_agent, mock_create_model, executor_component
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)
        executor_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args

        assert call_args[1]["name"] == "executor"
        assert call_args[1]["workflow_id"] == "test-workflow-123"
        assert call_args[1]["toolset"] == executor_component.executor_toolset
        assert call_args[1]["workflow_type"] == "test-workflow-type"

    @pytest.mark.asyncio
    @patch("duo_workflow_service.components.executor.component.create_chat_model")
    @patch("duo_workflow_service.components.executor.component.Agent")
    @patch("duo_workflow_service.components.executor.component.ToolsExecutor")
    @patch("duo_workflow_service.components.executor.component.PlanSupervisorAgent")
    @patch("duo_workflow_service.components.executor.component.HandoverAgent")
    async def test_component_run_with_no_approval_component(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        executor_component,
        approval_component,
        graph_input,
        graph_config,
        mock_tool_registry,
    ):
        graph = StateGraph(WorkflowState)

        mock_tool_registry.approval_required.return_value = False

        mock_agent.return_value.run.side_effect = [
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
        mock_supervisor_agent.return_value.run.return_value = {
            "conversation_history": {
                "executor": [
                    HumanMessage(
                        content="What is the next task? Call handover agent if task is complete"
                    )
                ]
            }
        }

        mock_handover_agent.return_value.run.return_value = {
            "status": WorkflowStatusEnum.COMPLETED,
            "handover": [AIMessage(content="This is a summary")],
        }

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

        entry_point = executor_component.attach(
            graph=graph, exit_node=END, next_node=END, approval_component=None
        )
        graph.set_entry_point(entry_point)
        compiled_graph = graph.compile()

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_called_once()
        mock_handover_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 3

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["handover"]) == 1
        assert response["handover"][-1] == AIMessage(content="This is a summary")
        assert len(response["conversation_history"]["executor"]) == 6

    @pytest.mark.asyncio
    @patch("duo_workflow_service.components.executor.component.create_chat_model")
    @patch("duo_workflow_service.components.executor.component.Agent")
    @patch("duo_workflow_service.components.executor.component.ToolsExecutor")
    @patch("duo_workflow_service.components.executor.component.PlanSupervisorAgent")
    @patch("duo_workflow_service.components.executor.component.HandoverAgent")
    async def test_component_run_with_approval_component(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        approval_component,
        executor_component,
        graph_input,
        graph_config,
        mock_tool_registry,
    ):
        graph = StateGraph(WorkflowState)

        mock_agent.return_value.run.side_effect = [
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

        mock_handover_agent.return_value.run.return_value = {
            "status": WorkflowStatusEnum.COMPLETED,
            "handover": [AIMessage(content="This is a summary")],
        }

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }
        approval_component_instance = approval_component.return_value
        approval_component_instance.attach.return_value = "execution_tools"

        entry_point = executor_component.attach(
            graph=graph,
            exit_node=END,
            next_node=END,
            approval_component=approval_component_instance,
        )
        graph.set_entry_point(entry_point)
        compiled_graph = graph.compile()

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_not_called()
        mock_handover_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["handover"]) == 1
        assert response["handover"][-1] == AIMessage(content="This is a summary")
        assert len(response["conversation_history"]["executor"]) == 4

    @pytest.mark.asyncio
    @patch("duo_workflow_service.components.executor.component.create_chat_model")
    @patch("duo_workflow_service.components.executor.component.Agent")
    @patch("duo_workflow_service.components.executor.component.ToolsExecutor")
    @patch("duo_workflow_service.components.executor.component.PlanSupervisorAgent")
    @patch("duo_workflow_service.components.executor.component.HandoverAgent")
    async def test_component_run_with_error(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        executor_component,
        graph_input,
        graph_config,
        mock_tool_registry,
    ):
        graph = StateGraph(WorkflowState)

        mock_tool_registry.approval_required.return_value = False
        mock_agent.return_value.run.side_effect = [
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

        mock_tools_executor.return_value.run.return_value = {
            "status": WorkflowStatusEnum.ERROR,
            "conversation_history": {},
        }

        entry_point = executor_component.attach(
            graph=graph, exit_node=END, next_node=END, approval_component=None
        )
        graph.set_entry_point(entry_point)
        compiled_graph = graph.compile()

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_not_called()
        mock_handover_agent.return_value.run.assert_not_called()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.ERROR
        assert len(response["handover"]) == 0
        assert len(response["conversation_history"]["executor"]) == 4

    @pytest.mark.parametrize(
        "additional_context,expected_os_info",
        [
            (None, ""),
            ([AdditionalContext(category="os_information", content="")], ""),
            (
                [AdditionalContext(category="os_information", content="Ubuntu 22.04")],
                "Ubuntu 22.04",
            ),
            (
                [
                    AdditionalContext(category="other_info", content="some data"),
                    AdditionalContext(category="os_information", content="Windows 11"),
                    AdditionalContext(category="more_info", content="more data"),
                ],
                "Windows 11",
            ),
            ([], ""),
            (
                [AdditionalContext(category="other_category", content="some content")],
                "",
            ),
        ],
    )
    def test_format_executor_message(
        self, additional_context, expected_os_info, executor_component
    ):
        executor_component.project = {
            "id": 123,
            "name": "Test Project",
            "http_url_to_repo": "https://github.com/test/project",
            "description": "This is a test project description",
            "web_url": "",
        }
        executor_component.additional_context = additional_context

        result = executor_component._format_system_prompt()

        assert "{os_information}" not in result

        if expected_os_info:
            assert f"<os_information>{expected_os_info}</os_information>" in result
            assert "Here is the information about the operating system" in result
        else:
            assert "<os_information>" not in result

        assert expected_os_info in result
