from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langgraph.constants import END
from langgraph.graph import StateGraph

from ai_gateway.models.mock import FakeModel
from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.executor.component import ExecutorComponent, Routes
from duo_workflow_service.components.executor.prompts import (
    EXECUTOR_SYSTEM_MESSAGE,
    GET_PLAN_TOOL_NAME,
    HANDOVER_TOOL_NAME,
    OS_INFORMATION_COMPONENT,
    SET_TASK_STATUS_TOOL_NAME,
)
from duo_workflow_service.entities import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.feature_flags.context import current_feature_flag_context


@pytest.fixture
def approval_component(mock_toolset):
    mock = MagicMock(
        spec=ToolsApprovalComponent,
        toolset=mock_toolset,
    )
    mock.attach.return_value = "execution_tools"

    return mock


@pytest.fixture
def compiled_graph(executor_component, approval_component):
    graph = StateGraph(WorkflowState)

    entry_point = executor_component.attach(
        graph=graph,
        exit_node=END,
        next_node=END,
        approval_component=approval_component,
    )
    graph.set_entry_point(entry_point)

    return graph.compile()


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


@pytest.fixture
def mock_create_model():
    with patch(
        "duo_workflow_service.components.executor.component.create_chat_model"
    ) as mock:
        mock.return_value = mock
        mock.bind_tools.return_value = mock
        yield mock


@pytest.fixture
def mock_model_ainvoke(
    duo_workflow_prompt_registry_enabled, mock_create_model, end_message
):
    if duo_workflow_prompt_registry_enabled:
        with patch.object(FakeModel, "ainvoke") as mock:
            mock.return_value = end_message
            yield mock
    else:
        mock_create_model.ainvoke = AsyncMock(return_value=end_message)
        yield mock_create_model.ainvoke


@pytest.fixture
def duo_workflow_prompt_registry_enabled() -> bool:
    return False


@pytest.fixture(autouse=True)
def stub_feature_flags(duo_workflow_prompt_registry_enabled: bool):
    if duo_workflow_prompt_registry_enabled:
        current_feature_flag_context.set({"duo_workflow_prompt_registry"})

    yield


@pytest.fixture
def mock_agent(duo_workflow_prompt_registry_enabled: bool):
    if duo_workflow_prompt_registry_enabled:
        factory = "ai_gateway.prompts.registry.LocalPromptRegistry.get_on_behalf"
    else:
        factory = "duo_workflow_service.components.executor.component.Agent"

    with patch(factory) as mock:
        yield mock


@pytest.fixture
def mock_tools_executor():
    with patch(
        "duo_workflow_service.components.executor.component.ToolsExecutor"
    ) as mock:
        yield mock


@pytest.fixture
def mock_supervisor_agent():
    with patch(
        "duo_workflow_service.components.executor.component.PlanSupervisorAgent"
    ) as mock:
        yield mock


@pytest.fixture
def mock_handover_agent():
    with patch(
        "duo_workflow_service.components.executor.component.HandoverAgent"
    ) as mock:
        mock.return_value.run.return_value = {
            "status": WorkflowStatusEnum.COMPLETED,
            "handover": [AIMessage(content="This is a summary")],
        }
        yield mock


@pytest.mark.usefixtures("mock_container")
@pytest.mark.parametrize("duo_workflow_prompt_registry_enabled", [False, True])
class TestExecutorComponent:
    @pytest.fixture
    def goal(self) -> str:
        return "Test goal"

    @pytest.fixture
    def mock_dependencies(
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
            "model_config": "",
            "project": project,
            "http_client": gl_http_client,
            "additional_context": additional_context,
            "user": user,
        }

    @pytest.fixture
    def executor_component(
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
        assert component.model_config == mock_dependencies["model_config"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    def test_attach_creates_nodes_and_edges(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
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
            call("execution", mock_agent.return_value.run),
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
        mock_agent,
        mock_create_model,
        executor_component,
        workflow_type,
        duo_workflow_prompt_registry_enabled,
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)
        executor_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        mock_agent.assert_called_once()

        if duo_workflow_prompt_registry_enabled:
            mock_agent.assert_called_once_with(
                executor_component.user,
                "workflow/executor",
                "^1.0.0",
                tools=executor_component.executor_toolset.bindable,
                workflow_id="test-workflow-123",
                http_client=executor_component.http_client,
            )
        else:
            call_args = mock_agent.call_args
            assert call_args[1]["name"] == "executor"
            assert call_args[1]["workflow_id"] == "test-workflow-123"
            assert call_args[1]["toolset"] == executor_component.executor_toolset
            assert call_args[1]["workflow_type"] == workflow_type

    @pytest.mark.asyncio
    async def test_component_run_with_no_approval_component(
        self,
        mock_handover_agent,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
        graph_input,
        graph_config,
        mock_tool_registry,
        compiled_graph,
    ):
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

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

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
        compiled_graph,
    ):
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

        mock_tools_executor.return_value.run.return_value = {
            "plan": Plan(steps=[]),
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
        }

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
        compiled_graph,
    ):
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

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_not_called()
        mock_handover_agent.return_value.run.assert_not_called()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.ERROR
        assert len(response["handover"]) == 0
        assert len(response["conversation_history"]["executor"]) == 4

    @pytest.mark.asyncio
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
    async def test_messages_to_model(
        self,
        expected_os_info,
        mock_model_ainvoke,
        compiled_graph,
        workflow_state,
        graph_config,
        project,
        goal,
    ):
        await compiled_graph.ainvoke(input=workflow_state, config=graph_config)

        os_information = (
            OS_INFORMATION_COMPONENT.format(os_information=expected_os_info)
            if expected_os_info
            else ""
        )

        expected_message = EXECUTOR_SYSTEM_MESSAGE.format(
            set_task_status_tool_name=SET_TASK_STATUS_TOOL_NAME,
            handover_tool_name=HANDOVER_TOOL_NAME,
            get_plan_tool_name=GET_PLAN_TOOL_NAME,
            project_id=project["id"],
            project_name=project["name"],
            project_url=project["http_url_to_repo"],
            os_information=os_information,
        )

        ainvoke_messages = mock_model_ainvoke.call_args.args[0]

        if isinstance(ainvoke_messages, ChatPromptValue):
            ainvoke_messages = ainvoke_messages.messages

        assert ainvoke_messages == [
            SystemMessage(content=expected_message),
            HumanMessage(content=f"Your goal is: {goal}"),
        ]
