import re
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langgraph.constants import END
from langgraph.graph import StateGraph

from ai_gateway.models.mock import FakeModel
from duo_workflow_service.components import PlanApprovalComponent, ToolsRegistry
from duo_workflow_service.components.planner.component import PlannerComponent, Routes
from duo_workflow_service.components.planner.prompt import (
    HANDOVER_TOOL_NAME,
    PLANNER_GOAL,
    PLANNER_INSTRUCTIONS,
    PLANNER_PROMPT,
)
from duo_workflow_service.entities import Plan, Task, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import DuoBaseTool
from lib.feature_flags.context import current_feature_flag_context


@pytest.fixture(name="approval_component")
def approval_component_fixture(mock_toolset):
    mock = MagicMock(
        spec=PlanApprovalComponent,
        toolset=mock_toolset,
    )
    mock.attach.return_value = END
    return mock


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
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


@pytest.fixture(name="mock_create_model")
def mock_create_model_fixture():
    with patch(
        "duo_workflow_service.components.planner.component.create_chat_model"
    ) as mock:
        mock.return_value = mock
        mock.bind_tools.return_value = mock
        yield mock


@pytest.fixture(name="mock_model_ainvoke")
def mock_model_ainvoke_fixture(
    duo_workflow_prompt_registry_enabled, mock_create_model, end_message
):
    if duo_workflow_prompt_registry_enabled:
        with patch.object(FakeModel, "ainvoke") as mock:
            mock.return_value = end_message
            yield mock
    else:
        mock_create_model.ainvoke = AsyncMock(return_value=end_message)
        yield mock_create_model.ainvoke


@pytest.fixture(name="duo_workflow_prompt_registry_enabled")
def duo_workflow_prompt_registry_enabled_fixture() -> bool:
    return False


@pytest.fixture(autouse=True)
def stub_feature_flags(duo_workflow_prompt_registry_enabled: bool):
    if duo_workflow_prompt_registry_enabled:
        current_feature_flag_context.set({"duo_workflow_prompt_registry"})

    yield


@pytest.fixture(name="mock_agent")
def mock_agent_fixture(duo_workflow_prompt_registry_enabled):
    if duo_workflow_prompt_registry_enabled:
        factory = "ai_gateway.prompts.registry.LocalPromptRegistry.get_on_behalf"
    else:
        factory = "duo_workflow_service.components.planner.component.Agent"

    with patch(factory) as mock:
        yield mock


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
@pytest.mark.parametrize("duo_workflow_prompt_registry_enabled", [False, True])
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
            "model_config": "",
            "project": project,
            "http_client": gl_http_client,
            "user": user,
        }

    @pytest.fixture(name="planner_component")
    def planner_component_fixture(self, mock_dependencies):
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
        assert component.model_config == mock_dependencies["model_config"]
        assert component.project == mock_dependencies["project"]
        assert component.http_client == mock_dependencies["http_client"]

    def test_attach_creates_nodes_and_edges(
        self,
        mock_supervisor_agent,
        mock_tools_executor,
        mock_agent,
        mock_create_model,
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
            call("planning", mock_agent.return_value.run),
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
        mock_agent,
        mock_create_model,
        planner_component,
        duo_workflow_prompt_registry_enabled,
        workflow_type,
        mock_tool,
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)

        planner_component.attach(mock_graph, "exit_node", "next_node", None)

        # Verify Agent was called with correct parameters
        if duo_workflow_prompt_registry_enabled:
            mock_agent.assert_called_once_with(
                planner_component.user,
                "workflow/planner",
                "^1.0.0",
                tools=planner_component.planner_toolset.bindable,
                workflow_id="test-workflow-123",
                http_client=planner_component.http_client,
                prompt_template_inputs={
                    "executor_agent_tools": f"{mock_tool.name}: {mock_tool.description}",
                    "create_plan_tool_name": "test_tool",
                    "get_plan_tool_name": "test_tool",
                    "add_new_task_tool_name": "test_tool",
                    "remove_task_tool_name": "test_tool",
                    "update_task_description_tool_name": "test_tool",
                },
            )
        else:
            mock_agent.assert_called_once()

            call_args = mock_agent.call_args
            assert call_args[1]["name"] == "planner"
            assert call_args[1]["workflow_id"] == "test-workflow-123"
            assert call_args[1]["toolset"] == planner_component.planner_toolset
            assert call_args[1]["workflow_type"] == workflow_type

    @pytest.mark.asyncio
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
        compiled_graph,
    ):
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

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        mock_supervisor_agent.return_value.run.assert_called_once()
        mock_tools_executor.return_value.run.assert_called_once()
        assert mock_agent.return_value.run.call_count == 3

        assert response["status"] == WorkflowStatusEnum.COMPLETED
        assert len(response["conversation_history"]["planner"]) == 6

    @pytest.mark.asyncio
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
        compiled_graph,
    ):
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

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)
        mock_tools_executor.return_value.run.assert_called_once()

        assert mock_agent.return_value.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED
        assert len(response["conversation_history"]["planner"]) == 4
        approval_component.attach.assert_called_once()

    @pytest.mark.asyncio
    async def test_messages_to_model(
        self,
        mock_model_ainvoke,
        compiled_graph,
        workflow_state,
        graph_config,
        project,
        goal,
        mock_tool,
    ):
        await compiled_graph.ainvoke(input=workflow_state, config=graph_config)

        planner_instructions = PLANNER_INSTRUCTIONS.format(
            create_plan_tool_name="test_tool",
            add_new_task_tool_name="test_tool",
            remove_task_tool_name="test_tool",
            update_task_description_tool_name="test_tool",
            get_plan_tool_name="test_tool",
            handover_tool_name=HANDOVER_TOOL_NAME,
            project_id=project["id"],
            project_name=project["name"],
            project_url=project["http_url_to_repo"],
        )

        expected_goal = PLANNER_GOAL.format(
            handover_tool_name=HANDOVER_TOOL_NAME,
            executor_agent_tools=f"{mock_tool.name}: {mock_tool.description}",
            goal=goal,
            create_plan_tool_name="test_tool",
            get_plan_tool_name="test_tool",
            add_new_task_tool_name="test_tool",
            remove_task_tool_name="test_tool",
            update_task_description_tool_name="test_tool",
            planner_instructions=planner_instructions,
        )

        ainvoke_messages = mock_model_ainvoke.call_args.args[0]

        if isinstance(ainvoke_messages, ChatPromptValue):
            ainvoke_messages = ainvoke_messages.messages

        assert ainvoke_messages[0] == SystemMessage(content=PLANNER_PROMPT)

        # Check the content of the HumanMessage ignoring whitespaces, since they may differ between jinja and in-code
        # templates
        assert (
            re.sub(r"\s+", " ", ainvoke_messages[1].content).strip()
            == re.sub(r"\s+", " ", f"Your goal is: {expected_goal}").strip()
        )
