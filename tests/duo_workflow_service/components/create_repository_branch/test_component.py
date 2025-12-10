from unittest.mock import Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.components.create_repository_branch.component import (
    CreateRepositoryBranchComponent,
    Routes,
)
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_build_agent")
def mock_build_agent_fixture():
    with patch(
        "duo_workflow_service.components.create_repository_branch.component.build_agent"
    ) as mock:
        yield mock


@pytest.fixture(name="mock_tools_executor")
def mock_tools_executor_fixture():
    with patch(
        "duo_workflow_service.components.create_repository_branch.component.ToolsExecutor"
    ) as mock:
        mock.return_value.run.return_value = {
            "status": WorkflowStatusEnum.NOT_STARTED,
            "conversation_history": {},
        }
        yield mock


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestCreateRepositoryBranchComponent:
    @pytest.fixture(name="goal")
    def goal_fixture(self) -> str:
        return "https://gitlab.com/test/project/-/issues/1"

    @pytest.fixture(name="mock_dependencies")
    def mock_dependencies_fixture(
        self,
        workflow_type,
        goal,
        mock_toolset,
        mock_tool_registry,
        gl_http_client,
        project,
        user,
    ):
        return {
            "workflow_id": "test-workflow-123",
            "workflow_type": workflow_type,
            "goal": goal,
            "toolset": mock_toolset,
            "tools_registry": mock_tool_registry,
            "project": project,
            "http_client": gl_http_client,
            "user": user,
        }

    @pytest.fixture(name="create_branch_component")
    def create_branch_component_fixture(
        self, mock_dependencies
    ) -> CreateRepositoryBranchComponent:
        return CreateRepositoryBranchComponent(**mock_dependencies)

    @pytest.fixture(name="compiled_graph")
    def compiled_graph_fixture(self, create_branch_component):
        graph = StateGraph(WorkflowState)

        entry_point = create_branch_component.attach(
            graph=graph,
            exit_node=END,
            next_node=END,
        )
        graph.set_entry_point(entry_point)

        return graph.compile()

    def test_init(self, mock_dependencies, workflow_type, goal, project):
        """Test CreateRepositoryBranchComponent initialization."""
        component = CreateRepositoryBranchComponent(**mock_dependencies)

        assert component.workflow_id == "test-workflow-123"
        assert component.workflow_type == workflow_type
        assert component.goal == goal
        assert component.toolset == mock_dependencies["toolset"]
        assert component.tools_registry == mock_dependencies["tools_registry"]
        assert component.project == project
        assert component.http_client == mock_dependencies["http_client"]

    def test_attach_creates_nodes_and_edges(
        self,
        mock_tools_executor,
        mock_agent,
        create_branch_component,
    ):
        """Test that attach method creates all necessary nodes and edges."""
        mock_graph = Mock(spec=StateGraph)

        entry_node = create_branch_component.attach(
            mock_graph, "exit_node", "next_node"
        )
        expected_calls = [
            call("create_branch", mock_agent.run),
            call("create_branch_tools", mock_tools_executor.return_value.run),
        ]
        mock_graph.add_node.assert_has_calls(expected_calls)

        # Verify conditional edges are added
        assert mock_graph.add_conditional_edges.call_count == 2

        # Verify conditional edges routing from create_branch
        first_conditional_call = mock_graph.add_conditional_edges.call_args_list[0]
        assert first_conditional_call[0][0] == "create_branch"
        routing_dict = first_conditional_call[0][2]
        assert routing_dict[Routes.CALL_TOOL] == "create_branch_tools"
        assert routing_dict[Routes.HANDOVER] == "create_branch_handover"
        assert routing_dict[Routes.STOP] == "exit_node"

        # Verify conditional edges routing from create_branch_tools
        second_conditional_call = mock_graph.add_conditional_edges.call_args_list[1]
        assert second_conditional_call[0][0] == "create_branch_tools"
        routing_dict = second_conditional_call[0][2]
        assert routing_dict[Routes.CONTINUE] == "create_branch"
        assert routing_dict[Routes.STOP] == "exit_node"

        # Verify edge from create_branch_handover to next_node
        mock_graph.add_edge.assert_called_once_with(
            "create_branch_handover", "next_node"
        )

        # Verify return value
        assert entry_node == "create_branch"

    def test_attach_creates_agent_with_correct_parameters(
        self,
        mock_build_agent: Mock,
        create_branch_component: CreateRepositoryBranchComponent,
        workflow_type: CategoryEnum,
        project,
    ):
        """Test that Agent is created with correct parameters."""
        mock_graph = Mock(spec=StateGraph)

        create_branch_component.attach(mock_graph, "exit_node", "next_node")

        # Verify Agent was called with correct parameters
        mock_build_agent.assert_called_once_with(
            "create_branch",
            create_branch_component.prompt_registry,
            create_branch_component.user,
            "workflow/create_branch",
            "^1.0.0",
            tools=create_branch_component.toolset.bindable,
            workflow_id="test-workflow-123",
            workflow_type=workflow_type,
            http_client=create_branch_component.http_client,
            prompt_template_inputs={
                "goal": create_branch_component.goal,
                "ref": project["default_branch"],
                "workflow_id": create_branch_component.workflow_id,
                "project_id": project["id"],
                "repository_url": project["http_url_to_repo"],
            },
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "status": WorkflowStatusEnum.NOT_STARTED,
                    "conversation_history": {
                        "create_branch": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Tool calls are present, route to create_branch_tools",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "test_tool",
                                        "args": {"title": "test"},
                                    }
                                ],
                            ),
                        ],
                    },
                },
                {
                    "status": WorkflowStatusEnum.NOT_STARTED,
                    "conversation_history": {
                        "create_branch": [
                            AIMessage(
                                content="Done with branch creation",
                                tool_calls=[
                                    {
                                        "id": "2",
                                        "name": "handover_tool",
                                        "args": {"summary": "created branch"},
                                    }
                                ],
                            ),
                        ],
                    },
                },
            ]
        ],
    )
    async def test_component_run_with_tool_call(
        self,
        mock_tools_executor,
        mock_agent,
        graph_input,
        graph_config,
        compiled_graph,
    ):
        """Test flow when agent calls a tool."""
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # Verify tools executor was called
        mock_tools_executor.return_value.run.assert_called_once()
        # Verify agent was called twice (once for tool call, once for handover)
        assert mock_agent.run.call_count == 2

        assert response["status"] == WorkflowStatusEnum.NOT_STARTED
        assert len(response["conversation_history"]["create_branch"]) == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "status": WorkflowStatusEnum.NOT_STARTED,
                    "conversation_history": {
                        "create_branch": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(
                                content="Done with branch creation, route to handover",
                                tool_calls=[
                                    {
                                        "id": "1",
                                        "name": "handover_tool",
                                        "args": {
                                            "summary": "created branch successfully"
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
    async def test_component_run_with_handover(
        self,
        mock_agent,
        graph_input,
        graph_config,
        compiled_graph,
    ):
        """Test flow when agent calls handover_tool."""
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # Verify agent was called once
        assert mock_agent.run.call_count == 1

        # Verify status is set to NOT_STARTED by HandoverAgent
        assert response["status"] == WorkflowStatusEnum.NOT_STARTED
        assert len(response["conversation_history"]["create_branch"]) == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses",
        [
            [
                {
                    "status": WorkflowStatusEnum.ERROR,
                    "conversation_history": {
                        "create_branch": [
                            SystemMessage(content="system message"),
                            HumanMessage(content="human message"),
                            AIMessage(content="Error occurred"),
                        ],
                    },
                },
            ]
        ],
    )
    async def test_component_run_with_stop(
        self,
        mock_agent,
        graph_input,
        graph_config,
        compiled_graph,
    ):
        """Test flow when status is ERROR or CANCELLED."""
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # Verify agent was called once
        assert mock_agent.run.call_count == 1

        # Verify status remains ERROR and execution stops
        assert response["status"] == WorkflowStatusEnum.ERROR
        assert len(response["conversation_history"]["create_branch"]) == 3
