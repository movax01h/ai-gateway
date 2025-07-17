from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langgraph.graph import StateGraph
from langgraph.types import Command

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.experimental.flows.base import Flow
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.experimental.routers.router import Router
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum
from lib.internal_events.event_enum import CategoryEnum


class TestFlow:
    """Test Flow class functionality."""

    def mock_component(self, name: str):
        mock_component = MagicMock(spec=BaseComponent)
        mock_component.__entry_hook__.return_value = f"{name}_entry_node"
        return mock_component

    @contextmanager
    def mock_components(self, names: list[str]):
        mock_components = [self.mock_component(name) for name in names]

        with patch(
            "duo_workflow_service.agent_platform.experimental.flows.base.load_component_class"
        ) as mock_load_class:
            mock_load_class.side_effect = [
                MagicMock(return_value=mock_comp) for mock_comp in mock_components
            ]
            yield mock_components

    @pytest.fixture
    def mock_project(self):
        """Fixture providing mock project data."""
        return {
            "id": 123,
            "name": "test-project",
            "web_url": "https://gitlab.com/test/project",
        }

    @pytest.fixture
    def mock_state_graph(self, mock_project):
        # Create mock StateGraph and compiled graph
        mock_state_graph = Mock(spec=StateGraph)
        mock_compiled_graph = Mock()

        # Mock the compiled graph's astream method
        async def mock_astream(*args, **kwargs):  # pylint: disable=unused-argument
            yield ("values", {"status": "running"})
            yield ("updates", [{"step": "agent_processing"}])

        # mock_compiled_graph.astream = AsyncMock(side_effect=mock_astream)
        mock_compiled_graph.astream = Mock(return_value=mock_astream())
        mock_state_graph.compile.return_value = mock_compiled_graph
        ui_notifier = MagicMock()
        ui_notifier.send_event = AsyncMock()

        with (
            patch("duo_workflow_service.workflows.abstract_workflow.get_http_client"),
            patch(
                "duo_workflow_service.workflows.abstract_workflow.empty_workflow_config"
            ),
            patch(
                "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_project_data"
            ) as mock_fetch,
            patch(
                "duo_workflow_service.workflows.abstract_workflow.UserInterface",
                return_value=ui_notifier,
            ),
            patch(
                "duo_workflow_service.workflows.abstract_workflow.GitLabUrlParser"
            ) as mock_parser,
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.StateGraph",
                return_value=mock_state_graph,
            ),
        ):
            mock_fetch.return_value = (
                mock_project,
                {"config": "test"},
            )
            mock_parser.extract_host_from_url.return_value = "gitlab.com"

            yield mock_state_graph

    @pytest.fixture
    def mock_checkpointer(self):
        mock_checkpointer = Mock()
        mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
        mock_gitlab_workflow = AsyncMock()
        mock_gitlab_workflow.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_gitlab_workflow.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow",
            return_value=mock_gitlab_workflow,
        ):
            yield mock_checkpointer

    @pytest.fixture
    def mock_tools_registry(self):
        with patch(
            "duo_workflow_service.workflows.abstract_workflow.ToolsRegistry"
        ) as mock_tools_registry_class:

            mock_tools_registry = Mock()
            mock_tools_registry.toolset.return_value = []
            mock_tools_registry_class.configure = AsyncMock(
                return_value=mock_tools_registry
            )
            yield mock_tools_registry

    @pytest.fixture
    def mock_flow_metadata(self):
        """Fixture providing mock flow metadata."""
        return {
            "git_url": "https://gitlab.com/test/project",
            "git_sha": "abc123",
            "extended_logging": False,
        }

    @pytest.fixture
    def mock_invocation_metadata(self):
        """Fixture providing mock invocation metadata."""
        return {
            "base_url": "https://gitlab.com",
            "gitlab_token": "test-token",
        }

    @pytest.fixture
    def sample_flow_config(self):
        """Fixture providing a sample flow configuration."""
        return FlowConfig(
            flow={"entry_point": "agent"},
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="local",
            version="experimental",
        )

    @pytest.fixture
    def flow_instance(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        sample_flow_config,
        mock_checkpointer,
        mock_tools_registry,
        mock_state_graph,
    ):  # pylint: disable=unused-argument
        """Fixture providing a Flow instance with mocked dependencies."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.experimental.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                config=sample_flow_config,
                invocation_metadata=mock_invocation_metadata,
            )
            yield flow

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_event,goal,expected_type",
        [
            (WorkflowStatusEventEnum.START, "test goal", dict),
            (WorkflowStatusEventEnum.RESUME, "resume goal", Command),  # Command object
            ("unknown_event", "test goal", type(None)),
        ],
        ids=["start_event", "resume_event", "unknown_event"],
    )
    async def test_graph_input(
        self,
        flow_instance,
        status_event,
        goal,
        expected_type,
        mock_checkpointer,
        mock_state_graph,
        mock_project,
    ):
        """Test get_graph_input returns appropriate input based on status event."""
        mock_checkpointer.initial_status_event = status_event

        await flow_instance.run(goal)

        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]

        input = kwargs.get("input")
        if expected_type == dict:
            assert isinstance(input, expected_type)
            assert input["context"]["goal"] == goal
            assert input["context"]["project_id"] == mock_project["id"]
            assert input["status"] == WorkflowStatusEnum.NOT_STARTED
            assert "conversation_history" in input
            assert "ui_chat_log" in input
            assert len(input["ui_chat_log"]) == 1
            assert input["ui_chat_log"][0]["message_type"] == MessageTypeEnum.TOOL
            assert input["ui_chat_log"][0]["content"] == "Starting Flow: " + goal
            assert "context" in input
        elif expected_type == Command:
            assert hasattr(input, "resume")
            assert input.resume == goal
        else:
            assert input is None

    @pytest.mark.asyncio
    async def test_flow_config_validation_duplicate_component_names(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        mock_state_graph,
        mock_tools_registry,  # pylint: disable=unused-argument
        mock_checkpointer,  # pylint: disable=unused-argument
    ):
        """Test that duplicate component names are detected during compilation."""
        # Create config with duplicate component names
        duplicate_config = FlowConfig(
            flow={"entry_point": "agent"},
            components=[
                {"name": "agent", "type": "AgentComponent"},
                {"name": "agent", "type": "AnotherComponent"},  # Duplicate name
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="local",
            version="experimental",
        )

        with (
            self.mock_components(["AgentComponent", "AnotherComponent"]),
            patch("duo_workflow_service.agent_platform.experimental.flows.base.Router"),
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.log_exception"
            ) as mock_log_exception,
        ):
            # Create flow instance
            flow_instance = Flow(
                workflow_id="duplicated-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                config=duplicate_config,
                invocation_metadata=mock_invocation_metadata,
            )

            await flow_instance.run("test goal")

            mock_state_graph.compile.assert_not_called()
            mock_log_exception.assert_called_once()
            mock_log_exception_call = mock_log_exception.call_args
            assert isinstance(mock_log_exception_call[0][0], ValueError)
            assert "Duplicate component name: 'agent'" in str(
                mock_log_exception_call[0][0]
            )
            assert mock_log_exception_call[1]["extra"] == {
                "workflow_id": "duplicated-workflow-123"
            }

    @pytest.mark.asyncio
    async def test_flow_orchestration_with_complex_config(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        mock_state_graph,
        mock_tools_registry,
        mock_checkpointer,
    ):
        """Test Flow with complex configuration via run method to trigger _compile."""
        complex_config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "prompt_id": "agents/awesome",
                    "inputs": ["context:goal"],
                    "toolset": ["read_file", "edit_file"],
                },
                {
                    "name": "human_input",
                    "type": "HiltChatBackComponent",
                    "inputs": [{"from": "conversation_history:agent", "as": "history"}],
                },
            ],
            routers=[
                {"from": "agent", "to": "human_input"},
                {
                    "from": "human_input",
                    "condition": {
                        "input": "status",
                        "routes": {"Execution": "agent", "default_route": "end"},
                    },
                },
            ],
            flow={"entry_point": "agent"},
        )

        # Create mock component instances
        mock_agent_component = self.mock_component("agent_entry")
        mock_human_input_component = self.mock_component("human_input_entry")
        mock_end_component = Mock(spec=EndComponent)

        # Create mock component classes
        mock_agent_class = Mock(return_value=mock_agent_component)
        mock_human_input_class = Mock(return_value=mock_human_input_component)

        # Create mock router instances
        mock_simple_router = Mock(spec=Router)
        mock_conditional_router = Mock(spec=Router)

        # Setup tools registry mocks
        mock_tools_registry.toolset.return_value = ["read_file", "edit_file"]

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.load_component_class"
            ) as mock_load_class,
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.StateGraph",
                return_value=mock_state_graph,
            ),
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.Router"
            ) as mock_router_class,
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.base.EndComponent",
                return_value=mock_end_component,
            ) as mock_end_component_class,
        ):

            # Setup component loading mocks
            mock_load_class.side_effect = [
                mock_agent_class,  # For "AgentComponent"
                mock_human_input_class,  # For "HiltChatBackComponent"
            ]

            # Setup router creation mocks
            mock_router_class.side_effect = [
                mock_simple_router,
                mock_conditional_router,
            ]

            # Create flow instance
            flow = Flow(
                workflow_id="complex-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                config=complex_config,
                invocation_metadata=mock_invocation_metadata,
            )

            # Run the workflow to trigger _compile
            goal = "Complex workflow test"
            await flow.run(goal)

            # Assert all component classes were loaded (excluding EndComponent which is built-in)
            assert mock_load_class.call_count == 2
            mock_load_class.assert_any_call("AgentComponent")
            mock_load_class.assert_any_call("HiltChatBackComponent")

            # Assert all component instances were created with correct parameters
            # Agent component
            mock_tools_registry.toolset.assert_called_once_with(
                ["read_file", "edit_file"]
            )
            mock_agent_class.assert_called_once()
            agent_call_args = mock_agent_class.call_args[1]
            assert agent_call_args["name"] == "agent"
            assert agent_call_args["flow_id"] == "complex-workflow-123"
            assert agent_call_args["flow_type"] == CategoryEnum.WORKFLOW_CHAT
            assert agent_call_args["prompt_id"] == "agents/awesome"
            assert agent_call_args["inputs"] == ["context:goal"]
            assert agent_call_args["toolset"] == [
                "read_file",
                "edit_file",
            ]  # From tools_registry.toolset()

            # Human input component
            mock_human_input_class.assert_called_once()
            human_input_call_args = mock_human_input_class.call_args[1]
            assert human_input_call_args["name"] == "human_input"
            assert human_input_call_args["inputs"] == [
                {"from": "conversation_history:agent", "as": "history"}
            ]
            assert human_input_call_args["flow_id"] == "complex-workflow-123"
            assert human_input_call_args["flow_type"] == CategoryEnum.WORKFLOW_CHAT

            # EndComponent component
            mock_end_component_class.assert_called_once()
            end_component_call_args = mock_end_component_class.call_args[1]
            assert end_component_call_args["name"] == "end"
            assert end_component_call_args["flow_id"] == "complex-workflow-123"
            assert end_component_call_args["flow_type"] == CategoryEnum.WORKFLOW_CHAT
            mock_end_component.attach.assert_called_once_with(mock_state_graph)

            # Assert routers were created and attached
            assert mock_router_class.call_count == 2

            # Simple router (agent -> human_input)
            simple_router_call = mock_router_class.call_args_list[0]
            assert simple_router_call[1]["from_component"] == mock_agent_component
            assert simple_router_call[1]["to_component"] == mock_human_input_component

            # Conditional router (human_input -> agent/end based on condition)
            conditional_router_call = mock_router_class.call_args_list[1]
            assert (
                conditional_router_call[1]["from_component"]
                == mock_human_input_component
            )
            assert conditional_router_call[1]["input"] == "status"
            # Note: The "end" component in routes refers to the auto-created EndComponent
            assert "Execution" in conditional_router_call[1]["to_component"]
            assert "default_route" in conditional_router_call[1]["to_component"]
            assert (
                conditional_router_call[1]["to_component"]["default_route"]
                == mock_end_component
            )

            # Assert routers were attached to the graph
            mock_simple_router.attach.assert_called_once_with(mock_state_graph)
            mock_conditional_router.attach.assert_called_once_with(mock_state_graph)

            # Assert correct entry point was set
            mock_state_graph.set_entry_point.assert_called_once_with(
                "agent_entry_entry_node"
            )
            mock_agent_component.__entry_hook__.assert_called_once()

            # Assert graph was compiled with checkpointer
            mock_state_graph.compile.assert_called_once_with(
                checkpointer=mock_checkpointer
            )

            # Assert workflow completed
            assert flow.is_done
