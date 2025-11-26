from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.graph import StateGraph
from langgraph.types import Command

from contract import contract_pb2
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.v1.flows.base import Flow, UserDecision
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    FlowConfigInput,
    FlowConfigMetadata,
)
from duo_workflow_service.agent_platform.v1.routers.router import Router
from duo_workflow_service.agent_platform.v1.state.base import FlowEventType
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.internal_events.event_enum import CategoryEnum


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestFlow:  # pylint: disable=too-many-public-methods
    """Test Flow class functionality."""

    def mock_component(self, name: str):
        mock_component = MagicMock(spec=BaseComponent)
        mock_component.__entry_hook__.return_value = f"{name}_entry_node"
        return mock_component

    @contextmanager
    def mock_components(self, names: list[str]):
        mock_components = [self.mock_component(name) for name in names]

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.load_component_class"
        ) as mock_load_class:
            mock_load_class.side_effect = [
                MagicMock(return_value=mock_comp) for mock_comp in mock_components
            ]
            yield mock_components

    @pytest.fixture(name="mock_project")
    def mock_project_fixture(self):
        """Fixture providing mock project data."""
        return {
            "id": 123,
            "name": "test-project",
            "web_url": "https://gitlab.com/test/project",
        }

    @pytest.fixture(name="mock_state_graph")
    def mock_state_graph_fixture(self, mock_project):
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
                "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data"
            ) as mock_fetch,
            patch(
                "duo_workflow_service.workflows.abstract_workflow.UserInterface",
                return_value=ui_notifier,
            ),
            patch(
                "duo_workflow_service.gitlab.gitlab_api.GitLabUrlParser"
            ) as mock_parser,
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.StateGraph",
                return_value=mock_state_graph,
            ),
        ):
            mock_fetch.return_value = (
                mock_project,
                None,
                {"config": "test"},
            )
            mock_parser.extract_host_from_url.return_value = "gitlab.com"

            yield mock_state_graph

    @pytest.fixture(name="mock_checkpointer")
    def mock_checkpointer_fixture(self):
        mock_checkpointer = Mock()
        mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
        mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
        mock_gitlab_workflow = AsyncMock()
        mock_gitlab_workflow.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_gitlab_workflow.__aexit__ = AsyncMock(return_value=None)
        with patch(
            "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow",
            return_value=mock_gitlab_workflow,
        ):
            yield mock_checkpointer

    @pytest.fixture(name="mock_tools_registry")
    def mock_tools_registry_fixture(self):
        with patch(
            "duo_workflow_service.workflows.abstract_workflow.ToolsRegistry"
        ) as mock_tools_registry_class:

            mock_tools_registry = Mock()
            mock_tools_registry.toolset.return_value = []
            mock_tools_registry_class.configure = AsyncMock(
                return_value=mock_tools_registry
            )
            yield mock_tools_registry

    @pytest.fixture(name="mock_flow_metadata")
    def mock_flow_metadata_fixture(self):
        """Fixture providing mock flow metadata."""
        return {
            "git_url": "https://gitlab.com/test/project",
            "git_sha": "abc123",
            "extended_logging": False,
        }

    @pytest.fixture(name="mock_invocation_metadata")
    def mock_invocation_metadata_fixture(self):
        """Fixture providing mock invocation metadata."""
        return {
            "base_url": "https://gitlab.com",
            "gitlab_token": "test-token",
        }

    @pytest.fixture(name="sample_flow_config_metadata")
    def sample_flow_config_metadata_fixture(self):
        return FlowConfigMetadata(
            entry_point="agent",
            inputs=[
                FlowConfigInput(
                    category="file",
                    input_schema={
                        "contents": {"type": "string"},
                        "file_name": {"type": "string"},
                    },
                ),
                FlowConfigInput(
                    category="snippet", input_schema={"snippet_str": {"type": "string"}}
                ),
            ],
        )

    @pytest.fixture(name="sample_flow_config")
    def sample_flow_config_fixture(self, sample_flow_config_metadata):
        """Fixture providing a sample flow configuration."""
        return FlowConfig(
            flow=sample_flow_config_metadata,
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                    "prompt_id": "test/prompt",
                    "prompt_version": "v1",
                    "toolset": ["read_file"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="ambient",
            version="v1",
        )

    @pytest.fixture(name="checkpoint_tuple")
    def checkpoint_tuple_fixture(self):
        """Fixture providing a CheckpointTuple for testing."""
        return CheckpointTuple(
            config={
                "configurable": {"thread_id": "123", "checkpoint_id": str(uuid4())}
            },
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

    @pytest.fixture(name="flow_instance")
    def flow_instance_fixture(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        user,
        sample_flow_config,
        mock_checkpointer,
        mock_tools_registry,
        mock_state_graph,
    ):  # pylint: disable=unused-argument
        """Fixture providing a Flow instance with mocked dependencies."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
                config=sample_flow_config,
                invocation_metadata=mock_invocation_metadata,
            )
            yield flow

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_event,goal,expected_type,checkpoint_tuple_present",
        [
            (WorkflowStatusEventEnum.START, "test goal", dict, False),
            ("unknown_event", "test goal", type(None), False),
            (WorkflowStatusEventEnum.RETRY, "test goal", type(None), True),
            (WorkflowStatusEventEnum.RETRY, "test goal", dict, False),
        ],
        ids=[
            "start_event",
            "unknown_event",
            "retry_with_checkpoint",
            "retry_without_checkpoint",
        ],
    )
    async def test_graph_input(
        self,
        flow_instance: Flow,
        status_event,
        goal,
        expected_type,
        checkpoint_tuple_present,
        mock_checkpointer,
        mock_state_graph,
        mock_project,
        checkpoint_tuple,
    ):
        """Test get_graph_input returns appropriate input based on status event."""
        mock_checkpointer.initial_status_event = status_event
        if checkpoint_tuple_present:
            mock_checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

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
        else:
            assert input is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "approval_decision,expected_event_type,expected_message",
        [
            (UserDecision.APPROVE, FlowEventType.APPROVE, None),
            (UserDecision.REJECT, FlowEventType.REJECT, "test goal for rejection"),
            (None, FlowEventType.RESPONSE, "test goal for response"),
        ],
        ids=["approve_decision", "reject_decision", "user_response"],
    )
    async def test_resume_command_with_approval_decision(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,
        mock_tools_registry,  # pylint: disable=unused-argument
        approval_decision,
        expected_event_type,
        expected_message,
    ):
        """Test _resume_command returns correct FlowEvent based on approval decision."""
        # Create approval mock if decision is provided
        approval = None
        if approval_decision:
            approval = Mock(spec=contract_pb2.Approval)
            approval.WhichOneof.return_value = approval_decision

            # Mock the rejection attribute for REJECT decision
            if approval_decision == UserDecision.REJECT:
                mock_rejection = Mock()
                mock_rejection.message = expected_message
                approval.rejection = mock_rejection

        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id=f"test-workflow-{approval_decision or 'no-approval'}",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
                config=sample_flow_config,
                invocation_metadata=mock_invocation_metadata,
                approval=approval,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            goal = expected_message or "test goal"
            await flow.run(goal)

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input = kwargs.get("input")

            assert isinstance(input, Command)
            assert input.resume["event_type"] == expected_event_type

            if expected_message:
                assert input.resume["message"] == expected_message
            else:
                # For APPROVE events, message should not be present
                assert (
                    "message" not in input.resume or input.resume.get("message") is None
                )

    @pytest.mark.asyncio
    async def test_graph_input_with_additional_context(
        self,
        goal,
        mock_flow_metadata,
        user,
        mock_invocation_metadata,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,  # pylint: disable=unused-argument
        mock_tools_registry,  # pylint: disable=unused-argument
    ):
        """Test get_graph_input returns appropriate input based on status event."""
        with (
            self.mock_components(["AgentComponent", "EndComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            additional_context = AdditionalContext(
                category="file",
                content='{"contents": "hello", "file_name": "test.txt"}',
            )

            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
                config=sample_flow_config,
                invocation_metadata=mock_invocation_metadata,
                additional_context=[additional_context],
            )

            await flow.run(goal)

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]

            input = kwargs.get("input")

            assert "context" in input
            assert "inputs" in input["context"]
            assert input["context"]["inputs"][additional_context.category] == {
                "contents": "hello",
                "file_name": "test.txt",
            }

    @pytest.mark.asyncio
    async def test_flow_config_validation_duplicate_component_names(
        self,
        mock_flow_metadata,
        user,
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
            environment="ambient",
            version="v1",
        )

        with (
            self.mock_components(["AgentComponent", "AnotherComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.log_exception"
            ) as mock_log_exception,
        ):
            # Create flow instance
            flow_instance = Flow(
                workflow_id="duplicated-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
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
                "workflow_id": "duplicated-workflow-123",
                "source": "duo_workflow_service.agent_platform.v1.flows.base",
            }

    @pytest.mark.asyncio
    async def test_flow_orchestration_with_complex_config(
        self,
        mock_flow_metadata,
        user,
        mock_invocation_metadata,
        mock_state_graph,
        mock_tools_registry,
        mock_checkpointer,
    ):
        """Test Flow with complex configuration via run method to trigger _compile."""
        complex_config = FlowConfig(
            version="v1",
            environment="chat",
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
                "duo_workflow_service.agent_platform.v1.flows.base.load_component_class"
            ) as mock_load_class,
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.StateGraph",
                return_value=mock_state_graph,
            ),
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.Router"
            ) as mock_router_class,
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.EndComponent",
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
                user=user,
                config=complex_config,
                invocation_metadata=mock_invocation_metadata,
            )

            # Run the workflow to trigger _compile
            goal = "Complex workflow test"
            goal = "test goal with additional context"
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

    @pytest.mark.asyncio
    async def test_resume_command_with_invalid_approval_decision(
        self,
        mock_flow_metadata,
        mock_invocation_metadata,
        user,
        sample_flow_config,
        mock_state_graph,  # pylint: disable=unused-argument
        mock_checkpointer,
        mock_tools_registry,  # pylint: disable=unused-argument
    ):
        """Test _resume_command raises ValueError for invalid approval decision."""
        # Create approval mock with invalid decision
        approval = Mock(spec=contract_pb2.Approval)
        approval.WhichOneof.return_value = "invalid_decision"

        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.log_exception"
            ) as mock_log_exception,
        ):
            flow = Flow(
                workflow_id="test-workflow-invalid-approval",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
                config=sample_flow_config,
                invocation_metadata=mock_invocation_metadata,
                approval=approval,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            goal = "test goal"

            # The error should be logged via _handle_workflow_failure
            await flow.run(goal)

            # Verify that log_exception was called with the ValueError
            mock_log_exception.assert_called_once()
            mock_log_exception_call = mock_log_exception.call_args
            assert isinstance(mock_log_exception_call[0][0], ValueError)
            assert "Unexpected approval decision: invalid_decision" in str(
                mock_log_exception_call[0][0]
            )
            assert mock_log_exception_call[1]["extra"] == {
                "workflow_id": "test-workflow-invalid-approval",
                "source": "duo_workflow_service.agent_platform.v1.flows.base",
            }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "toolset,tool_name,want_toolset",
        [
            (None, "read_file", ["read_file"]),
            (["create_file_with_contents"], "read_file", ["create_file_with_contents"]),
            (["create_file_with_contents"], None, ["create_file_with_contents"]),
            (None, None, None),
        ],
        ids=[
            "tool_name_without_toolset",
            "tool_name_with_toolset",
            "toolset_without_tool_name",
            "no_toolset_or_tool_name",
        ],
    )
    async def test_flow_config_tool_name(
        self,
        toolset,
        tool_name,
        want_toolset,
        mock_flow_metadata,
        user,
        mock_invocation_metadata,
        mock_tools_registry,
        mock_checkpointer,  # pylint: disable=unused-argument
        mock_state_graph,  # pylint: disable=unused-argument
    ):
        """Test that tool_names can be used without explicit toolsets."""

        # Build component config based on parameters
        component_config = {
            "name": "tool_call",
            "type": "DeterministicStepComponent",
            "inputs": ["context:goal"],
        }

        if toolset is not None:
            component_config["toolset"] = toolset
        if tool_name is not None:
            component_config["tool_name"] = tool_name

        config = FlowConfig(
            flow={"entry_point": "tool_call"},
            components=[component_config],
            routers=[{"from": "tool_call", "to": "end"}],
            environment="chat-partial",
            version="v1",
        )

        # Mock the toolset return value
        mock_tools_registry.toolset.return_value = want_toolset or []

        with (
            self.mock_components(["DeterministicStepComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=CategoryEnum.WORKFLOW_CHAT,
                user=user,
                config=config,
                invocation_metadata=mock_invocation_metadata,
            )

            await flow.run("test goal")

            # Verify tools_registry.toolset was called with the expected arguments
            if want_toolset is not None:
                mock_tools_registry.toolset.assert_called_once_with(want_toolset)
            else:
                # When neither toolset nor tool_name is specified, toolset shouldn't be called
                mock_tools_registry.toolset.assert_not_called()

    def test_process_additional_context_empty_list(self, flow_instance):
        """Test _process_additional_context with empty list."""
        result = flow_instance._process_additional_context([])
        assert result == {}

    def test_process_additional_context(self, flow_instance):
        """Test _process_additional_context."""
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "file content", "file_name": "test.txt"}',
            ),
            AdditionalContext(
                category="snippet", content='{"snippet_str": "code snippet"}'
            ),
        ]

        result = flow_instance._process_additional_context(additional_context)

        expected = {
            "file": {"contents": "file content", "file_name": "test.txt"},
            "snippet": {"snippet_str": "code snippet"},
        }
        assert result == expected

    def test_process_additional_context_missing_required_field(self, flow_instance):
        """Test _process_additional_context with a missing schema field."""
        additional_context = [
            AdditionalContext(category="file", content='{"contents": "file content"}'),
        ]

        with pytest.raises(
            ValueError,
            match="input 'file' does not match specified schema: 'file_name' is a required property.*",
        ):
            flow_instance._process_additional_context(additional_context)

    def test_process_additional_context_no_schema(self, flow_instance):
        """Test _process_additional_context logs warning and skips unknown category."""
        additional_context = [
            AdditionalContext(category="unknown_category", content='{"os": "linux"}')
        ]

        # Unknown categories should be skipped with a warning, not raise an error
        result = flow_instance._process_additional_context(additional_context)

        # Should return empty dict since the unknown category was skipped
        assert result == {}

    def test_process_additional_context_invalid_json(self, flow_instance):
        """Test _process_additional_context raises error for invalid JSON."""
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"invalid": json}',
            )
        ]

        with pytest.raises(ValueError, match="Invalid JSON in input item.*"):
            flow_instance._process_additional_context(additional_context)

    def test_process_additional_context_schema_validation_error(
        self,
        flow_instance,
    ):
        """Test _process_additional_context raises error when JSON doesn't match schema."""
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"file_type": "file.txt"}',
            )
        ]

        with pytest.raises(
            ValueError,
            match=(
                r".*input 'file' does not match specified schema: "
                r"Additional properties are not allowed \('file_type' was unexpected\).*"
            ),
        ):
            flow_instance._process_additional_context(additional_context)

    def test_process_additional_context_executor_context_categories(
        self, flow_instance
    ):
        """Test _process_additional_context with executor context categories."""
        additional_context = [
            AdditionalContext(category="os_information", content="Linux Ubuntu 20.04"),
            AdditionalContext(category="shell_information", content="bash 5.0"),
            AdditionalContext(
                category="agent_user_environment",
                content='{"user": "testuser", "home": "/home/testuser"}',
            ),
        ]

        result = flow_instance._process_additional_context(additional_context)

        expected = {
            "os_information": "Linux Ubuntu 20.04",
            "shell_information": "bash 5.0",
            "agent_user_environment": '{"user": "testuser", "home": "/home/testuser"}',
        }
        assert result == expected
