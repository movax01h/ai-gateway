# pylint: disable=too-many-lines
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.types import Command
from structlog.testing import capture_logs

from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.v1.flows.base import (
    _ENVELOPE_DEFAULT_CONSTRAINT,
    _ENVELOPE_DEFAULT_VERSION,
    Flow,
    UserDecision,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    FlowConfigInput,
    FlowConfigMetadata,
)
from duo_workflow_service.agent_platform.v1.routers.router import Router
from duo_workflow_service.agent_platform.v1.state.base import FlowEventType
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.errors.typing import (
    EnvelopeVersionMismatchException,
    InvalidRequestException,
)
from duo_workflow_service.gitlab.gitlab_api import Namespace
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.workflows.abstract_workflow import TraceableException
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.events import GLReportingEventContext
from lib.internal_events.context import (
    merge_request_url_context,
    pipeline_source_context,
)
from lib.version import resolve_version


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

    @pytest.fixture(name="flow_type")
    def flow_type_fixture(self) -> GLReportingEventContext:
        return GLReportingEventContext.from_workflow_definition("chat")

    @pytest.fixture(name="mock_state_graph")
    def mock_state_graph_fixture(
        self,
        mock_fetch_workflow_and_container_data,  # pylint: disable=unused-argument  # fixture-on-fixture ordering dep
    ):
        # Create mock StateGraph and compiled graph
        mock_state_graph = Mock(spec=StateGraph)
        mock_compiled_graph = Mock()

        # Mock the compiled graph's astream method
        async def mock_astream(*_args, **_kwargs):
            yield ("values", {"status": "running"})
            yield ("updates", [{"step": "agent_processing"}])

        # mock_compiled_graph.astream = AsyncMock(side_effect=mock_astream)
        mock_compiled_graph.astream = Mock(return_value=mock_astream())
        mock_state_graph.compile.return_value = mock_compiled_graph
        ui_notifier = MagicMock()
        ui_notifier.send_event = AsyncMock()

        with (
            patch(
                "duo_workflow_service.workflows.abstract_workflow.empty_workflow_config",
                return_value={
                    "agent_privileges_names": [],
                    "pre_approved_agent_privileges_names": [],
                    "allow_agent_to_request_user": False,
                    "mcp_enabled": False,
                    "first_checkpoint": None,
                    "workflow_status": "",
                    "gitlab_host": "",
                },
            ),
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
    def flow_instance_fixture(  # pylint: disable=unused-argument  # fixture-on-fixture ordering deps
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_checkpointer,
        mock_tools_registry,
        mock_state_graph,
        flow_type: GLReportingEventContext,
    ):
        """Fixture providing a Flow instance with mocked dependencies."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
            )
            yield flow

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_event,goal,expected_type,first_checkpoint",
        [
            (WorkflowStatusEventEnum.START, "test goal", dict, None),
            ("unknown_event", "test goal", type(None), None),
            (
                WorkflowStatusEventEnum.RETRY,
                "test goal",
                type(None),
                {"checkpoint": "test"},
            ),
            (WorkflowStatusEventEnum.RETRY, "test goal", dict, None),
        ],
        ids=[
            "start_event",
            "unknown_event",
            "retry_with_checkpoint",
            "retry_without_checkpoint",
        ],
    )
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_graph_input(
        self,
        flow_instance: Flow,
        status_event,
        goal,
        expected_type,
        mock_checkpointer,
        mock_state_graph,
        project,
    ):
        """Test get_graph_input returns appropriate input based on status event."""
        mock_checkpointer.initial_status_event = status_event
        await flow_instance.run(goal)

        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]

        input = kwargs.get("input")
        if expected_type is dict:
            assert isinstance(input, expected_type)
            assert input["context"]["goal"] == goal
            assert input["context"]["project_id"] == project["id"]
            assert input["context"]["namespace"] is None
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
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_graph_input_chat_environment(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config_metadata,
        mock_checkpointer,
        mock_tools_registry,  # pylint: disable=unused-argument
        mock_state_graph,
        flow_type: GLReportingEventContext,
    ):
        """Test get_workflow_state uses USER message type and raw goal content for chat environment."""
        chat_config = FlowConfig(
            flow=sample_flow_config_metadata,
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                    "prompt_id": "test/prompt",
                    "toolset": ["read_file"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="chat",
            version="v1",
        )

        goal = "test chat goal"
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-chat",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=chat_config,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
            await flow.run(goal)

        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
        input_data = kwargs.get("input")

        assert isinstance(input_data, dict)
        assert len(input_data["ui_chat_log"]) == 1
        log_entry = input_data["ui_chat_log"][0]
        assert log_entry["message_type"] == MessageTypeEnum.USER
        assert log_entry["content"] == goal

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_graph_input_namespace_level(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_checkpointer,
        mock_tools_registry,  # pylint: disable=unused-argument
        mock_state_graph,
        flow_type: GLReportingEventContext,
    ):
        """Test get_workflow_state includes namespace and handles missing project at namespace level."""
        namespace = Namespace(
            id=42,
            description="Test namespace",
            name="test-group",
            web_url="https://gitlab.com/groups/test-group",
        )

        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
            patch(
                "duo_workflow_service.workflows.abstract_workflow.fetch_workflow_and_container_data",
                return_value=(None, namespace, mock_state_graph.compile.return_value),
            ),
        ):
            flow = Flow(
                workflow_id="test-workflow-namespace",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
            goal = "namespace-level goal"
            await flow.run(goal)

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input_data = kwargs.get("input")

            assert isinstance(input_data, dict)
            assert input_data["context"]["goal"] == goal
            assert input_data["context"]["project_id"] is None
            assert input_data["context"]["project_http_url_to_repo"] is None
            assert input_data["context"]["project_default_branch"] is None
            assert input_data["context"]["namespace"] == namespace

    def test_support_namespace_level_workflow(self, flow_instance):
        """Test that Flow supports namespace-level workflows."""
        assert flow_instance._support_namespace_level_workflow() is True

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_get_workflow_state_injects_gitlab_service_context(
        self,
        flow_instance: Flow,
        mock_checkpointer,
        mock_state_graph,
    ):
        """Test that get_workflow_state injects GitLabServiceContext variables into context."""
        mock_instance_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="17.0.0",
        )

        mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.GitLabServiceContext.get_current_instance_info",
            return_value=mock_instance_info,
        ):
            await flow_instance.run("test goal")

        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
        context = kwargs["input"]["context"]

        assert context["gitlab_instance_type"] == "GitLab.com (SaaS)"
        assert context["gitlab_instance_url"] == "https://gitlab.com"
        assert context["gitlab_instance_version"] == "17.0.0"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_get_workflow_state_falls_back_to_unknown_when_no_context(
        self,
        flow_instance: Flow,
        mock_checkpointer,
        mock_state_graph,
    ):
        """Test that get_workflow_state falls back to 'Unknown' when GitLabServiceContext is unavailable."""
        mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.GitLabServiceContext.get_current_instance_info",
            return_value=None,
        ):
            await flow_instance.run("test goal")

        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
        context = kwargs["input"]["context"]

        assert context["gitlab_instance_type"] == "Unknown"
        assert context["gitlab_instance_url"] == "Unknown"
        assert context["gitlab_instance_version"] == "Unknown"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "approval_decision,expected_event_type,expected_message",
        [
            (UserDecision.APPROVE, FlowEventType.APPROVE, None),
            (UserDecision.REJECT, FlowEventType.MODIFY, "test goal for rejection"),
            (UserDecision.REJECT, FlowEventType.REJECT, None),
            (None, FlowEventType.RESPONSE, "test goal for response"),
        ],
        ids=[
            "approve_decision",
            "reject_decision_with_message",
            "reject_decision_without_message",
            "user_response",
        ],
    )
    @pytest.mark.usefixtures("mock_tools_registry")
    async def test_resume_command_with_approval_decision(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,
        approval_decision,
        expected_event_type,
        expected_message,
        flow_type: GLReportingEventContext,
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
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
                approval=approval,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            goal = expected_message or "test goal"
            await flow.run(goal)

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input = kwargs.get("input")

            assert isinstance(input, Command)
            assert input.resume["event_type"] == expected_event_type  # type: ignore[index]

            if expected_message:
                assert input.resume["message"] == expected_message  # type: ignore[index]
            else:
                # For APPROVE events, message should not be present
                assert (
                    "message" not in input.resume  # type: ignore[operator]
                    or input.resume.get("message") is None  # type: ignore[union-attr]
                )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
    async def test_graph_input_resume_empty_goal_passes_empty_command_to_graph(
        self,
        flow_instance: Flow,
        mock_checkpointer,
        mock_state_graph,
    ):
        """Regression test for https://gitlab.com/gitlab-org/gitlab/-/work_items/602799.

        When the websocket drops while a turn is paused at INPUT_REQUIRED, the Duo CLI
        auto-retries by RESUMING the workflow with an empty goal (no approval). The
        flow layer must forward this as a ``Command(resume=FlowEvent(RESPONSE,
        message=""))`` to the graph — it is the recipient component (``FetchNode``)
        that detects the missing message and raises ``InvalidRequestException``.

        This test verifies that:
        - The graph IS invoked (the flow layer does not short-circuit).
        - The graph receives a ``Command`` with an empty-message RESPONSE event.
        - The ``InvalidRequestException`` raised by ``FetchNode`` propagates correctly
            so the server layer can map it to ``INVALID_ARGUMENT``.
        """
        mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME

        # Make the mocked graph raise InvalidRequestException to simulate FetchNode behaviour.
        bad_request_error = InvalidRequestException(
            "RESPONSE event must include a non-empty message. "
            "The workflow remains paused; please provide real user input to continue."
        )

        async def _raising_gen():
            """Async generator that immediately raises InvalidRequestException."""
            if True:  # pylint: disable=using-constant-test
                raise bad_request_error
            yield  # pragma: no cover — makes this an async generator

        mock_state_graph.compile.return_value.astream = Mock(
            return_value=_raising_gen()
        )

        await flow_instance.run("")

        # The graph MUST have been invoked — the flow layer passes the empty-message
        # command through to the graph rather than short-circuiting.
        mock_state_graph.compile.return_value.astream.assert_called_once()

        # The Command passed to the graph must carry an empty-message RESPONSE event.
        kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
        graph_input = kwargs.get("input")
        assert isinstance(graph_input, Command)
        assert graph_input.resume["event_type"] == FlowEventType.RESPONSE  # type: ignore[index]
        assert graph_input.resume.get("message") == ""  # type: ignore[union-attr]

        # The workflow's last_error must be a InvalidRequestException so the server layer
        # can map it to INVALID_ARGUMENT.
        assert isinstance(flow_instance.last_error, InvalidRequestException)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tools_registry")
    async def test_resume_command_refreshes_inputs_from_additional_context(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,
        flow_type: GLReportingEventContext,
    ):
        """Inputs are re-resolved on resume so per-turn additional context refreshes state.

        `context.inputs` is otherwise populated only once, at workflow START, so
        without this the resume Command would carry no input update and a flow
        input changed between turns (e.g. an operating mode toggle) would never
        reach the running graph.
        """
        additional_context = AdditionalContext(
            category="agent_user_environment",
            content='{"shell_name": "fish"}',
        )

        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-resume-inputs",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
                additional_context=[additional_context],
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            await flow.run("test goal")

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input = kwargs.get("input")

            assert isinstance(input, Command)
            assert input.update is not None
            assert input.update["context"]["inputs"]["agent_user_environment"] == (
                '{"shell_name": "fish"}'
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tools_registry")
    async def test_resume_command_without_additional_context_sends_no_update(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,
        flow_type: GLReportingEventContext,
    ):
        """With no additional context the resume Command carries no state update."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-resume-no-inputs",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            await flow.run("test goal")

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input = kwargs.get("input")

            assert isinstance(input, Command)
            assert input.update is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tools_registry")
    async def test_resume_command_refreshes_inputs_and_appends_rejection_log(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        mock_checkpointer,
        flow_type: GLReportingEventContext,
    ):
        """Both refreshed inputs and a rejection message land in the same update.

        When additional context is sent alongside a rejection approval that
        carries a message, the resume Command must merge both keys into its
        `update`: `context.inputs` (refreshed per-turn inputs) and `ui_chat_log`
        (the user's feedback entry). This guards against a future refactor
        dropping either key from the combined `state_update`.
        """
        additional_context = AdditionalContext(
            category="agent_user_environment",
            content='{"shell_name": "fish"}',
        )
        rejection_message = "please redo this"
        approval = Mock(spec=contract_pb2.Approval)
        approval.WhichOneof.return_value = UserDecision.REJECT
        mock_rejection = Mock()
        mock_rejection.message = rejection_message
        approval.rejection = mock_rejection

        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-workflow-resume-inputs-and-rejection",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
                approval=approval,
                additional_context=[additional_context],
            )

            mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.RESUME
            await flow.run(rejection_message)

            kwargs = mock_state_graph.compile.return_value.astream.call_args[1]
            input = kwargs.get("input")

            assert isinstance(input, Command)
            assert input.update is not None
            assert input.update["context"]["inputs"]["agent_user_environment"] == (
                '{"shell_name": "fish"}'
            )
            ui_chat_log = input.update["ui_chat_log"]
            assert len(ui_chat_log) == 1
            assert ui_chat_log[0]["content"] == rejection_message
            assert ui_chat_log[0]["message_type"] == MessageTypeEnum.USER
            assert input.resume["event_type"] == FlowEventType.MODIFY  # type: ignore[index]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_checkpointer", "mock_tools_registry")
    async def test_graph_input_with_additional_context(
        self,
        goal,
        mock_flow_metadata,
        user,
        sample_flow_config,
        mock_state_graph,
        flow_type: GLReportingEventContext,
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
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
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
    @pytest.mark.usefixtures("mock_tools_registry", "mock_checkpointer")
    async def test_flow_config_validation_duplicate_component_names(
        self,
        mock_flow_metadata,
        user,
        mock_state_graph,
        flow_type: GLReportingEventContext,
    ):
        """Test that duplicate component names are detected during compilation."""
        # Create config with duplicate component names
        duplicate_config = FlowConfig(
            flow={"entry_point": "agent"},  # type: ignore[arg-type]
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
                workflow_type=flow_type,
                user=user,
                config=duplicate_config,
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
        mock_state_graph,
        mock_tools_registry,
        mock_checkpointer,
        flow_type: GLReportingEventContext,
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
            flow={"entry_point": "agent"},  # type: ignore[arg-type]
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
                workflow_type=flow_type,
                user=user,
                config=complex_config,
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
                ["read_file", "edit_file"], tool_options={}
            )
            mock_agent_class.assert_called_once()
            agent_call_args = mock_agent_class.call_args[1]
            assert agent_call_args["name"] == "agent"
            assert agent_call_args["flow_id"] == "complex-workflow-123"
            assert agent_call_args["flow_type"] == flow_type
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
            assert human_input_call_args["flow_type"] == flow_type

            # EndComponent component
            mock_end_component_class.assert_called_once()
            end_component_call_args = mock_end_component_class.call_args[1]
            assert end_component_call_args["name"] == "end"
            assert end_component_call_args["flow_id"] == "complex-workflow-123"
            assert end_component_call_args["flow_type"] == flow_type
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
    @pytest.mark.usefixtures("mock_state_graph", "mock_tools_registry")
    async def test_resume_command_with_invalid_approval_decision(
        self,
        mock_flow_metadata,
        user,
        sample_flow_config,
        flow_type: GLReportingEventContext,
        mock_checkpointer,
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
                workflow_type=flow_type,
                user=user,
                config=sample_flow_config,
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
    @pytest.mark.usefixtures("mock_checkpointer", "mock_state_graph")
    async def test_flow_config_tool_name(
        self,
        toolset,
        tool_name,
        want_toolset,
        mock_flow_metadata,
        user,
        mock_tools_registry,
        flow_type: GLReportingEventContext,
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
            flow={"entry_point": "tool_call"},  # type: ignore[arg-type]
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
                workflow_type=flow_type,
                user=user,
                config=config,
            )

            await flow.run("test goal")

            # Verify tools_registry.toolset was called with the expected arguments
            if want_toolset is not None:
                if toolset is not None:
                    # When toolset is specified in config, _parse_toolset is used which passes tool_options
                    mock_tools_registry.toolset.assert_called_once_with(
                        want_toolset, tool_options={}
                    )
                else:
                    # When only tool_name is specified, toolset is called directly without tool_options
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

    def test_process_additional_context_extra_property_is_tolerated(
        self,
        flow_instance,
    ):
        """Test _process_additional_context ignores unknown properties instead of raising.

        Adding a new field to an envelope (e.g. service_account_name) must not break custom flows that have not yet
        declared the field in their input_schema.
        """
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "hello", "file_name": "test.txt", "file_type": "text"}',
            )
        ]

        result = flow_instance._process_additional_context(additional_context)

        # The extra field is accepted; the declared fields are present
        assert result["file"]["contents"] == "hello"
        assert result["file"]["file_name"] == "test.txt"
        assert result["file"]["file_type"] == "text"

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

    def test_process_additional_context_agent_skills_routing(self, flow_instance):
        """Test that user_rule with agent-skills-instructions id routes to workspace_agent_skills."""
        additional_context = [
            AdditionalContext(
                category="user_rule",
                id="agents-md-user-instructions",
                content="# AGENTS.md content",
            ),
            AdditionalContext(
                category="user_rule",
                id="agent-skills-instructions",
                content="<available_skills>...</available_skills>",
            ),
        ]

        result = flow_instance._process_additional_context(additional_context)

        assert result["user_rule"] == "# AGENTS.md content"
        assert (
            result["workspace_agent_skills"]
            == "<available_skills>...</available_skills>"
        )
        assert "agent-skills-instructions" not in result

    @pytest.mark.usefixtures("mock_state_graph")
    def test_build_routers_always_passes_tracking_params_for_conditional_routers(
        self,
        mock_flow_metadata,
        user,
        flow_type: GLReportingEventContext,
    ):
        """Test _build_routers always passes instrumentation params for conditional routers."""
        config = FlowConfig(
            version="v1",
            environment="ambient",
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": "status",
                        "routes": {"Execution": "end"},
                    },
                },
            ],
            flow=FlowConfigMetadata(entry_point="agent"),
        )

        components = {
            "agent": self.mock_component("agent"),
            "end": self.mock_component("end"),
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.Router"
        ) as mock_router_class:
            mock_router_class.return_value = Mock(spec=Router)

            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=config,
            )

            flow._build_routers(components, Mock(spec=StateGraph))

            call_kwargs = mock_router_class.call_args[1]
            assert call_kwargs["flow_id"] == "test-workflow-123"
            assert call_kwargs["flow_type"] == flow_type
            assert "internal_event_client" in call_kwargs

    @pytest.mark.usefixtures("mock_state_graph")
    def test_build_routers_accepts_mapping_condition_input(
        self,
        mock_flow_metadata,
        user,
        flow_type: GLReportingEventContext,
    ):
        """A condition input may be a mapping ({from: ..., optional: true})."""
        mapping_input = {
            "from": "context:inputs.agent_platform_trigger_context.event_type",
            "optional": True,
        }
        config = FlowConfig(
            version="v1",
            environment="ambient",
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": mapping_input,
                        "routes": {"mention": "end"},
                    },
                },
            ],
            flow=FlowConfigMetadata(entry_point="agent"),
        )

        components = {
            "agent": self.mock_component("agent"),
            "end": self.mock_component("end"),
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.Router"
        ) as mock_router_class:
            mock_router_class.return_value = Mock(spec=Router)

            flow = Flow(
                workflow_id="test-workflow-123",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=config,
            )

            flow._build_routers(components, Mock(spec=StateGraph))

            assert mock_router_class.call_args[1]["input"] == mapping_input

    @pytest.mark.usefixtures("mock_state_graph")
    def test_build_routers_rejects_non_string_non_mapping_condition_input(
        self,
        mock_flow_metadata,
        user,
        flow_type: GLReportingEventContext,
    ):
        """Anything that is not a string or a mapping is still rejected."""
        config = FlowConfig(
            version="v1",
            environment="ambient",
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": ["status"],
                        "routes": {"Execution": "end"},
                    },
                },
            ],
            flow=FlowConfigMetadata(entry_point="agent"),
        )

        components = {
            "agent": self.mock_component("agent"),
            "end": self.mock_component("end"),
        }

        flow = Flow(
            workflow_id="test-workflow-123",
            workflow_metadata=mock_flow_metadata,
            workflow_type=flow_type,
            user=user,
            config=config,
        )

        with pytest.raises(
            ValueError, match="Router input must be a string or a mapping"
        ):
            flow._build_routers(components, Mock(spec=StateGraph))

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_appends_error_log(self, flow_instance):
        """Existing ui_chat_log entries are preserved."""
        existing_log = {
            "message_type": MessageTypeEnum.TOOL,
            "content": "Previous log entry",
        }
        notifier = MagicMock()
        notifier.ui_chat_log = [existing_log]
        flow_instance.checkpoint_notifier = notifier

        graph = AsyncMock()
        config = {"configurable": {"thread_id": "test-workflow-123"}}

        await flow_instance._handle_workflow_failure(
            RuntimeError("boom"), graph, config
        )

        state = graph.aupdate_state.call_args[0][1]
        ui_chat_log = state["ui_chat_log"].value
        assert len(ui_chat_log) == 2
        assert ui_chat_log[0] == existing_log

        error_entry = ui_chat_log[1]
        assert error_entry["message_type"] == MessageTypeEnum.AGENT
        assert error_entry["status"] == ToolStatus.FAILURE
        assert error_entry["message_id"].startswith("error-")

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_does_not_leak_details(self, flow_instance):
        """Internal exception text must not appear in the UI log."""
        notifier = MagicMock()
        notifier.ui_chat_log = []
        flow_instance.checkpoint_notifier = notifier

        graph = AsyncMock()

        await flow_instance._handle_workflow_failure(
            RuntimeError("secret internal detail"), graph, {}
        )

        state = graph.aupdate_state.call_args[0][1]
        content = state["ui_chat_log"].value[0]["content"]
        assert "secret internal detail" not in content
        assert "error" in content.lower()

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_no_compiled_graph(self, flow_instance):
        """When compiled_graph is None, only log_exception runs."""
        await flow_instance._handle_workflow_failure(
            RuntimeError("early failure"), None, {}
        )

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_aupdate_state_fails(self, flow_instance):
        """If aupdate_state raises, the error is caught and a warning is logged."""
        notifier = MagicMock()
        notifier.ui_chat_log = []
        flow_instance.checkpoint_notifier = notifier

        graph = AsyncMock()
        graph.aupdate_state.side_effect = Exception("closed")

        flow_instance.log = MagicMock()

        await flow_instance._handle_workflow_failure(RuntimeError("boom"), graph, {})

        flow_instance.log.warning.assert_called_once_with(
            "Failed to persist error ui_chat_log to checkpoint",
            workflow_id="test-workflow-123",
            exc_info=graph.aupdate_state.side_effect,
        )

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_notifiable_agent_exception_surfaces_ui_message(
        self, flow_instance
    ):
        """NotifiableAgentException surfaces ui_message in the UI chat log."""
        notifier = MagicMock()
        notifier.ui_chat_log = []
        flow_instance.checkpoint_notifier = notifier

        graph = AsyncMock()
        secret = "internal-token-shh"
        error = NotifiableAgentException(
            "Safe user-facing message", internal_detail=secret
        )

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.log_exception"
        ) as mock_log_exception:
            await flow_instance._handle_workflow_failure(error, graph, {})

        mock_log_exception.assert_called_once()
        log_extra = mock_log_exception.call_args.kwargs["extra"]
        assert log_extra["workflow_id"] == "test-workflow-123"
        assert log_extra["internal_detail"] == secret

        state = graph.aupdate_state.call_args[0][1]
        ui_chat_log = state["ui_chat_log"].value
        assert len(ui_chat_log) == 1
        assert ui_chat_log[0]["content"] == "Safe user-facing message"
        assert secret not in ui_chat_log[0]["content"]
        assert ui_chat_log[0]["status"] == ToolStatus.FAILURE

    @pytest.mark.asyncio
    async def test_handle_workflow_failure_notifiable_agent_exception_no_internal_detail(
        self, flow_instance
    ):
        """NotifiableAgentException without internal_detail is logged without that key."""
        notifier = MagicMock()
        notifier.ui_chat_log = []
        flow_instance.checkpoint_notifier = notifier

        graph = AsyncMock()
        error = NotifiableAgentException("Safe message only")

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.log_exception"
        ) as mock_log_exception:
            await flow_instance._handle_workflow_failure(error, graph, {})

        log_extra = mock_log_exception.call_args.kwargs["extra"]
        assert "internal_detail" not in log_extra

        state = graph.aupdate_state.call_args[0][1]
        assert state["ui_chat_log"].value[0]["content"] == "Safe message only"

    @pytest.mark.asyncio
    async def test_handle_compile_and_run_exception_converts_graph_recursion_error(
        self, flow_instance
    ):
        """GraphRecursionError is converted to NotifiableAgentException before propagating."""
        notifier = MagicMock()
        notifier.ui_chat_log = []
        notifier.send_event = AsyncMock()
        flow_instance.checkpoint_notifier = notifier

        compiled_graph = AsyncMock()

        with (
            patch("duo_workflow_service.agent_platform.v1.flows.base.log_exception"),
            pytest.raises(TraceableException) as exc_info,
        ):
            await flow_instance._handle_compile_and_run_exception(
                GraphRecursionError("Recursion limit of 300 reached"),
                compiled_graph,
                {},
            )

        wrapped = exc_info.value.original_exception
        assert isinstance(wrapped, NotifiableAgentException)
        assert "300" in wrapped.internal_detail
        assert (
            wrapped.ui_message
            == "The workflow reached its maximum step limit and could not complete. "
            "Please try again with a more focused goal, or break the task into smaller steps."
        )

    # ---------------------------------------------------------------------------
    # Envelope version validation tests
    # ---------------------------------------------------------------------------

    @pytest.fixture(name="versioned_flow_config")
    def versioned_flow_config_fixture(self):
        """Flow config with a version_constraint on agent_platform_standard_context."""
        return FlowConfig(
            flow=FlowConfigMetadata(
                entry_point="agent",
                inputs=[
                    FlowConfigInput(
                        category="agent_platform_standard_context",
                        version_constraint="^1.0.0",
                        input_schema={
                            "primary_branch": {"type": "string"},
                        },
                    ),
                ],
            ),
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                    "prompt_id": "test/prompt",
                    "toolset": ["read_file"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="ambient",
            version="v1",
        )

    @pytest.fixture(name="versioned_flow_instance")
    def versioned_flow_instance_fixture(
        self,
        mock_flow_metadata,
        user,
        versioned_flow_config,
        mock_checkpointer,  # pylint: disable=unused-argument
        mock_tools_registry,  # pylint: disable=unused-argument
        mock_state_graph,  # pylint: disable=unused-argument
        flow_type: GLReportingEventContext,
    ):
        """Flow instance using the versioned config."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-versioned-workflow",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=versioned_flow_config,
            )
            yield flow

    @pytest.fixture(name="higher_constraint_flow_config")
    def higher_constraint_flow_config_fixture(self):
        """Flow config with a ^1.1.0 constraint — implicit 1.0.0 envelopes must fail."""
        return FlowConfig(
            flow=FlowConfigMetadata(
                entry_point="agent",
                inputs=[
                    FlowConfigInput(
                        category="agent_platform_standard_context",
                        version_constraint="^1.1.0",
                        input_schema={
                            "primary_branch": {"type": "string"},
                        },
                    ),
                ],
            ),
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                    "prompt_id": "test/prompt",
                    "toolset": ["read_file"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            environment="ambient",
            version="v1",
        )

    @pytest.fixture(name="higher_constraint_flow_instance")
    def higher_constraint_flow_instance_fixture(
        self,
        mock_flow_metadata,
        user,
        higher_constraint_flow_config,
        mock_checkpointer,  # pylint: disable=unused-argument
        mock_tools_registry,  # pylint: disable=unused-argument
        mock_state_graph,  # pylint: disable=unused-argument
        flow_type: GLReportingEventContext,
    ):
        """Flow instance with ^1.1.0 version constraint."""
        with (
            self.mock_components(["AgentComponent"]),
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            flow = Flow(
                workflow_id="test-higher-constraint-workflow",
                workflow_metadata=mock_flow_metadata,
                workflow_type=flow_type,
                user=user,
                config=higher_constraint_flow_config,
            )
            yield flow

    def test_process_additional_context_version_satisfied(
        self, versioned_flow_instance
    ):
        """Envelope with a version that satisfies the constraint is accepted."""
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "1.1.0"},
            )
        ]
        result = versioned_flow_instance._process_additional_context(additional_context)
        assert result["agent_platform_standard_context"]["primary_branch"] == "main"

    def test_process_additional_context_version_not_satisfied_single_envelope(
        self, versioned_flow_instance
    ):
        """A single envelope whose version does NOT satisfy the constraint raises NotifiableAgentException.

        The user-facing ``NotifiableAgentException`` carries a safe ``ui_message`` and
        chains an ``EnvelopeVersionMismatchException`` as its ``__cause__`` so the server
        can still map it to a ``FAILED_PRECONDITION`` gRPC status.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "2.0.0"},
            )
        ]
        with pytest.raises(NotifiableAgentException) as exc_info:
            versioned_flow_instance._process_additional_context(additional_context)
        assert isinstance(exc_info.value.__cause__, EnvelopeVersionMismatchException)
        assert "2.0.0" in str(exc_info.value.__cause__)

    def test_process_additional_context_no_version_field_treated_as_1_0_0(
        self, versioned_flow_instance
    ):
        """Envelope without a version field is treated as 1.0.0 (backwards compat).

        CONTRACT — DO NOT BREAK. Older GitLab instances and Custom Flows send envelopes
        with no ``version`` field. We must keep treating those as ``1.0.0`` so existing
        senders continue to work. Changing this behaviour is a backward-incompatible
        break for any client that predates envelope versioning.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
            )
        ]
        # ^1.0.0 constraint should accept implicit 1.0.0
        result = versioned_flow_instance._process_additional_context(additional_context)
        assert result["agent_platform_standard_context"]["primary_branch"] == "main"

    def test_process_additional_context_no_constraint_uses_implicit_constraint(
        self, flow_instance
    ):
        """When no version_constraint is declared, implicit ^1.0.0 constraint is used.

        CONTRACT — DO NOT BREAK. Existing Custom Flows were authored before
        ``version_constraint`` existed and declare none. We must keep applying the
        implicit ``^1.0.0`` constraint for those flows so they continue to accept
        ``1.x.x`` envelopes. Changing this default is a backward-incompatible break for
        every flow that predates envelope versioning.
        """
        # flow_instance uses sample_flow_config which has no version_constraint
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "data", "file_name": "test.txt"}',
            )
        ]
        # No version field → implicit 1.0.0, satisfies ^1.0.0, should not raise.
        result = flow_instance._process_additional_context(additional_context)
        assert result["file"]["contents"] == "data"

    def test_process_additional_context_incompatible_envelope_skipped_when_compatible_present(
        self, versioned_flow_instance
    ):
        """When multiple envelopes are present, the highest compatible version wins.

        With constraint ``^1.0.0``, version ``2.0.0`` is incompatible and ``1.1.0``
        is compatible.  ``resolve_version`` selects ``1.1.0`` as the best match, so
        the result comes from the ``1.1.0`` envelope (``primary_branch == "main"``).
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "feature"}',
                metadata={"version": "2.0.0"},
            ),
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "1.1.0"},
            ),
        ]
        result = versioned_flow_instance._process_additional_context(additional_context)
        assert result["agent_platform_standard_context"]["primary_branch"] == "main"

    def test_process_additional_context_all_envelopes_incompatible_raises(
        self, versioned_flow_instance
    ):
        """When all envelopes are incompatible with the constraint, NotifiableAgentException is raised."""
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "2.0.0"},
            ),
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "3.0.0"},
            ),
        ]
        with pytest.raises(NotifiableAgentException) as exc_info:
            versioned_flow_instance._process_additional_context(additional_context)
        assert isinstance(exc_info.value.__cause__, EnvelopeVersionMismatchException)

    def test_process_additional_context_highest_compatible_version_selected(
        self, versioned_flow_instance
    ):
        """When multiple envelopes all satisfy the constraint, the highest version wins.

        Both ``1.0.0`` and ``1.1.0`` satisfy ``^1.0.0``.  ``resolve_version`` returns
        ``1.1.0`` as the best match, so the result comes from the ``1.1.0`` envelope.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "old"}',
                metadata={"version": "1.0.0"},
            ),
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "new"}',
                metadata={"version": "1.1.0"},
            ),
        ]
        result = versioned_flow_instance._process_additional_context(additional_context)
        assert result["agent_platform_standard_context"]["primary_branch"] == "new"

    def test_process_additional_context_implicit_version_fails_higher_constraint(
        self, higher_constraint_flow_instance
    ):
        """An envelope without a version field is treated as 1.0.0, which fails ^1.1.0.

        The implicit baseline "1.0.0" must appear in the ``__cause__`` detail so
        operators can diagnose the mismatch without inspecting the envelope payload.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
            )
        ]
        with pytest.raises(NotifiableAgentException) as exc_info:
            higher_constraint_flow_instance._process_additional_context(
                additional_context
            )
        assert isinstance(exc_info.value.__cause__, EnvelopeVersionMismatchException)
        assert "1.0.0" in str(exc_info.value.__cause__)

    @pytest.mark.parametrize(
        "bad_version",
        ["not_a_version", "abc.def.ghi", "!!invalid!!"],
    )
    def test_process_additional_context_malformed_envelope_version_raises(
        self, versioned_flow_instance, bad_version
    ):
        """A malformed string version field is passed to resolve_version which skips it.

        With no parseable candidates, ``resolve_version`` raises ``ValueError`` and
        the method surfaces a ``NotifiableAgentException`` chained from
        ``EnvelopeVersionMismatchException``.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": bad_version},
            )
        ]
        with pytest.raises(NotifiableAgentException) as exc_info:
            versioned_flow_instance._process_additional_context(additional_context)
        assert isinstance(exc_info.value.__cause__, EnvelopeVersionMismatchException)

    def test_process_additional_context_non_string_version_treated_as_implicit(
        self, versioned_flow_instance
    ):
        """A non-string JSON version field (e.g. a number) falls back to implicit 1.0.0.

        ``str(1.0)`` produces ``"1.0"`` which semver cannot parse; the fix treats
        non-string values as absent so they default to ``"1.0.0"`` instead of
        triggering a spurious version-mismatch error.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                # Integer in metadata — not a string version; should be treated as 1.0.0
                metadata={"version": 1},
            )
        ]
        result = versioned_flow_instance._process_additional_context(additional_context)
        assert result["agent_platform_standard_context"]["primary_branch"] == "main"

    def test_resolve_envelope_content_raises_when_no_constraint_and_empty_content(
        self, flow_instance
    ):
        """Envelope with None content raises ValueError in the unconstrained branch."""
        additional_context = [AdditionalContext(category="file", content=None)]
        with pytest.raises(
            ValueError, match="content must be specified for input 'file'"
        ):
            flow_instance._process_additional_context(additional_context)

    def test_resolve_envelope_content_raises_when_constraint_and_empty_content(
        self, versioned_flow_instance
    ):
        """Envelope with None content raises ValueError in the versioned branch."""
        additional_context = [
            AdditionalContext(category="agent_platform_standard_context", content=None)
        ]
        with pytest.raises(
            ValueError,
            match="content must be specified for input 'agent_platform_standard_context'",
        ):
            versioned_flow_instance._process_additional_context(additional_context)

    # ---------------------------------------------------------------------------
    # Default-fallback consistency contract
    # ---------------------------------------------------------------------------

    def test_default_version_satisfies_default_constraint(self):
        """The default version must satisfy the default constraint.

        CONTRACT — DO NOT BREAK. Unversioned envelopes fall back to
        ``_ENVELOPE_DEFAULT_VERSION`` and flows without a declared constraint fall back
        to ``_ENVELOPE_DEFAULT_CONSTRAINT``. Both defaults are applied independently, so
        if someone bumps the default version (e.g. to ``2.0.0``) without also updating
        the default constraint, every older Custom Flow relying on the implicit
        constraint would suddenly reject the implicit envelope version. This test guards
        that invariant: the default version must always be resolvable against the default
        constraint.
        """
        resolved = resolve_version(
            [_ENVELOPE_DEFAULT_VERSION], _ENVELOPE_DEFAULT_CONSTRAINT
        )
        assert resolved == _ENVELOPE_DEFAULT_VERSION

    # ---------------------------------------------------------------------------
    # Warning-log tests for default fallbacks
    # ---------------------------------------------------------------------------

    def test_warning_emitted_when_default_constraint_used(self, flow_instance):
        """A warning is logged when no version_constraint is declared for an input category.

        ``flow_instance`` uses ``sample_flow_config`` which has no ``version_constraint``
        on its inputs, so ``_ENVELOPE_DEFAULT_CONSTRAINT`` is substituted and a warning
        must be emitted.
        """
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "data", "file_name": "test.txt"}',
                metadata={"version": "1.0.0"},
            )
        ]
        with capture_logs() as cap_logs:
            flow_instance._process_additional_context(additional_context)

        warning_events = [
            log
            for log in cap_logs
            if log.get("log_level") == "warning"
            and "version_constraint" in log.get("event", "")
        ]
        assert warning_events, (
            f"Expected a warning about missing version_constraint, got: {cap_logs}"
        )
        assert warning_events[0]["category"] == "file"
        assert "default_constraint" in warning_events[0]

    def test_no_warning_when_explicit_constraint_provided(
        self, versioned_flow_instance
    ):
        """No default-constraint warning is emitted when version_constraint is explicitly set.

        ``versioned_flow_instance`` uses a config with ``version_constraint="^1.0.0"``,
        so the fallback path must NOT be taken.
        """
        additional_context = [
            AdditionalContext(
                category="agent_platform_standard_context",
                content='{"primary_branch": "main"}',
                metadata={"version": "1.0.0"},
            )
        ]
        with capture_logs() as cap_logs:
            versioned_flow_instance._process_additional_context(additional_context)

        constraint_warnings = [
            log
            for log in cap_logs
            if log.get("log_level") == "warning"
            and "version_constraint" in log.get("event", "")
        ]
        assert constraint_warnings == [], (
            f"Unexpected default-constraint warning(s): {constraint_warnings}"
        )

    def test_warning_emitted_when_default_version_used(self, flow_instance):
        """A warning is logged when an envelope payload does not declare a version field.

        The envelope has no ``metadata`` (and therefore no ``version``), so
        ``_ENVELOPE_DEFAULT_VERSION`` is substituted and a warning must be emitted.
        """
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "data", "file_name": "test.txt"}',
                # No metadata / version field → default version substituted
            )
        ]
        with capture_logs() as cap_logs:
            flow_instance._process_additional_context(additional_context)

        version_warnings = [
            log
            for log in cap_logs
            if log.get("log_level") == "warning"
            and "default version" in log.get("event", "")
        ]
        assert version_warnings, (
            f"Expected a warning about missing envelope version, got: {cap_logs}"
        )
        assert version_warnings[0]["category"] == "file"
        assert "default_version" in version_warnings[0]

    def test_no_warning_when_explicit_version_provided(self, flow_instance):
        """No default-version warning is emitted when the envelope declares an explicit version.

        The envelope carries ``metadata={"version": "1.0.0"}``, so the fallback
        path must NOT be taken.
        """
        additional_context = [
            AdditionalContext(
                category="file",
                content='{"contents": "data", "file_name": "test.txt"}',
                metadata={"version": "1.0.0"},
            )
        ]
        with capture_logs() as cap_logs:
            flow_instance._process_additional_context(additional_context)

        version_warnings = [
            log
            for log in cap_logs
            if log.get("log_level") == "warning"
            and "default version" in log.get("event", "")
        ]
        assert version_warnings == [], (
            f"Unexpected default-version warning(s): {version_warnings}"
        )


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestSetTrackingContextFromAdditionalContext:
    """Tests for _set_tracking_context_from_additional_context called during Flow.__init__."""

    def _create_flow(self, user, additional_context=None):
        config = FlowConfig(
            flow=FlowConfigMetadata(
                entry_point="agent",
                inputs=[],
            ),
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
        flow_type = GLReportingEventContext.from_workflow_definition("chat")
        with (
            patch(
                "duo_workflow_service.agent_platform.v1.flows.base.load_component_class"
            ) as mock_load_class,
            patch("duo_workflow_service.agent_platform.v1.flows.base.Router"),
        ):
            mock_component = MagicMock(spec=BaseComponent)
            mock_component.__entry_hook__.return_value = "agent_entry_node"
            mock_load_class.return_value = MagicMock(return_value=mock_component)
            return Flow(
                workflow_id="test-id",
                workflow_metadata={
                    "git_url": "https://gitlab.com/test/project",
                    "git_sha": "abc123",
                    "extended_logging": False,
                },
                workflow_type=flow_type,
                user=user,
                config=config,
                additional_context=additional_context,
            )

    def test_sets_merge_request_url_and_pipeline_source(self, user):
        mr_token = merge_request_url_context.set(None)
        ps_token = pipeline_source_context.set(None)
        try:
            self._create_flow(
                user,
                additional_context=[
                    AdditionalContext(
                        category="merge_request",
                        content='{"url": "https://gitlab.com/project/-/merge_requests/1"}',
                    ),
                    AdditionalContext(
                        category="pipeline",
                        content='{"source_branch": "main", "source": "merge_request_event"}',
                    ),
                ],
            )
            assert (
                merge_request_url_context.get()
                == "https://gitlab.com/project/-/merge_requests/1"
            )
            assert pipeline_source_context.get() == "merge_request_event"
        finally:
            merge_request_url_context.reset(mr_token)
            pipeline_source_context.reset(ps_token)

    def test_handles_invalid_merge_request_json(self, user):
        mr_token = merge_request_url_context.set(None)
        try:
            self._create_flow(
                user,
                additional_context=[
                    AdditionalContext(
                        category="merge_request",
                        content="not valid json",
                    ),
                ],
            )
            assert merge_request_url_context.get() is None
        finally:
            merge_request_url_context.reset(mr_token)

    def test_handles_invalid_pipeline_json(self, user):
        ps_token = pipeline_source_context.set(None)
        try:
            self._create_flow(
                user,
                additional_context=[
                    AdditionalContext(
                        category="pipeline",
                        content="not valid json",
                    ),
                ],
            )
            assert pipeline_source_context.get() is None
        finally:
            pipeline_source_context.reset(ps_token)

    def test_no_context_set_without_additional_context(self, user):
        mr_token = merge_request_url_context.set(None)
        ps_token = pipeline_source_context.set(None)
        try:
            self._create_flow(user)
            assert merge_request_url_context.get() is None
            assert pipeline_source_context.get() is None
        finally:
            merge_request_url_context.reset(mr_token)
            pipeline_source_context.reset(ps_token)
