from unittest.mock import Mock, call, patch

import pytest
from langgraph.graph import StateGraph

from duo_workflow_service.agent_platform.v1.components.human_input.component import (
    ChatHumanInputComponent,
    HumanInputComponent,
    human_input_component_factory,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.v1.state import FlowState, IOKey
from duo_workflow_service.agent_platform.v1.ui_log.base import UIHistory
from lib.events import GLReportingEventContext


class TestHumanInputComponent:
    """Test suite for HumanInputComponent."""

    @pytest.fixture(name="flow_type")
    def flow_type_fixture(self) -> GLReportingEventContext:
        return GLReportingEventContext.from_workflow_definition("chat")

    @pytest.fixture
    def human_input_component(self, user, flow_type: GLReportingEventContext):
        """Create a HumanInputComponent instance for testing."""
        return HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            message_template="Test message template",
            ui_log_events=[
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
            ],
        )

    def test_iokey_template_replacement(self, human_input_component):
        """Test that IOKeyTemplate correctly replaces SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE."""
        outputs = human_input_component.outputs

        assert len(outputs) == 4

        # First output should be conversation_history
        conversation_output = outputs[0]
        assert isinstance(conversation_output, IOKey)
        assert conversation_output.target == "conversation_history"
        assert conversation_output.subkeys == ["awesome_agent"]

        # Second output should be approval context
        approval_output = outputs[1]
        assert isinstance(approval_output, IOKey)
        assert approval_output.target == "context"
        assert approval_output.subkeys == ["test_human_input", "approval"]

        # Third output should be status
        status_output = outputs[2]
        assert isinstance(status_output, IOKey)
        assert status_output.target == "status"
        assert status_output.subkeys is None

        # Fourth output should be the cancelled-turn consume-once cleanup
        cancelled_turn_output = outputs[3]
        assert isinstance(cancelled_turn_output, IOKey)
        assert cancelled_turn_output.target == "context"
        assert cancelled_turn_output.subkeys == ["inputs", "cancelled_turn"]

    def test_entry_hook_returns_correct_node_name(self, human_input_component):
        """Test that __entry_hook__ returns the component's request node name."""
        assert human_input_component.__entry_hook__() == "test_human_input#request"

    def test_attach_method_creates_nodes(self, human_input_component):
        """Test that attach method creates proper nodes in the graph."""

        graph = StateGraph(FlowState)
        router = Mock()
        router.route = Mock(return_value="next_node")

        with (
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):
            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            request_instance.run = Mock()
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            fetch_instance.run = Mock()
            mock_fetch_node.return_value = fetch_instance

            # Mock graph methods to verify calls
            graph.add_node = Mock()
            graph.add_edge = Mock()
            graph.add_conditional_edges = Mock()

            human_input_component.attach(graph, router)

            # Verify nodes were created with correct arguments
            mock_request_node.assert_called_once_with(
                name="test_human_input#request",
                component_name="test_human_input",
                message_template="Test message template",
                inputs=human_input_component.inputs,
                request_type="approval",
                ui_history=mock_request_node.call_args[1]["ui_history"],
                status_key=human_input_component._status_output,
            )

            mock_fetch_node.assert_called_once_with(
                name="test_human_input#fetch",
                component_name="test_human_input",
                output=human_input_component._approval_output,
                conversation_history_key=human_input_component._conversation_history_input,
                ui_history=mock_fetch_node.call_args[1]["ui_history"],
                status_key=human_input_component._status_output,
                cancelled_turn_key=human_input_component._cancelled_turn_input,
            )

            # Verify graph received calls to add_node, add_edge and add_conditional_edges with correct arguments
            graph.add_node.assert_any_call(
                "test_human_input#request", request_instance.run
            )
            graph.add_node.assert_any_call("test_human_input#fetch", fetch_instance.run)
            graph.add_edge.assert_called_once_with(
                "test_human_input#request", "test_human_input#fetch"
            )
            graph.add_conditional_edges.assert_called_once_with(
                "test_human_input#fetch", router.route
            )

    def test_attach_passes_default_cancelled_turn_key(
        self, user, flow_type: GLReportingEventContext
    ):
        """Without an author override, FetchNode receives the implicit default input."""
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            message_template="Test message",
        )
        graph = StateGraph(FlowState)
        graph.add_node = Mock()
        graph.add_edge = Mock()
        graph.add_conditional_edges = Mock()

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.component.FetchNode"
        ) as mock_fetch_node:
            mock_fetch_node.return_value = Mock(name="test_human_input#fetch")
            component.attach(graph, Mock())

        cancelled_turn_key = mock_fetch_node.call_args[1]["cancelled_turn_key"]
        assert cancelled_turn_key == IOKey(
            target="context",
            subkeys=["inputs", "cancelled_turn"],
            optional=True,
        )

    @pytest.mark.parametrize(
        "inputs,expected_key",
        [
            (
                # Override via alias
                [
                    "context:goal",
                    {"from": "context:custom.discarded", "as": "cancelled_turn"},
                ],
                IOKey(
                    target="context",
                    subkeys=["custom", "discarded"],
                    alias="cancelled_turn",
                ),
            ),
            (
                # Override via last subkey (no alias needed)
                ["context:custom.cancelled_turn"],
                IOKey(target="context", subkeys=["custom", "cancelled_turn"]),
            ),
            (
                # Multiple overrides: the latest declaration wins
                [
                    {"from": "context:first.discarded", "as": "cancelled_turn"},
                    {"from": "context:second.discarded", "as": "cancelled_turn"},
                ],
                IOKey(
                    target="context",
                    subkeys=["second", "discarded"],
                    alias="cancelled_turn",
                ),
            ),
        ],
    )
    def test_attach_cancelled_turn_override_latest_wins(
        self,
        user,
        flow_type: GLReportingEventContext,
        inputs,
        expected_key,
    ):
        """Flow-author inputs named cancelled_turn override the default; latest wins."""
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            message_template="Test message",
            inputs=inputs,
        )
        graph = StateGraph(FlowState)
        graph.add_node = Mock()
        graph.add_edge = Mock()
        graph.add_conditional_edges = Mock()

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.component.FetchNode"
        ) as mock_fetch_node:
            mock_fetch_node.return_value = Mock(name="test_human_input#fetch")
            component.attach(graph, Mock())

        assert mock_fetch_node.call_args[1]["cancelled_turn_key"] == expected_key

    def test_default_ui_log_events(self, user, flow_type: GLReportingEventContext):
        """Test that ui_log_events defaults to both ON_USER_INPUT_PROMPT and ON_USER_RESPONSE."""
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            message_template="Test message",
            # ui_log_events not specified - should use defaults
        )

        # Verify defaults are set correctly
        assert component.ui_log_events == [
            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            UILogEventsHumanInput.ON_USER_RESPONSE,
        ]

    def test_validation_requires_on_user_input_prompt(
        self, user, flow_type: GLReportingEventContext
    ):
        """Test that validation fails if ON_USER_INPUT_PROMPT is missing."""
        with pytest.raises(
            ValueError,
            match="ui_log_events must include.*on_user_input_prompt",
        ):
            HumanInputComponent(
                name="test_human_input",
                sends_response_to="awesome_agent",
                flow_id="test_flow",
                flow_type=flow_type,
                user=user,
                message_template="Test message",
                ui_log_events=[
                    UILogEventsHumanInput.ON_USER_RESPONSE
                ],  # Missing ON_USER_INPUT_PROMPT
            )

    def test_validation_requires_on_user_response(
        self, user, flow_type: GLReportingEventContext
    ):
        """Test that validation fails if ON_USER_RESPONSE is missing."""
        with pytest.raises(
            ValueError,
            match="ui_log_events must include.*on_user_response",
        ):
            HumanInputComponent(
                name="test_human_input",
                sends_response_to="awesome_agent",
                flow_id="test_flow",
                flow_type=flow_type,
                user=user,
                message_template="Test message",
                ui_log_events=[
                    UILogEventsHumanInput.ON_USER_INPUT_PROMPT
                ],  # Missing ON_USER_RESPONSE
            )

    def test_validation_allows_additional_events(
        self, user, flow_type: GLReportingEventContext
    ):
        """Test that validation allows additional events beyond the required ones."""
        # This should succeed - all required events are present
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            message_template="Test message",
            ui_log_events=[
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
                # Could add more events here in the future
            ],
        )

        assert len(component.ui_log_events) >= 2

    def test_ui_log_events_integration(self, human_input_component):
        """Test UI log events are properly passed to nodes using factory functions."""
        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.UIHistory"
            ) as mock_ui_history,
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.agent_log_writer_class"
            ) as mock_agent_factory,
            patch(
                "duo_workflow_service.agent_platform.v1.components.human_input.component.user_log_writer_class"
            ) as mock_user_factory,
        ):
            mock_ui_history.return_value = Mock(spec=UIHistory)
            mock_agent_factory.return_value = Mock()
            mock_user_factory.return_value = Mock()

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            human_input_component.attach(graph, router)

            # Verify factories were called with the component name
            mock_agent_factory.assert_called_once_with(
                component_name="test_human_input"
            )
            mock_user_factory.assert_called_once_with(component_name="test_human_input")

            # Verify UIHistory was called with the factory results
            mock_ui_history.assert_has_calls(
                [
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=mock_agent_factory.return_value,
                    ),
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=mock_user_factory.return_value,
                    ),
                ]
            )

            request_node_ui_history = mock_request_node.call_args[1]["ui_history"]
            assert request_node_ui_history == mock_ui_history.return_value

            fetch_node_ui_history = mock_fetch_node.call_args[1]["ui_history"]
            assert fetch_node_ui_history == mock_ui_history.return_value


class TestChatHumanInputComponent:
    """Test suite for ChatHumanInputComponent."""

    @pytest.fixture(name="flow_type")
    def flow_type_fixture(self) -> GLReportingEventContext:
        return GLReportingEventContext.from_workflow_definition("chat")

    def _make_component(self, user, flow_type, **kwargs):
        return ChatHumanInputComponent(
            name="test_human_input",
            sends_response_to="some_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
            environment="chat",
            message_template="Test message",
            **kwargs,
        )

    def test_accepts_only_on_user_response(self, user, flow_type):
        """Chat component accepts ui_log_events without ON_USER_INPUT_PROMPT."""
        component = self._make_component(
            user,
            flow_type,
            ui_log_events=[UILogEventsHumanInput.ON_USER_RESPONSE],
        )
        assert component.ui_log_events == [UILogEventsHumanInput.ON_USER_RESPONSE]

    def test_accepts_both_events(self, user, flow_type):
        """Chat component also accepts both events when provided."""
        component = self._make_component(
            user,
            flow_type,
            ui_log_events=[
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
            ],
        )
        assert UILogEventsHumanInput.ON_USER_RESPONSE in component.ui_log_events


class TestHumanInputComponentFactory:
    """Test suite for human_input_component_factory."""

    @pytest.fixture(name="flow_type")
    def flow_type_fixture(self) -> GLReportingEventContext:
        return GLReportingEventContext.from_workflow_definition("chat")

    def _base_kwargs(self, user, flow_type):
        return {
            "name": "test_human_input",
            "sends_response_to": "some_agent",
            "flow_id": "test_flow",
            "flow_type": flow_type,
            "user": user,
            "message_template": "Test message",
            "ui_log_events": [
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
            ],
        }

    def test_chat_environment_returns_chat_component(self, user, flow_type):
        """Factory returns ChatHumanInputComponent for chat environment."""
        kwargs = self._base_kwargs(user, flow_type)
        kwargs["environment"] = "chat"
        kwargs["ui_log_events"] = [UILogEventsHumanInput.ON_USER_RESPONSE]
        component = human_input_component_factory(**kwargs)
        assert isinstance(component, ChatHumanInputComponent)

    def test_ambient_environment_returns_base_component(self, user, flow_type):
        """Factory returns HumanInputComponent for ambient environment."""
        kwargs = self._base_kwargs(user, flow_type)
        kwargs["environment"] = "ambient"
        component = human_input_component_factory(**kwargs)
        assert isinstance(component, HumanInputComponent)
