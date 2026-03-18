from unittest.mock import Mock, call, patch

import pytest
from langgraph.graph import StateGraph

from duo_workflow_service.agent_platform.experimental.components.human_input.component import (
    HumanInputComponent,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
    UserLogWriter,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey
from duo_workflow_service.agent_platform.experimental.ui_log.base import UIHistory
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
            message_template="test_message_template",
            ui_log_events=[
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
            ],
        )

    def test_iokey_template_replacement(self, human_input_component):
        """Test that IOKeyTemplate correctly replaces SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE."""
        outputs = human_input_component.outputs

        assert len(outputs) == 2

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

    def test_attach_method_creates_nodes(self, human_input_component):
        """Test that attach method creates proper nodes in the graph."""

        graph = StateGraph(FlowState)
        router = Mock()
        router.route = Mock(return_value="next_node")

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
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
                message_template=mock_request_node.call_args[1]["message_template"],
                inputs=human_input_component.inputs,
                ui_history=mock_request_node.call_args[1]["ui_history"],
            )

            mock_fetch_node.assert_called_once_with(
                name="test_human_input#fetch",
                component_name="test_human_input",
                sends_response_to="awesome_agent",
                output=human_input_component._approval_output,
                ui_history=mock_fetch_node.call_args[1]["ui_history"],
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

    def test_message_template(self, human_input_component):
        """Test integration with AgentComponent."""
        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            human_input_component.attach(graph, router)

            # Verify request node was created with message_template
            call_args = mock_request_node.call_args
            assert call_args[1]["message_template"] == "test_message_template"

    def test_optional_message_template(self, user, flow_type: GLReportingEventContext):
        """Test that message_template is optional when no message_template provided."""
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=flow_type,
            user=user,
        )

        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            component.attach(graph, router)

            # Verify request node was created with None message_template
            call_args = mock_request_node.call_args
            assert call_args[1]["message_template"] is None

    def test_ui_log_events_integration(self, human_input_component):
        """Test UI log events are properly passed to nodes."""
        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.UIHistory"
            ) as mock_ui_history,
        ):
            mock_ui_history.return_value = Mock(spec=UIHistory)

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            human_input_component.attach(graph, router)

            mock_ui_history.assert_has_calls(
                [
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=AgentLogWriter,
                    ),
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=UserLogWriter,
                    ),
                ]
            )

            request_node_ui_history = mock_request_node.call_args[1]["ui_history"]
            assert request_node_ui_history == mock_ui_history.return_value

            fetch_node_ui_history = mock_fetch_node.call_args[1]["ui_history"]
            assert fetch_node_ui_history == mock_ui_history.return_value
