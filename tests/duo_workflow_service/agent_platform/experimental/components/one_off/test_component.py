"""Test suite for OneOffComponent class."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.agent_platform.experimental.components import RoutingError
from duo_workflow_service.agent_platform.experimental.components.one_off.component import (
    OneOffComponent,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory


@pytest.fixture(name="prompt_id")
def prompt_id_fixture():
    """Fixture for prompt ID."""
    return "one_off_prompt_id"


@pytest.fixture(name="prompt_version")
def prompt_version_fixture():
    """Fixture for prompt version."""
    return "v1.0"


@pytest.fixture(name="ui_log_events")
def ui_log_events_fixture():
    return []


@pytest.fixture(name="max_correction_attempts")
def max_correction_attempts_fixture():
    return 3


@pytest.fixture(name="one_off_component")
def one_off_component_fixture(
    component_name,
    flow_id,
    flow_type,
    prompt_id,
    prompt_version,
    ui_log_events,
    max_correction_attempts,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    """Fixture for OneOffComponent instance."""
    return OneOffComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        inputs=["context:user_input", "context:task_description"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        ui_log_events=ui_log_events,
        max_correction_attempts=max_correction_attempts,
    )


@pytest.fixture(name="mock_agent_node_cls")
def mock_agent_node_cls_fixture(component_name):
    """Fixture for mocked AgentNode class."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.one_off.component.AgentNode"
    ) as mock_cls:
        mock_agent_node = Mock()
        mock_agent_node.name = f"{component_name}#llm"
        mock_cls.return_value = mock_agent_node

        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(component_name):
    """Fixture for mocked ToolNodeWithErrorCorrection class."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.one_off.component.ToolNodeWithErrorCorrection"
    ) as mock_cls:
        mock_tool_node = Mock()
        mock_tool_node.name = f"{component_name}#tools"
        mock_cls.return_value = mock_tool_node

        yield mock_cls


class TestOneOffComponentInitialization:
    """Test suite for OneOffComponent initialization."""

    @pytest.mark.parametrize(
        "input_output",
        [
            "context:user_input",
            "conversation_history:agent_component",
        ],
    )
    def test_allowed_targets_through_validation(
        self,
        component_name,
        flow_id,
        flow_type,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        input_output,
    ):
        """Test that component validates input targets correctly."""
        # This should succeed without raising an exception
        OneOffComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            inputs=[input_output],
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )

    def test_iokey_template_replacement(self, one_off_component):
        """Test that IOKeyTemplate correctly replaces component name template."""
        outputs = one_off_component.outputs

        assert len(outputs) == 5

        # Check ui_chat_log output
        ui_log_output = outputs[0]
        assert ui_log_output.target == "ui_chat_log"
        assert ui_log_output.subkeys is None

        # Check conversation_history output
        conversation_output = outputs[1]
        assert conversation_output.target == "conversation_history"
        assert conversation_output.subkeys == [one_off_component.name]

        # Check tool_calls output
        tool_calls_output = outputs[2]
        assert tool_calls_output.target == "context"
        assert tool_calls_output.subkeys == [one_off_component.name, "tool_calls"]

        # Check tool_responses output
        tool_responses_output = outputs[3]
        assert tool_responses_output.target == "context"
        assert tool_responses_output.subkeys == [
            one_off_component.name,
            "tool_responses",
        ]


class TestOneOffComponentEntryHook:
    """Test suite for OneOffComponent entry hook."""

    def test_entry_hook_returns_correct_node_name(
        self, one_off_component, component_name
    ):
        """Test that __entry_hook__ returns the correct node name."""
        expected_entry_node = f"{component_name}#llm"
        assert one_off_component.__entry_hook__() == expected_entry_node


class TestOneOffComponentAttachNodes:
    """Test suite for OneOffComponent attach method."""

    def test_attach_creates_nodes_with_correct_parameters(
        self,
        mock_tool_node_cls,
        mock_agent_node_cls,
        one_off_component,
        mock_state_graph,
        mock_router,
        component_name,
        flow_id,
        flow_type,
        inputs,
        mock_toolset,
        mock_prompt_registry,
        prompt_id,
        prompt_version,
        ui_log_events,
        max_correction_attempts,
    ):
        """Test that nodes are created with correct parameters."""
        one_off_component.attach(mock_state_graph, mock_router)

        # Verify prompt registry is called with correct parameters
        mock_prompt_registry.get.assert_called_once()
        call_args = mock_prompt_registry.get.call_args

        assert call_args[0][0] == prompt_id
        assert call_args[0][1] == prompt_version

        # Check that tools and tool_choice are set correctly
        assert call_args[1]["tools"] == mock_toolset.bindable
        assert call_args[1]["tool_choice"] == "any"

        # Verify AgentNode creation
        mock_agent_node_cls.assert_called_once()
        agent_call_kwargs = mock_agent_node_cls.call_args[1]
        assert agent_call_kwargs["name"] == f"{component_name}#llm"
        assert agent_call_kwargs["component_name"] == component_name
        assert agent_call_kwargs["prompt"] == mock_prompt_registry.get.return_value
        assert agent_call_kwargs["inputs"] == inputs
        assert agent_call_kwargs["flow_id"] == flow_id
        assert agent_call_kwargs["flow_type"] == flow_type
        assert (
            agent_call_kwargs["internal_event_client"]
            == one_off_component.internal_event_client
        )

        # Verify ToolNodeWithErrorCorrection creation
        mock_tool_node_cls.assert_called_once()
        tool_call_kwargs = mock_tool_node_cls.call_args[1]
        assert tool_call_kwargs["name"] == f"{component_name}#tools"
        assert tool_call_kwargs["component_name"] == component_name
        assert tool_call_kwargs["toolset"] == mock_toolset
        assert tool_call_kwargs["flow_id"] == flow_id
        assert tool_call_kwargs["flow_type"] == flow_type
        assert (
            tool_call_kwargs["internal_event_client"]
            == one_off_component.internal_event_client
        )
        assert tool_call_kwargs["max_correction_attempts"] == max_correction_attempts

        # Tool Node UI logging
        assert "ui_history" in tool_call_kwargs
        assert isinstance(tool_call_kwargs["ui_history"], UIHistory)
        assert tool_call_kwargs["ui_history"].events == ui_log_events

        # Verify IOKey parameters
        assert "tool_calls_key" in tool_call_kwargs
        assert "tool_responses_key" in tool_call_kwargs


class TestOneOffComponentAttachEdges:
    """Test suite for OneOffComponent graph structure and routing."""

    def test_attach_creates_graph_structure(
        self,
        one_off_component,
        mock_state_graph,
        mock_router,
        component_name,
        mock_agent_node_cls,
        mock_tool_node_cls,
    ):
        """Test that attach method creates proper graph structure."""
        one_off_component.attach(mock_state_graph, mock_router)

        expected_llm_node = f"{component_name}#llm"
        expected_tools_node = f"{component_name}#tools"
        expected_exit_node = f"{component_name}#exit"

        # Verify nodes were added
        mock_state_graph.add_node.assert_any_call(
            expected_llm_node, mock_agent_node_cls.return_value.run
        )
        mock_state_graph.add_node.assert_any_call(
            expected_tools_node, mock_tool_node_cls.return_value.run
        )

        # Verify exit node was added (dynamically created)
        exit_node_calls = [
            call
            for call in mock_state_graph.add_node.call_args_list
            if call[0][0] == expected_exit_node
        ]
        assert len(exit_node_calls) == 1

        # Verify edges were added
        mock_state_graph.add_edge.assert_called_once_with(
            expected_llm_node, expected_tools_node
        )

        # Verify conditional edges were added
        mock_state_graph.add_conditional_edges.assert_any_call(
            expected_tools_node,
            one_off_component._tools_router,
            {
                "retry": expected_llm_node,
                "exit": expected_exit_node,
            },
        )

        mock_state_graph.add_conditional_edges.assert_any_call(
            expected_exit_node, mock_router.route
        )


class TestOneOffComponentToolsRouter:
    """Test suite for OneOffComponent tools routing logic."""

    def test_tools_router_with_success_message(
        self, one_off_component, base_flow_state, component_name
    ):
        """Test tools router with success message routes to exit."""
        # Create state with success message
        success_message = HumanMessage(content="Tool execution completed successfully")
        state_with_success = base_flow_state.copy()
        state_with_success[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [success_message]
        }

        result = one_off_component._tools_router(state_with_success)
        assert result == "exit"

    def test_tools_router_with_retry_message(
        self, one_off_component, base_flow_state, component_name
    ):
        """Test tools router with retry message routes to retry."""
        # Create state with retry message
        retry_message = HumanMessage(content="Error occurred. 2 attempts remaining")
        state_with_retry = base_flow_state.copy()
        state_with_retry[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [retry_message]
        }

        result = one_off_component._tools_router(state_with_retry)
        assert result == "retry"

    def test_tools_router_with_max_attempts_reached(
        self, one_off_component, base_flow_state, component_name
    ):
        """Test tools router with max attempts message routes to exit."""
        # Create state with max attempts message
        max_attempts_message = HumanMessage(
            content="Error occurred. 0 attempts remaining"
        )
        state_with_max_attempts = base_flow_state.copy()
        state_with_max_attempts[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [max_attempts_message]
        }

        result = one_off_component._tools_router(state_with_max_attempts)
        assert result == "exit"

    def test_tools_router_with_empty_conversation_history(
        self, one_off_component, base_flow_state, component_name
    ):
        """Test tools router with empty conversation history raises error."""
        state_empty_history = base_flow_state.copy()
        state_empty_history[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: []}

        with pytest.raises(RoutingError) as exc_info:
            one_off_component._tools_router(state_empty_history)

        assert "No conversation history found for component" in str(exc_info.value)
