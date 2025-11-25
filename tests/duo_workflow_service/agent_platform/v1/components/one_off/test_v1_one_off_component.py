"""Test suite for OneOffComponent class."""

from unittest.mock import ANY, Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ai_gateway.model_metadata import ModelMetadata, current_model_metadata_context
from duo_workflow_service.agent_platform.v1.components.one_off.component import (
    OneOffComponent,
)
from duo_workflow_service.agent_platform.v1.state import FlowState, FlowStateKeys
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory


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
        "duo_workflow_service.agent_platform.v1.components.one_off.component.AgentNode"
    ) as mock_cls:
        mock_agent_node = Mock()
        mock_agent_node.name = f"{component_name}#llm"
        mock_cls.return_value = mock_agent_node

        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(component_name):
    """Fixture for mocked ToolNodeWithErrorCorrection class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.one_off.component.ToolNodeWithErrorCorrection"
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
        assert call_args[1]["internal_event_extra"] == {
            "agent_name": component_name,
            "workflow_id": flow_id,
            "workflow_type": flow_type.value,
        }

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

        # Verify nodes were added
        mock_state_graph.add_node.assert_any_call(
            expected_llm_node, mock_agent_node_cls.return_value.run
        )
        mock_state_graph.add_node.assert_any_call(
            expected_tools_node, mock_tool_node_cls.return_value.run
        )

        # Verify edges were added
        mock_state_graph.add_edge.assert_called_once_with(
            expected_llm_node, expected_tools_node
        )

        # Verify conditional edges were added
        mock_state_graph.add_conditional_edges.assert_any_call(expected_tools_node, ANY)


class TestOneOffComponentToolsRouter:
    """Test suite for OneOffComponent execution flow and routing behavior."""

    def test_successful_tool_execution_flow(
        self,
        component_name,
        flow_id,
        flow_type,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        ui_log_events,
        max_correction_attempts,
        mock_router,
        mock_agent_node_cls,
        mock_tool_node_cls,
        base_flow_state,
    ):
        """Test execution flow when tool execution is successful."""
        # Mock agent node to produce tool calls
        graph = StateGraph(FlowState)
        mock_agent_node = mock_agent_node_cls.return_value
        mock_agent_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="I need to call a tool")]
            },
        }

        # Mock tool node to simulate successful completion
        mock_tool_node = mock_tool_node_cls.return_value
        mock_tool_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [
                    ToolMessage(
                        content="Tool execution completed successfully",
                        tool_call_id="123",
                    ),
                ]
            },
        }

        # Mock router to return END (exit)
        mock_router.route.return_value = END

        component = OneOffComponent(
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

        component.attach(graph, mock_router)
        graph.set_entry_point(component.__entry_hook__())
        compiled_graph = graph.compile()

        # Execute the graph
        result = compiled_graph.invoke(base_flow_state)

        # Verify both nodes were called
        mock_agent_node.run.assert_called_once()
        mock_tool_node.run.assert_called_once()

        # Verify outgoing router was called
        mock_router.route.assert_called_once()

        # Verify final state contains success message
        final_conversation = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert any(
            "completed successfully" in msg.content for msg in final_conversation
        )

    def test_tool_execution_retry_flow(
        self,
        component_name,
        flow_id,
        flow_type,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        ui_log_events,
        max_correction_attempts,
        mock_router,
        mock_agent_node_cls,
        mock_tool_node_cls,
        base_flow_state,
    ):
        """Test execution flow when tool execution needs retry."""
        # Mock agent node to produce tool calls (called twice due to retry)
        graph = StateGraph(FlowState)
        mock_agent_node = mock_agent_node_cls.return_value
        mock_agent_node.run.side_effect = [
            # First call - initial attempt
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    component_name: [AIMessage(content="I need to call a tool")]
                },
            },
            # Second call - retry attempt
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    component_name: [
                        AIMessage(content="Retrying with corrected approach"),
                    ]
                },
            },
        ]

        # Mock tool node to simulate error then success
        mock_tool_node = mock_tool_node_cls.return_value
        mock_tool_node.run.side_effect = [
            # First call - error with retry
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    component_name: [
                        ToolMessage(
                            content="Error occurred. 2 attempts remaining",
                            tool_call_id="123",
                        ),
                    ]
                },
            },
            # Second call - success
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    component_name: [
                        ToolMessage(
                            content="Tool execution completed successfully",
                            tool_call_id="123",
                        ),
                    ]
                },
            },
        ]

        # Mock router to return END after success
        mock_router.route.return_value = END

        component = OneOffComponent(
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

        component.attach(graph, mock_router)
        graph.set_entry_point(component.__entry_hook__())
        compiled_graph = graph.compile()

        # Execute the graph
        result = compiled_graph.invoke(base_flow_state)

        # Verify both nodes were called multiple times (retry logic)
        assert mock_agent_node.run.call_count == 2
        assert mock_tool_node.run.call_count == 2

        # Verify final state contains success message
        final_conversation = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert any(
            "completed successfully" in msg.content for msg in final_conversation
        )

    def test_max_attempts_reached_flow(
        self,
        component_name,
        flow_id,
        flow_type,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        ui_log_events,
        max_correction_attempts,
        mock_router,
        mock_agent_node_cls,
        mock_tool_node_cls,
        base_flow_state,
    ):
        """Test execution flow when max attempts are reached."""
        # Mock agent node
        graph = StateGraph(FlowState)
        mock_agent_node = mock_agent_node_cls.return_value
        mock_agent_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="I need to call a tool")]
            },
        }

        # Mock tool node to simulate max attempts reached
        mock_tool_node = mock_tool_node_cls.return_value
        mock_tool_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [
                    ToolMessage(
                        content="Error occurred. 0 attempts remaining",
                        tool_call_id="123",
                    ),
                ]
            },
        }

        # Mock router to return END (exit due to max attempts)
        mock_router.route.return_value = END

        component = OneOffComponent(
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

        component.attach(graph, mock_router)
        graph.set_entry_point(component.__entry_hook__())
        compiled_graph = graph.compile()

        # Execute the graph
        result = compiled_graph.invoke(base_flow_state)

        # Verify execution occurred
        mock_agent_node.run.assert_called_once()
        mock_tool_node.run.assert_called_once()

        # Verify final state contains max attempts message
        final_conversation = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert any("0 attempts remaining" in msg.content for msg in final_conversation)

    def test_component_state_management_through_execution(
        self,
        component_name,
        flow_id,
        flow_type,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        ui_log_events,
        max_correction_attempts,
        mock_router,
        mock_agent_node_cls,
        mock_tool_node_cls,
        base_flow_state,
    ):
        """Test that component properly manages state through execution."""
        # Mock agent node to update state
        graph = StateGraph(FlowState)
        mock_agent_node = mock_agent_node_cls.return_value
        mock_agent_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="Agent response")]
            },
            "context": {component_name: {"tool_calls": ["call_1", "call_2"]}},
        }

        # Mock tool node to update state further
        mock_tool_node = mock_tool_node_cls.return_value
        mock_tool_node.run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [
                    ToolMessage(
                        content="Tool execution completed successfully",
                        tool_call_id="123",
                    ),
                ]
            },
            "context": {
                component_name: {
                    "tool_calls": ["call_1", "call_2"],
                    "tool_responses": ["response_1", "response_2"],
                    "execution_result": "success",
                }
            },
        }

        # Mock router to return END
        mock_router.route.return_value = END

        component = OneOffComponent(
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

        component.attach(graph, mock_router)
        graph.set_entry_point(component.__entry_hook__())
        compiled_graph = graph.compile()

        # Execute the graph
        result = compiled_graph.invoke(base_flow_state)

        # Verify final state contains all expected context data
        assert "context" in result
        assert component_name in result["context"]
        context = result["context"][component_name]

        assert "tool_calls" in context
        assert "tool_responses" in context
        assert "execution_result" in context
        assert context["execution_result"] == "success"


class TestOneOffComponentModelMetadata:
    """Test suite for OneOffComponent model metadata handling."""

    def test_attach_passes_model_metadata_from_context_to_prompt_registry(
        self,
        one_off_component,
        mock_state_graph,
        mock_router,
        mock_prompt_registry,
        model_metadata: ModelMetadata,
        prompt_id,
        prompt_version,
    ):
        metadata_token = current_model_metadata_context.set(model_metadata)

        try:
            with (
                patch(
                    "duo_workflow_service.agent_platform.v1.components.one_off.component.AgentNode"
                ),
                patch(
                    "duo_workflow_service.agent_platform.v1.components.one_off.component.ToolNodeWithErrorCorrection"
                ),
            ):
                one_off_component.attach(mock_state_graph, mock_router)

            mock_prompt_registry.get.assert_called_once()
            call_kwargs = mock_prompt_registry.get.call_args[1]

            assert "model_metadata" in call_kwargs
            assert call_kwargs["model_metadata"] == model_metadata
        finally:
            current_model_metadata_context.reset(metadata_token)

    def test_attach_passes_none_when_no_model_metadata_in_context(
        self,
        one_off_component,
        mock_state_graph,
        mock_router,
        mock_prompt_registry,
    ):
        metadata_token = current_model_metadata_context.set(None)

        try:
            with (
                patch(
                    "duo_workflow_service.agent_platform.v1.components.one_off.component.AgentNode"
                ),
                patch(
                    "duo_workflow_service.agent_platform.v1.components.one_off.component.ToolNodeWithErrorCorrection"
                ),
            ):
                one_off_component.attach(mock_state_graph, mock_router)

            mock_prompt_registry.get.assert_called_once()
            call_kwargs = mock_prompt_registry.get.call_args[1]

            assert "model_metadata" in call_kwargs
            assert call_kwargs["model_metadata"] is None
        finally:
            current_model_metadata_context.reset(metadata_token)
