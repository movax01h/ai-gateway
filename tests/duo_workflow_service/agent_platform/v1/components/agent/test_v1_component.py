"""Test suite for AgentComponent class."""

# pylint: disable=file-naming-for-tests
from typing import ClassVar, Literal
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field

from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import IOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory


@pytest.fixture(name="prompt_id")
def prompt_id_fixture():
    """Fixture for prompt ID."""
    return "test_prompt_id"


@pytest.fixture(name="prompt_version")
def prompt_version_fixture():
    """Fixture for prompt version."""
    return "v1.0"


@pytest.fixture(name="ui_log_events")
def ui_log_events_fixture():
    return []


@pytest.fixture(name="ui_role_as")
def ui_role_as_fixture() -> Literal["agent", "tool"]:
    return "agent"


@pytest.fixture(name="agent_component")
def agent_component_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    prompt_id,
    prompt_version,
    ui_log_events,
    ui_role_as,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    """Fixture for AgentComponent instance."""
    return AgentComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=["context:user_input", "context:task_description"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        ui_log_events=ui_log_events,
        ui_role_as=ui_role_as,
    )


@pytest.fixture(name="agent_component_with_custom_schema")
def agent_component_with_custom_schema_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    prompt_id,
    prompt_version,
    ui_log_events,
    ui_role_as,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
    mock_schema_registry,
):
    """Fixture for AgentComponent instance with custom response schema."""
    # Exclude response schema tool from toolset to avoid collision
    mock_schema = mock_schema_registry.get.return_value
    mock_toolset.__contains__ = lambda self, name: name != mock_schema.tool_title

    return AgentComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=["context:user_input", "context:task_description"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        ui_log_events=ui_log_events,
        ui_role_as=ui_role_as,
        response_schema_id="general/structured_response",
        response_schema_version="1.0.0",
        schema_registry=mock_schema_registry,
    )


@pytest.fixture(name="agent_component_no_output")
def agent_component_no_output_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    prompt_id,
    prompt_version,
    ui_log_events,
    ui_role_as,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    """Fixture for AgentComponent instance without output."""
    return AgentComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=["context:user_input", "context:task_description"],
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        ui_log_events=ui_log_events,
        ui_role_as=ui_role_as,
    )


@pytest.fixture(name="mock_agent_node_cls")
def mock_agent_node_cls_fixture(component_name):
    """Fixture for mocked AgentNode class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.component.AgentNode"
    ) as mock_cls:
        mock_agent_node = Mock()
        mock_agent_node.name = f"{component_name}#agent"
        mock_cls.return_value = mock_agent_node

        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(component_name):
    """Fixture for mocked ToolNode class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.component.ToolNode"
    ) as mock_cls:
        mock_tool_node = Mock()
        mock_tool_node.name = f"{component_name}#tools"
        mock_cls.return_value = mock_tool_node

        yield mock_cls


@pytest.fixture(name="mock_final_response_node_cls")
def mock_final_response_node_cls_fixture(component_name):
    """Fixture for mocked FinalResponseNode class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.component.FinalResponseNode"
    ) as mock_cls:
        mock_final_response_node = Mock()
        mock_final_response_node.name = f"{component_name}#final_response"
        mock_cls.return_value = mock_final_response_node

        yield mock_cls


@pytest.fixture(name="mock_schema_registry")
def mock_schema_registry_fixture():
    """Fixture for mock schema registry."""
    mock_registry = Mock(spec=BaseResponseSchemaRegistry)

    # Create a mock custom schema class
    class CustomResponseTool(BaseModel):
        model_config = ConfigDict(frozen=True)

        tool_title: ClassVar[str] = "custom_response_tool"

        summary: str = Field(description="Summary")
        score: int = Field(description="Score")

        @classmethod
        def from_ai_message(cls, msg):
            return cls(**msg.tool_calls[0]["args"])

    mock_registry.get.return_value = CustomResponseTool
    return mock_registry


class TestAgentComponentInitialization:
    """Test suite for AgentComponent initialization."""

    @pytest.mark.parametrize(
        ("input_output"),
        [
            "context:user_input",
            "conversation_history:agent_component",
            "status",
            "ui_chat_log",
        ],
    )
    def test_allowed_targets_through_validation(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        input_output,
    ):
        """Test that component validates input targets correctly."""
        # This should succeed without raising an exception
        AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[input_output],
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )


class TestAgentComponentEntryHook:
    """Test suite for AgentComponent entry hook."""

    def test_entry_hook_returns_correct_node_name(
        self, agent_component, component_name
    ):
        """Test that __entry_hook__ returns the correct node name."""
        expected_entry_node = f"{component_name}#agent"
        assert agent_component.__entry_hook__() == expected_entry_node


class TestAgentComponentAttachNodes:
    """Test suite for AgentComponent attach method."""

    @pytest.mark.parametrize(
        ("ui_log_events", "ui_role_as"),
        [
            ([], "agent"),
            # Default values
            ([UILogEventsAgent.ON_AGENT_FINAL_ANSWER], "agent"),
            # Custom events, default role
            ([], "tool"),
            # Default events, custom role
            (
                [
                    UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
                    UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
                ],
                "tool",
            ),
            # Custom values
        ],
    )
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    def test_attach_creates_nodes_with_correct_parameters(
        self,
        mock_final_response_node_cls,
        mock_tool_node_cls,
        mock_agent_node_cls,
        agent_component,
        mock_state_graph,
        mock_router,
        component_name,
        flow_id,
        flow_type,
        user,
        inputs,
        mock_toolset,
        mock_internal_event_client,
        mock_prompt_registry,
        prompt_id,
        prompt_version,
        ui_log_events,
        ui_role_as,
    ):
        """Test that nodes are created with correct parameters."""
        agent_component.attach(mock_state_graph, mock_router)

        # Verify prompt registry is called with correct parameters
        mock_prompt_registry.get_on_behalf.assert_called_once_with(
            user,
            prompt_id,
            prompt_version,
            model_metadata=None,
            tools=mock_toolset.bindable,
            tool_choice="auto",
            internal_event_extra={
                "agent_name": component_name,
                "workflow_id": flow_id,
                "workflow_type": flow_type.value,
            },
        )

        # Verify AgentNode creation
        mock_agent_node_cls.assert_called_once()
        agent_call_kwargs = mock_agent_node_cls.call_args[1]
        assert agent_call_kwargs["name"] == f"{component_name}#agent"
        assert agent_call_kwargs["component_name"] == component_name
        assert (
            agent_call_kwargs["prompt"]
            == mock_prompt_registry.get_on_behalf.return_value
        )
        assert agent_call_kwargs["inputs"] == inputs
        assert agent_call_kwargs["flow_id"] == flow_id
        assert agent_call_kwargs["flow_type"] == flow_type
        assert agent_call_kwargs["internal_event_client"] == mock_internal_event_client

        # Verify ToolNode creation
        mock_tool_node_cls.assert_called_once()
        tool_call_kwargs = mock_tool_node_cls.call_args[1]
        assert tool_call_kwargs["name"] == f"{component_name}#tools"
        assert tool_call_kwargs["component_name"] == component_name
        assert tool_call_kwargs["toolset"] == mock_toolset
        assert tool_call_kwargs["flow_id"] == flow_id
        assert tool_call_kwargs["flow_type"] == flow_type
        assert tool_call_kwargs["internal_event_client"] == mock_internal_event_client

        # Tool Node UI logging
        assert "ui_history" in tool_call_kwargs
        assert isinstance(tool_call_kwargs["ui_history"], UIHistory)
        assert tool_call_kwargs["ui_history"].events == ui_log_events

        # Verify FinalResponseNode creation
        mock_final_response_node_cls.assert_called_once()
        final_call_kwargs = mock_final_response_node_cls.call_args[1]
        assert final_call_kwargs["name"] == f"{component_name}#final_response"
        assert final_call_kwargs["component_name"] == component_name
        assert final_call_kwargs["output"] == IOKey(
            target="context", subkeys=[component_name, "final_answer"]
        )

        # FinalResponse Node UI logging
        assert "ui_history" in final_call_kwargs
        assert isinstance(final_call_kwargs["ui_history"], UIHistory)
        assert final_call_kwargs["ui_history"].events == ui_log_events


class TestAgentComponentAttachEdges:
    """Test suite for AgentComponent routing behavior through graph execution."""

    # pylint: disable=unused-argument
    def test_routing_with_custom_schema_tool_call_goes_to_final_response(
        self,
        request,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
        mock_custom_schema_tool_call,
        agent_component_with_custom_schema,
    ):
        """Test that custom schema tool call routes to final response node."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [mock_custom_schema_tool_call]

        state_with_final_tool = base_flow_state.copy()
        state_with_final_tool[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component_with_custom_schema.attach(mock_state_graph, mock_router)

        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        result = router_function(state_with_final_tool)
        assert result == f"{component_name}#final_response"

    # pylint: disable=unused-argument
    @pytest.mark.parametrize(
        "component_fixture",
        ["agent_component", "agent_component_with_custom_schema"],
        ids=["default_schema", "custom_schema"],
    )
    def test_routing_with_other_tool_calls_goes_to_tools(
        self,
        request,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_other_tool_call,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
        component_fixture,
    ):
        """Test that non-final tool calls route to tools node."""
        agent_component = request.getfixturevalue(component_fixture)

        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [mock_other_tool_call]

        state_with_other_tool = base_flow_state.copy()
        state_with_other_tool[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component.attach(mock_state_graph, mock_router)

        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        result = router_function(state_with_other_tool)
        assert result == f"{component_name}#tools"

    # pylint: disable=unused-argument
    def test_routing_with_mixed_tool_calls_prioritizes_final_response(
        self,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_other_tool_call,
        mock_custom_schema_tool_call,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
        agent_component_with_custom_schema,
    ):
        """Test that mixed tool calls prioritize final response routing (custom schema)."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [mock_other_tool_call, mock_custom_schema_tool_call]

        state_with_mixed_tools = base_flow_state.copy()
        state_with_mixed_tools[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component_with_custom_schema.attach(mock_state_graph, mock_router)

        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        result = router_function(state_with_mixed_tools)
        assert result == f"{component_name}#final_response"

    # pylint: disable=unused-argument
    def test_routing_with_without_conversation_history(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_final_tool_call,
        mock_other_tool_call,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
    ):
        """Test that mixed tool calls prioritize final response routing."""
        # Create state with mixed tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [mock_other_tool_call, mock_final_tool_call]

        state_with_mixed_tools = base_flow_state.copy()
        state_with_mixed_tools[FlowStateKeys.CONVERSATION_HISTORY] = {}

        agent_component.attach(mock_state_graph, mock_router)

        # Get the router function that was passed to add_conditional_edges
        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        # Test the routing behavior - should raise RoutingError
        with pytest.raises(
            RoutingError, match=f"Conversation history not found for {component_name}"
        ):
            router_function(base_flow_state)

    # pylint: disable=unused-argument
    def test_routing_with_non_ai_message_raises_error(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
    ):
        """Test that non-AIMessage raises RoutingError."""
        # Create state with non-AIMessage
        mock_message = Mock()  # Not an AIMessage

        state_with_non_ai_message = base_flow_state.copy()
        state_with_non_ai_message[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component.attach(mock_state_graph, mock_router)

        # Get the router function that was passed to add_conditional_edges
        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        # Test the routing behavior - should raise RoutingError
        with pytest.raises(
            RoutingError,
            match=f"Last message is not AIMessage for component {component_name}",
        ):
            router_function(state_with_non_ai_message)

    # pylint: disable=unused-argument
    def test_routing_with_no_tool_calls_goes_to_final_response(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
    ):
        """Test that messages with no tool calls route to final_response."""
        # Create state with AIMessage but no tool calls
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state_with_no_tools = base_flow_state.copy()
        state_with_no_tools[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component.attach(mock_state_graph, mock_router)

        # Get the router function that was passed to add_conditional_edges
        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        result = router_function(state_with_no_tools)
        assert result == f"{component_name}#final_response"


class TestAgentComponentResponseSchema:
    """Test suite for custom response schema functionality."""

    # pylint: disable=unused-argument
    @pytest.mark.parametrize(
        ("component_fixture", "use_custom_schema"),
        [
            ("agent_component", False),
            ("agent_component_with_custom_schema", True),
        ],
        ids=["default_schema", "custom_schema"],
    )
    def test_attach_passes_correct_schema_to_nodes(
        self,
        request,
        mock_agent_node_cls,
        mock_final_response_node_cls,
        mock_state_graph,
        mock_router,
        mock_schema_registry,
        component_fixture,
        use_custom_schema,
    ):
        """Test that nodes receive the correct response schema (default or custom)."""
        component = request.getfixturevalue(component_fixture)
        component.attach(mock_state_graph, mock_router)

        # Determine expected schema
        if use_custom_schema:
            expected_schema = mock_schema_registry.get.return_value
        else:
            expected_schema = None

        # Verify AgentNode received correct schema
        agent_call_kwargs = mock_agent_node_cls.call_args[1]
        assert agent_call_kwargs["response_schema"] == expected_schema

        # Verify FinalResponseNode received correct schema
        final_call_kwargs = mock_final_response_node_cls.call_args[1]
        assert final_call_kwargs["response_schema"] == expected_schema

    @pytest.mark.parametrize(
        ("schema_id", "schema_version", "should_raise"),
        [
            ("test/schema", "^1.0.0", False),  # Both provided - valid
            (None, None, False),  # Neither provided - valid
            ("test/schema", None, True),  # Only ID - invalid
            (None, "^1.0.0", True),  # Only version - invalid
        ],
    )
    def test_id_and_version_are_provided(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        schema_id,
        schema_version,
        should_raise,
        mock_schema_registry,
    ):
        """Test validation of response_schema_id and response_schema_version parameters."""
        # Exclude response schema tool from toolset to avoid collision
        mock_schema = mock_schema_registry.get.return_value
        mock_toolset.__contains__ = lambda self, name: name != mock_schema.tool_title

        if should_raise:
            with pytest.raises(ValueError, match="must be provided together"):
                AgentComponent(
                    name=component_name,
                    flow_id=flow_id,
                    flow_type=flow_type,
                    user=user,
                    inputs=["context:user_input"],
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    toolset=mock_toolset,
                    response_schema_id=schema_id,
                    response_schema_version=schema_version,
                    prompt_registry=mock_prompt_registry,
                    internal_event_client=mock_internal_event_client,
                    schema_registry=mock_schema_registry,
                )
        else:
            # Should not raise
            component = AgentComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                toolset=mock_toolset,
                response_schema_id=schema_id,
                response_schema_version=schema_version,
                prompt_registry=mock_prompt_registry,
                internal_event_client=mock_internal_event_client,
                schema_registry=mock_schema_registry,
            )
            assert component.response_schema_id == schema_id
            assert component.response_schema_version == schema_version

    def test_tool_name_collision_with_response_schema(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        prompt_version,
        mock_toolset,
        mock_prompt_registry,
        mock_schema_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
    ):
        """Test that response schema colliding with tool name raises error."""
        # Create a mock tool with name "colliding_response_tool"
        mock_tool = Mock()
        mock_tool.name = "colliding_response_tool"
        mock_toolset.__contains__ = Mock(return_value=True)  # Simulate collision

        # Mock schema with same tool_title
        mock_schema = Mock()
        mock_schema.tool_title = "colliding_response_tool"
        mock_schema.model_fields = {}  # Add model_fields for outputs property
        mock_schema_registry.get.return_value = mock_schema

        # Override toolset to report this specific tool exists (creating actual collision)
        mock_toolset.__contains__ = lambda self, name: name == "colliding_response_tool"

        # Collision is now detected during __init__, not attach()
        with pytest.raises(ValueError, match="collides with existing tool"):
            AgentComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                toolset=mock_toolset,
                response_schema_id="test/schema",
                response_schema_version="1.0.0",
                schema_registry=mock_schema_registry,
                prompt_registry=mock_prompt_registry,
                internal_event_client=mock_internal_event_client,
            )


class TestAgentComponentOutputs:
    """Test suite for Agent Component Outputs."""

    def test_outputs_with_agent_final_output(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    ):
        """Test that outputs property returns base outputs only for default schema."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            # No response_schema_id/version - no custom response schema
        )

        outputs = component.outputs

        # Should return only the 3 base outputs (no custom field outputs)
        assert len(outputs) == 3
        assert all(isinstance(output, IOKey) for output in outputs)

        # Verify the base outputs exist
        output_targets = [(out.target, out.subkeys) for out in outputs]
        assert ("conversation_history", [component_name]) in output_targets
        assert ("status", None) in output_targets
        assert ("context", [component_name, "final_answer"]) in output_targets

    def test_outputs_with_custom_schema_includes_field_outputs(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
    ):
        """Test that custom schema outputs include base outputs + field-level outputs."""
        # Exclude response schema tool from toolset
        mock_schema = mock_schema_registry.get.return_value
        mock_toolset.__contains__ = lambda self, name: name != mock_schema.tool_title

        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            response_schema_id="test/custom_schema",
            response_schema_version="1.0.0",
            schema_registry=mock_schema_registry,
        )

        outputs = component.outputs

        # Should have 3 base outputs + 2 custom fields (summary, score)
        assert len(outputs) == 5
        assert all(isinstance(output, IOKey) for output in outputs)

        # Verify base outputs exist
        output_keys = [(out.target, out.subkeys) for out in outputs]
        assert ("conversation_history", [component_name]) in output_keys
        assert ("status", None) in output_keys
        assert ("context", [component_name, "final_answer"]) in output_keys

        # Verify custom field outputs exist
        assert ("context", [component_name, "final_answer", "summary"]) in output_keys
        assert ("context", [component_name, "final_answer", "score"]) in output_keys
