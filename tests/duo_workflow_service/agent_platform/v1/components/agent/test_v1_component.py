# pylint: disable=too-many-lines,file-naming-for-tests
"""Test suite for AgentComponent class."""

from typing import ClassVar, Literal
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_gateway.response_schemas import (
    BaseResponseSchemaRegistry,
    InlineResponseSchemaRegistry,
)
from ai_gateway.response_schemas.base import BaseAgentOutput
from ai_gateway.response_schemas.registry import ResponseSchemaRegistry
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
    AgentComponentBase,
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowEventType,
    FlowState,
    FlowStateKeys,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.state.base import IOKey, NoneIOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.conversation.compaction import CompactionConfig
from duo_workflow_service.tools.toolset import Toolset


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


class TestAgentComponentBase:
    """Test suite for AgentComponentBase abstract stubs."""

    def test_agent_node_invoke_config_raises_not_implemented(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    ):
        """Base _agent_node_invoke_config must raise NotImplementedError."""

        class ConcreteBase(AgentComponentBase):
            pass

        component = ConcreteBase(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],
            prompt_id="test",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )
        with pytest.raises(NotImplementedError):
            component._agent_node_invoke_config()

    def test_agent_node_router_raises_not_implemented(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    ):
        """Base _agent_node_router must raise NotImplementedError."""

        class ConcreteBase(AgentComponentBase):
            pass

        component = ConcreteBase(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],
            prompt_id="test",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )
        with pytest.raises(NotImplementedError):
            component._agent_node_router({})

    def test_attach_raises_not_implemented(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
    ):
        """Base attach must raise NotImplementedError."""

        class ConcreteBase(AgentComponentBase):
            pass

        component = ConcreteBase(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],
            prompt_id="test",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )
        with pytest.raises(NotImplementedError):
            component.attach(mock_state_graph, mock_router)

    def test_default_conversation_history_key_returns_correct_runtime_io_key(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    ):
        """_default_conversation_history_key returns a RuntimeIOKey wrapping the correct static IOKey."""

        class ConcreteBase(AgentComponentBase):
            pass

        component = ConcreteBase(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],
            prompt_id="test",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )
        runtime_key = component._default_conversation_history_key
        assert isinstance(runtime_key, RuntimeIOKey)

        iokey = runtime_key.to_iokey({})
        assert iokey.target == "conversation_history"
        assert iokey.subkeys == [component_name]
        assert iokey.optional is True

    @pytest.mark.parametrize(
        ("schema_id", "schema_version", "should_raise"),
        [
            ("test/schema", "^1.0.0", False),  # Both provided — registry mode, valid
            (None, None, False),  # Neither provided — text-only mode, valid
            ("test/schema", None, False),  # ID only — inline mode, valid
            (None, "^1.0.0", True),  # Version only (no ID) — invalid
        ],
    )
    def test_response_schema_id_and_version_combinations(
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
        schema_id,
        schema_version,
        should_raise,
    ):
        """response_schema_id alone = inline mode; both = registry mode; version alone = invalid."""

        class ConcreteBase(AgentComponentBase):
            pass

        if should_raise:
            with pytest.raises(ValueError, match="requires response_schema_id"):
                ConcreteBase(
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
                )
        else:
            if schema_id:
                mock_schema = mock_schema_registry.get.return_value
                mock_toolset.__contains__ = lambda self, name: (
                    name != mock_schema.tool_title
                )

            component = ConcreteBase(
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
                schema_registry=mock_schema_registry,
                prompt_registry=mock_prompt_registry,
                internal_event_client=mock_internal_event_client,
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
    ):
        """Response schema tool title colliding with an existing tool raises ValueError."""

        class ConcreteBase(AgentComponentBase):
            pass

        mock_schema = Mock()
        mock_schema.tool_title = "colliding_response_tool"
        mock_schema_registry.get.return_value = mock_schema
        mock_toolset.__contains__ = lambda self, name: name == "colliding_response_tool"

        with pytest.raises(ValueError, match="collides with existing tool"):
            ConcreteBase(
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


class TestResponseSchemaDefinition:
    """Tests for inline response schemas via the response_schemas flow config block."""

    INLINE_SCHEMA_ID = "my_inline_schema"

    @pytest.fixture(name="simple_inline_schema")
    def simple_inline_schema_fixture(self):
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["summary", "score"],
        }

    @pytest.fixture(name="inline_schema_registry")
    def inline_schema_registry_fixture(self, simple_inline_schema):
        """An InlineResponseSchemaRegistry pre-loaded with the simple inline schema."""
        registry = InlineResponseSchemaRegistry(ResponseSchemaRegistry())
        registry.register_schema(self.INLINE_SCHEMA_ID, simple_inline_schema)
        return registry

    def _make_component(
        self,
        inline_schema_registry,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        extra_kwargs=None,
    ):
        """Helper to construct a ConcreteBase AgentComponent."""

        class ConcreteBase(AgentComponentBase):
            pass

        kwargs = {
            "name": component_name,
            "flow_id": flow_id,
            "flow_type": flow_type,
            "user": user,
            "inputs": ["context:user_input"],
            "prompt_id": "test_prompt",
            "toolset": mock_toolset,
            "prompt_registry": mock_prompt_registry,
            "internal_event_client": mock_internal_event_client,
            "schema_registry": inline_schema_registry,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return ConcreteBase(**kwargs)

    def test_inline_schema_resolves_response_schema(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        inline_schema_registry,
    ):
        """response_schema_id with no version resolves an inline BaseAgentOutput subclass."""
        inline_schema_id = self.INLINE_SCHEMA_ID
        mock_toolset.__contains__ = lambda _self, name: name != inline_schema_id

        component = self._make_component(
            inline_schema_registry,
            component_name,
            flow_id,
            flow_type,
            user,
            mock_toolset,
            mock_prompt_registry,
            mock_internal_event_client,
            extra_kwargs={"response_schema_id": self.INLINE_SCHEMA_ID},
        )
        assert component._response_schema is not None
        assert issubclass(component._response_schema, BaseAgentOutput)
        assert component._response_schema.tool_title == self.INLINE_SCHEMA_ID

    def test_inline_schema_with_explicit_title(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    ):
        """Inline schema with an explicit title uses that title as tool_title."""
        schema_with_title = {
            "title": "my_custom_tool",
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        registry = InlineResponseSchemaRegistry(ResponseSchemaRegistry())
        registry.register_schema("my_schema_id", schema_with_title)

        mock_toolset.__contains__ = lambda self, name: name != "my_custom_tool"

        class ConcreteBase(AgentComponentBase):
            pass

        component = ConcreteBase(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:user_input"],
            prompt_id="test_prompt",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=registry,
            response_schema_id="my_schema_id",
        )
        assert component._response_schema.tool_title == "my_custom_tool"

    def test_inline_schema_tool_collision_raises(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_prompt_registry,
        mock_internal_event_client,
        inline_schema_registry,
    ):
        """Inline schema tool_title colliding with an existing tool raises ValueError."""
        inline_schema_id = self.INLINE_SCHEMA_ID
        colliding_toolset = Mock(spec=Toolset)
        colliding_toolset.__contains__ = lambda _self, name: name == inline_schema_id

        class ConcreteBase(AgentComponentBase):
            pass

        with pytest.raises(ValueError, match="collides with existing tool"):
            ConcreteBase(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                prompt_id="test_prompt",
                toolset=colliding_toolset,
                prompt_registry=mock_prompt_registry,
                internal_event_client=mock_internal_event_client,
                schema_registry=inline_schema_registry,
                response_schema_id=self.INLINE_SCHEMA_ID,
            )

    def test_no_schema_fields_gives_none_response_schema(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        inline_schema_registry,
    ):
        """Component with no schema fields has _response_schema = None (default text mode)."""
        component = self._make_component(
            inline_schema_registry,
            component_name,
            flow_id,
            flow_type,
            user,
            mock_toolset,
            mock_prompt_registry,
            mock_internal_event_client,
        )
        assert component._response_schema is None


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
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_attach_creates_nodes_with_correct_parameters(  # noqa: PLR0913
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
        ui_role_as,  # pylint: disable=unused-argument  # parametrize value
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
            is_graph_node=True,
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
        assert "conversation_history_key" in agent_call_kwargs
        assert (
            agent_call_kwargs["prompt"]
            == mock_prompt_registry.get_on_behalf.return_value
        )
        assert agent_call_kwargs["inputs"] == inputs
        assert agent_call_kwargs["flow_id"] == flow_id
        assert agent_call_kwargs["flow_type"] == flow_type
        assert agent_call_kwargs["internal_event_client"] == mock_internal_event_client
        # Agent Node UI logging
        assert "ui_history" in agent_call_kwargs
        assert isinstance(agent_call_kwargs["ui_history"], UIHistory)
        assert agent_call_kwargs["ui_history"].events == ui_log_events
        # Wiring guard: attach passes the resolved per-agent limit through.
        assert "max_context_tokens" in agent_call_kwargs
        # In all parametrized cases here the component is not streaming-eligible
        # (requires both ON_AGENT_FINAL_ANSWER and ON_AGENT_REASONING), so
        # invoke_config carries STREAMING_DISABLED_CONFIG.
        assert (
            agent_call_kwargs["invoke_config"]
            == AgentComponentBase.STREAMING_DISABLED_CONFIG
        )

        # Verify ToolNode creation
        mock_tool_node_cls.assert_called_once()
        tool_call_kwargs = mock_tool_node_cls.call_args[1]
        tracker = tool_call_kwargs["tracker"]
        assert tool_call_kwargs["name"] == f"{component_name}#tools"
        assert "conversation_history_key" in tool_call_kwargs
        assert tool_call_kwargs["toolset"] == mock_toolset
        assert tracker._flow_id == flow_id
        assert tracker._flow_type == flow_type
        assert tracker._internal_event_client == mock_internal_event_client
        # Tool Node UI logging
        assert "ui_history" in tool_call_kwargs
        assert isinstance(tool_call_kwargs["ui_history"], UIHistory)
        assert tool_call_kwargs["ui_history"].events == ui_log_events

        # Verify FinalResponseNode creation
        mock_final_response_node_cls.assert_called_once()
        final_call_kwargs = mock_final_response_node_cls.call_args[1]
        assert final_call_kwargs["name"] == f"{component_name}#final_response"
        assert final_call_kwargs["component_name"] == component_name
        assert "conversation_history_key" in final_call_kwargs
        assert "output_key" in final_call_kwargs

        # FinalResponse Node UI logging
        assert "ui_history" in final_call_kwargs
        assert isinstance(final_call_kwargs["ui_history"], UIHistory)
        assert final_call_kwargs["ui_history"].events == ui_log_events


@pytest.mark.usefixtures(
    "mock_agent_node_cls", "mock_tool_node_cls", "mock_final_response_node_cls"
)
class TestAgentComponentAttachEdges:
    """Test suite for AgentComponent routing behavior through graph execution."""

    def test_routing_with_custom_schema_tool_call_goes_to_final_response(
        self,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
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

    def test_routing_with_mixed_tool_calls_prioritizes_final_response(
        self,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_other_tool_call,
        mock_custom_schema_tool_call,
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

    def test_routing_with_without_conversation_history(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
        mock_final_tool_call,
        mock_other_tool_call,
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

        # Test the routing behavior - should raise NotifiableAgentException
        with pytest.raises(NotifiableAgentException) as exc_info:
            router_function(base_flow_state)

        assert (
            "Conversation history not found for key" in exc_info.value.internal_detail
        )
        assert (
            "Conversation history not found for key" in exc_info.value.internal_detail
        )

    def test_routing_with_schema_mode_and_no_tool_calls_raises_error(
        self,
        agent_component_with_custom_schema,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
    ):
        """Test that schema mode with no tool calls raises NotifiableAgentException."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state_with_no_tools = base_flow_state.copy()
        state_with_no_tools[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_message]
        }

        agent_component_with_custom_schema.attach(mock_state_graph, mock_router)

        router_calls = mock_state_graph.add_conditional_edges.call_args_list
        agent_router_call = next(
            call for call in router_calls if call[0][0] == f"{component_name}#agent"
        )
        router_function = agent_router_call[0][1]

        with pytest.raises(NotifiableAgentException) as exc_info:
            router_function(state_with_no_tools)
        assert "Schema mode requires a tool call" in exc_info.value.internal_detail

    def test_routing_with_non_ai_message_raises_error(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
    ):
        """Test that non-AIMessage raises NotifiableAgentException."""
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

        # Test the routing behavior - should raise NotifiableAgentException
        with pytest.raises(NotifiableAgentException) as exc_info:
            router_function(state_with_non_ai_message)

        assert (
            f"Last message is not AIMessage for component {component_name}"
            in exc_info.value.internal_detail
        )

    def test_routing_with_no_tool_calls_goes_to_final_response(
        self,
        agent_component,
        mock_state_graph,
        mock_router,
        base_flow_state,
        component_name,
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
            ("test/schema", "^1.0.0", False),  # Both provided — registry mode, valid
            (None, None, False),  # Neither provided — text-only mode, valid
            ("test/schema", None, False),  # ID only — inline mode, valid
            (None, "^1.0.0", True),  # Version only (no ID) — invalid
        ],
    )
    def test_response_schema_id_and_version_combinations(
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
        """response_schema_id alone = inline mode; both = registry mode; version alone = invalid."""
        mock_schema = mock_schema_registry.get.return_value
        mock_toolset.__contains__ = lambda self, name: name != mock_schema.tool_title

        if should_raise:
            with pytest.raises(ValueError, match="requires response_schema_id"):
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

    @pytest.mark.usefixtures("mock_state_graph", "mock_router")
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


class TestPromptVariableValidation:
    """Tests for validate_prompt_variable_coverage on AgentComponent."""

    def _build(
        self,
        inputs,
        required_vars,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
        strict_validation=False,
    ):
        mock_prompt_registry.get_required_variables.return_value = required_vars
        return AgentComponent(
            name="comp",
            flow_id="test",
            flow_type=flow_type,
            user=user,
            inputs=inputs,
            prompt_id="p",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            strict_validation=strict_validation,
        )

    def test_happy_path(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        comp = self._build(
            ["context:goal"],
            {"goal"},
            mock_prompt_registry,
            mock_internal_event_client,
            flow_type,
            user,
            mock_toolset,
        )
        assert comp is not None

    def test_missing_input_raises(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        with pytest.raises(ValidationError, match="missing input variables"):
            self._build(
                [],
                {"goal"},
                mock_prompt_registry,
                mock_internal_event_client,
                flow_type,
                user,
                mock_toolset,
            )

    def test_extra_input_without_strict_validation(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        """Extra inputs do NOT raise when strict_validation is False (default)."""
        comp = self._build(
            ["context:goal", "context:extra"],
            {"goal"},
            mock_prompt_registry,
            mock_internal_event_client,
            flow_type,
            user,
            mock_toolset,
        )
        assert comp is not None

    def test_extra_input_with_strict_validation(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        """Extra inputs raise when strict_validation is True."""
        with pytest.raises(ValidationError, match="extra input variables"):
            self._build(
                ["context:goal", "context:extra"],
                {"goal"},
                mock_prompt_registry,
                mock_internal_event_client,
                flow_type,
                user,
                mock_toolset,
                strict_validation=True,
            )

    def test_optional_inputs_excluded_from_both_required_and_extra_checks(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        # `project` IS referenced by the template (e.g. `{% if project %}`);
        # `namespace` is NOT referenced by the template.
        mock_prompt_registry.get_required_variables.return_value = {"goal", "project"}

        comp = AgentComponent(
            name="comp",
            flow_id="test",
            flow_type=flow_type,
            user=user,
            inputs=[
                "context:goal",
                {"from": "context:project", "as": "project", "optional": True},
                {"from": "context:namespace", "as": "namespace", "optional": True},
            ],
            prompt_id="p",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            strict_validation=True,
        )
        assert comp is not None

    def test_optional_input_excluded_from_extra_check_in_strict_mode(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        mock_prompt_registry.get_required_variables.return_value = {"goal"}
        comp = AgentComponent(
            name="comp",
            flow_id="test",
            flow_type=flow_type,
            user=user,
            inputs=[
                "context:goal",
                {"from": "context:project", "as": "project", "optional": True},
            ],
            prompt_id="p",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            strict_validation=True,
        )
        assert comp is not None

    def test_missing_required_input_still_raises_when_optional_present(
        self,
        mock_prompt_registry,
        mock_internal_event_client,
        flow_type,
        user,
        mock_toolset,
    ):
        mock_prompt_registry.get_required_variables.return_value = {
            "goal",
            "required_var",
        }
        with pytest.raises(ValidationError, match="missing input variables"):
            AgentComponent(
                name="comp",
                flow_id="test",
                flow_type=flow_type,
                user=user,
                inputs=[
                    "context:goal",
                    {"from": "context:project", "as": "project", "optional": True},
                ],
                prompt_id="p",
                toolset=mock_toolset,
                prompt_registry=mock_prompt_registry,
                internal_event_client=mock_internal_event_client,
            )


class TestAgentComponentCompaction:
    """Test suite for AgentComponent compaction configuration."""

    def test_compaction_default_is_false(
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
        """Test that compaction defaults to False."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )
        assert component.compaction is False

    def test_compaction_accepts_bool_true(
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
        """Test that compaction accepts boolean True."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            compaction=True,
        )
        assert component.compaction is True

    def test_compaction_accepts_config_object(
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
        """Test that compaction accepts CompactionConfig object."""
        config = CompactionConfig(
            max_recent_messages=20,
            trim_threshold=0.8,
        )
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            compaction=config,
        )
        assert component.compaction == config
        assert component.compaction.max_recent_messages == 20
        assert component.compaction.trim_threshold == 0.8

    @pytest.mark.parametrize(
        ("compaction_value", "should_create_compactor"),
        [
            (False, False),
            (True, True),
            (CompactionConfig(), True),
        ],
        ids=["disabled", "enabled_bool", "enabled_config"],
    )
    @pytest.mark.usefixtures("mock_tool_node_cls", "mock_final_response_node_cls")
    def test_attach_creates_compactor_based_on_config(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
        compaction_value,
        should_create_compactor,
    ):
        """Test that attach creates compactor only when compaction is enabled."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            compaction=compaction_value,
        )

        with (
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent.component.AgentNode"
            ) as mock_agent_node_cls,
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent.component.create_conversation_compactor"
            ) as mock_create_compactor,
        ):
            mock_agent_node = Mock()
            mock_agent_node.name = f"{component_name}#agent"
            mock_agent_node_cls.return_value = mock_agent_node

            component.attach(mock_state_graph, mock_router)

            if should_create_compactor:
                mock_create_compactor.assert_called_once()
                agent_call_kwargs = mock_agent_node_cls.call_args[1]
                assert agent_call_kwargs["compactor"] is not None
            else:
                mock_create_compactor.assert_not_called()
                agent_call_kwargs = mock_agent_node_cls.call_args[1]
                assert agent_call_kwargs["compactor"] is None


class TestAgentComponentBindToSupervisor:
    """Test suite for AgentComponent.bind_to_supervisor functionality."""

    def test_bind_to_supervisor_sets_factories(
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
        """Test that bind_to_supervisor sets the key factories."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )

        # Initially keys should have default values
        assert isinstance(component._conversation_history_key, RuntimeIOKey)
        assert isinstance(component._output_key, RuntimeIOKey)
        assert not component._is_bound_to_supervisor

        # Create mock RuntimeIOKey instances
        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        # Bind to supervisor
        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=mock_goal_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # Keys should now be set to the provided values
        assert component._conversation_history_key is mock_history_key
        assert component._output_key is mock_output_key
        assert component._is_bound_to_supervisor
        # goal_key should be appended to inputs so AgentNode receives it transparently
        assert mock_goal_key in component.inputs

    def test_bind_to_supervisor_removes_existing_goal_input_and_appends_new_one(
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
        """Test that bind_to_supervisor removes any pre-existing goal input before appending the new one."""
        existing_goal_key = IOKey(target="context", subkeys=["goal"])
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )
        component.inputs = [existing_goal_key]

        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        new_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=new_goal_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # The old goal key should have been removed and replaced by the new one
        assert existing_goal_key not in component.inputs
        assert new_goal_key in component.inputs

    def test_bind_to_supervisor_requires_description(
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
        """Test that bind_to_supervisor raises ValueError if description is not set."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            # No description
        )

        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        with pytest.raises(ValueError, match="must have a description"):
            component.bind_to_supervisor(
                conversation_history_key=mock_history_key,
                output_key=mock_output_key,
                goal_key=mock_goal_key,
                tool_approval_decision_key=mock_approval_key,
            )

    def test_outputs_returns_empty_when_bound(
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
        """Test that outputs returns empty tuple when bound to supervisor."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )

        # Before binding, outputs should be populated
        assert len(component.outputs) > 0

        # Bind to supervisor
        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())
        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=mock_goal_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # After binding, outputs should be empty
        assert component.outputs == ()

    def test_bind_to_supervisor_sets_session_id_key(
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
        """Test that bind_to_supervisor stores the provided session_id_key."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )

        # Before binding, session_id_key should be NoneIOKey (standalone mode)
        assert isinstance(component._session_id_key, NoneIOKey)

        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_session_id_key = IOKey(
            target="context", subkeys=["supervisor", "active_subsession"], optional=True
        )
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=mock_goal_key,
            session_id_key=mock_session_id_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # session_id_key should now be set to the provided value
        assert component._session_id_key is mock_session_id_key

    def test_bind_to_supervisor_session_id_key_defaults_to_none(
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
        """Test that session_id_key defaults to NoneIOKey when not provided to bind_to_supervisor."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )

        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        # Bind without session_id_key
        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=mock_goal_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # session_id_key should be NoneIOKey (always resolves to None)
        assert isinstance(component._session_id_key, NoneIOKey)

    def test_standalone_tool_approval_decision_key_is_component_scoped(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        base_flow_state,
    ):
        """Test that standalone component uses a component-scoped tool approval decision key."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
        )

        # Standalone: key should be component-scoped (no subsession namespace)
        assert isinstance(component._tool_approval_decision_key, RuntimeIOKey)
        resolved = component._tool_approval_decision_key.to_iokey(base_flow_state)
        assert resolved.target == "context"
        assert resolved.subkeys == [component_name, "tool_approval_decision"]

    def test_bind_to_supervisor_sets_tool_approval_decision_key(
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
        """Test that bind_to_supervisor stores the provided tool_approval_decision_key."""
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            description="Test agent for supervisor",
        )

        mock_history_key = RuntimeIOKey(alias="conversation_history", factory=Mock())
        mock_output_key = RuntimeIOKey(alias="final_answer", factory=Mock())
        mock_goal_key = RuntimeIOKey(alias="goal", factory=Mock())
        mock_approval_key = RuntimeIOKey(alias="tool_approval_decision", factory=Mock())

        component.bind_to_supervisor(
            conversation_history_key=mock_history_key,
            output_key=mock_output_key,
            goal_key=mock_goal_key,
            tool_approval_decision_key=mock_approval_key,
        )

        # tool_approval_decision_key should now be set to the provided value
        assert component._tool_approval_decision_key is mock_approval_key


_AGENT_COMPONENT_MODULE = (
    "duo_workflow_service.agent_platform.v1.components.agent.component"
)


class TestAgentComponentToolApprovalExecutionFlow:
    """Tests for AgentComponent tool-approval routing via a real compiled graph.

    Follows the same pattern as ``TestSupervisorExecutionFlow``: the component
    is attached to a real ``StateGraph``, the graph is compiled, and assertions
    are made on which nodes are actually invoked — rather than extracting and
    calling private router functions directly.
    """

    @pytest.fixture(name="mock_agent_node_cls")
    def mock_agent_node_cls_fixture(self, component_name):
        """Patch AgentNode in the agent component module."""
        with patch(f"{_AGENT_COMPONENT_MODULE}.AgentNode") as mock_cls:
            mock_cls.return_value.name = f"{component_name}#agent"
            yield mock_cls

    @pytest.fixture(name="mock_tool_node_cls")
    def mock_tool_node_cls_fixture(self, component_name):
        """Patch ToolNode in the agent component module."""
        with patch(f"{_AGENT_COMPONENT_MODULE}.ToolNode") as mock_cls:
            mock_cls.return_value.name = f"{component_name}#tools"
            yield mock_cls

    @pytest.fixture(name="mock_final_response_node_cls")
    def mock_final_response_node_cls_fixture(self, component_name):
        """Patch FinalResponseNode in the agent component module."""
        with patch(f"{_AGENT_COMPONENT_MODULE}.FinalResponseNode") as mock_cls:
            mock_cls.return_value.name = f"{component_name}#final_response"
            yield mock_cls

    @pytest.fixture(name="mock_tool_approval_request_node_cls")
    def mock_tool_approval_request_node_cls_fixture(self, component_name):
        """Patch ToolApprovalRequestNode in the agent component module."""
        with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalRequestNode") as mock_cls:
            mock_cls.return_value.name = f"{component_name}#tool_approval_request"
            yield mock_cls

    @pytest.fixture(name="mock_tool_approval_fetch_node_cls")
    def mock_tool_approval_fetch_node_cls_fixture(self, component_name):
        """Patch ToolApprovalFetchNode in the agent component module."""
        with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalFetchNode") as mock_cls:
            mock_cls.return_value.name = f"{component_name}#tool_approval_fetch"
            yield mock_cls

    @pytest.fixture(name="all_node_mocks")
    def all_node_mocks_fixture(
        self,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
        mock_tool_approval_request_node_cls,
        mock_tool_approval_fetch_node_cls,
    ):
        """Activate all node mocks and return them keyed by short name."""
        return {
            "agent": mock_agent_node_cls.return_value,
            "tools": mock_tool_node_cls.return_value,
            "final_response": mock_final_response_node_cls.return_value,
            "tool_approval_request": mock_tool_approval_request_node_cls.return_value,
            "tool_approval_fetch": mock_tool_approval_fetch_node_cls.return_value,
        }

    @pytest.fixture(name="agent_component_with_tool_approval")
    def agent_component_with_tool_approval_fixture(
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
        """AgentComponent with tool approval enabled."""
        return AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:user_input"],
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            require_tool_approval=True,
            pre_approved_tools=[],
        )

    def _compile(self, component, mock_router):
        """Attach component to a real StateGraph with entry at tool_approval_fetch."""
        graph = StateGraph(FlowState)
        component.attach(graph, mock_router)
        # Start execution directly at the fetch node so we exercise its router
        graph.set_entry_point(f"{component.name}#tool_approval_fetch")
        return graph.compile()

    def _state_with_decision(self, base_flow_state, component_name, decision):
        """Return a copy of base_flow_state with the approval decision set in context."""
        return {
            **base_flow_state,
            FlowStateKeys.CONTEXT: {
                **base_flow_state.get(FlowStateKeys.CONTEXT, {}),
                component_name: {"tool_approval_decision": decision},
            },
        }

    def test_approve_decision_routes_to_tools(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        component_name,
        agent_component_with_tool_approval,
    ):
        """APPROVE decision: fetch node routes execution to the tools node."""
        nodes = all_node_mocks
        state = self._state_with_decision(
            base_flow_state, component_name, FlowEventType.APPROVE
        )

        # fetch node returns state unchanged; tools node terminates via agent → final_response → router
        nodes["tool_approval_fetch"].run.return_value = state
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="Done.")]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(agent_component_with_tool_approval, mock_router)
        compiled.invoke(state)

        nodes["tool_approval_fetch"].run.assert_called_once()
        nodes["tools"].run.assert_called_once()

    def test_reject_decision_routes_to_agent(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        component_name,
        agent_component_with_tool_approval,
    ):
        """REJECT decision: fetch node routes execution back to the agent node."""
        nodes = all_node_mocks
        state = self._state_with_decision(
            base_flow_state, component_name, FlowEventType.REJECT
        )

        # fetch node returns state with REJECT decision; agent then produces a final response
        nodes["tool_approval_fetch"].run.return_value = state
        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="Rejected, stopping.")]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(agent_component_with_tool_approval, mock_router)
        compiled.invoke(state)

        nodes["tool_approval_fetch"].run.assert_called_once()
        nodes["agent"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()

    def test_modify_decision_routes_to_agent(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        component_name,
        agent_component_with_tool_approval,
    ):
        """MODIFY decision: fetch node routes execution back to the agent node."""
        nodes = all_node_mocks
        state = self._state_with_decision(
            base_flow_state, component_name, FlowEventType.MODIFY
        )

        nodes["tool_approval_fetch"].run.return_value = state
        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                component_name: [AIMessage(content="Modified, stopping.")]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(agent_component_with_tool_approval, mock_router)
        compiled.invoke(state)

        nodes["tool_approval_fetch"].run.assert_called_once()
        nodes["agent"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()

    def test_missing_decision_raises_routing_error(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        agent_component_with_tool_approval,
    ):
        """Missing approval decision: the fetch router raises RoutingError."""
        nodes = all_node_mocks
        # No approval decision in context
        state = {
            **base_flow_state,
            FlowStateKeys.CONTEXT: {},
        }

        nodes["tool_approval_fetch"].run.return_value = state

        compiled = self._compile(agent_component_with_tool_approval, mock_router)

        with pytest.raises(RoutingError, match="No approval decision found in state"):
            compiled.invoke(state)

    def test_unexpected_decision_raises_routing_error(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        component_name,
        agent_component_with_tool_approval,
    ):
        """Unrecognised approval decision value: the fetch router raises RoutingError."""
        nodes = all_node_mocks
        state = self._state_with_decision(
            base_flow_state, component_name, "unknown_decision"
        )

        nodes["tool_approval_fetch"].run.return_value = state

        compiled = self._compile(agent_component_with_tool_approval, mock_router)

        with pytest.raises(RoutingError, match="Unexpected approval decision"):
            compiled.invoke(state)


# ---------------------------------------------------------------------------
# Tests for _agent_node_invoke_config TAG_NOSTREAM logic
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestAgentNodeInvokeConfig:
    """Tests that attach() passes the correct invoke_config to AgentNode based on ui_log_events."""

    def _attach_and_get_invoke_config(
        self,
        ui_log_events,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
    ):
        component = AgentComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:goal"],
            prompt_id=prompt_id,
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            ui_log_events=ui_log_events,
        )
        with (
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent.component.AgentNode"
            ) as mock_agent_node_cls,
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent.component.ToolNode"
            ),
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent.component.FinalResponseNode"
            ),
        ):
            mock_agent_node_cls.return_value = Mock()
            mock_agent_node_cls.return_value.name = f"{component_name}#agent"
            component.attach(mock_state_graph, mock_router)
            return mock_agent_node_cls.call_args[1]["invoke_config"]

    def test_both_events_declared_passes_streaming_enabled_config(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
    ):
        """Attach() passes STREAMING_ENABLED_CONFIG when both streaming events are declared."""
        invoke_config = self._attach_and_get_invoke_config(
            ui_log_events=[
                UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
                UILogEventsAgent.ON_AGENT_REASONING,
            ],
            component_name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            mock_toolset=mock_toolset,
            mock_prompt_registry=mock_prompt_registry,
            mock_internal_event_client=mock_internal_event_client,
            mock_state_graph=mock_state_graph,
            mock_router=mock_router,
        )
        assert invoke_config == AgentComponentBase.STREAMING_ENABLED_CONFIG

    @pytest.mark.parametrize(
        ("ui_log_events", "case"),
        [
            ([], "no events"),
            ([UILogEventsAgent.ON_AGENT_FINAL_ANSWER], "only ON_AGENT_FINAL_ANSWER"),
            ([UILogEventsAgent.ON_AGENT_REASONING], "only ON_AGENT_REASONING"),
            (
                [
                    UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
                    UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
                ],
                "tool events only",
            ),
        ],
    )
    def test_incomplete_events_pass_streaming_disabled_config(
        self,
        ui_log_events,
        case,  # pylint: disable=unused-argument
        component_name,
        flow_id,
        flow_type,
        user,
        prompt_id,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_state_graph,
        mock_router,
    ):
        """Attach() passes STREAMING_DISABLED_CONFIG when both streaming events are not declared."""
        invoke_config = self._attach_and_get_invoke_config(
            ui_log_events=ui_log_events,
            component_name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id=prompt_id,
            mock_toolset=mock_toolset,
            mock_prompt_registry=mock_prompt_registry,
            mock_internal_event_client=mock_internal_event_client,
            mock_state_graph=mock_state_graph,
            mock_router=mock_router,
        )
        assert invoke_config == AgentComponentBase.STREAMING_DISABLED_CONFIG
