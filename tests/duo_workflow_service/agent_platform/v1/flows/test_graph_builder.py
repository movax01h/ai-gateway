"""Unit tests for ``FlowGraphBuilder``, exercised without a ``Flow``.

``Flow`` behaviour that happens to run the builder is covered in ``test_base.py``.
What is covered here is the builder's own contract: it holds a run's dependencies
once and turns any config handed to it into a ``StateGraph``, which is what makes a
flow buildable without being a ``Flow``.
"""

from contextlib import contextmanager
from typing import Any, Callable, Optional, override
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langgraph.graph import StateGraph

from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas.base import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import FlowGraphBuilder
from duo_workflow_service.agent_platform.v1.routers.router import Router
from duo_workflow_service.components.tools_registry import ToolsRegistry
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

_MODULE = "duo_workflow_service.agent_platform.v1.flows.graph_builder"


class _ComponentClassStub:
    """Stands in for a component class: records the params it was called with.

    The builder resolves component classes through ``load_component_class``, so a
    stub is enough to assert what the builder passes to a component's constructor
    without pulling a real component's dependencies into the test.
    """

    def __init__(self, component: Any):
        self._component = component
        self.params: dict[str, Any] = {}

    def __call__(self, **params: Any) -> Any:
        self.params = params
        return self._component


def _component_stub(name: str) -> MagicMock:
    component = MagicMock(spec=BaseComponent)
    component.__entry_hook__.return_value = f"{name}_entry_node"
    return component


class _NodeComponent(BaseComponent):
    """A real component contributing one real node, for the unpatched-graph test.

    Mirrors what every shipped component does in ``attach``: add its node, then let
    the router own the conditional edge out of it.
    """

    @override
    def __entry_hook__(self) -> str:
        return f"{self.name}_entry_node"

    @override
    def attach(
        self, graph: StateGraph, router: Optional[RouterProtocol] = None
    ) -> None:
        graph.add_node(self.__entry_hook__(), lambda _state: {})
        if router is not None:
            graph.add_conditional_edges(self.__entry_hook__(), router.route)


def _config(**overrides: Any) -> FlowConfig:
    """A minimal single-component config, overridable per test."""
    params: dict[str, Any] = {
        "flow": {"entry_point": "agent"},
        "components": [{"name": "agent", "type": "AgentComponent"}],
        "routers": [{"from": "agent", "to": "end"}],
        "environment": "ambient",
        "version": "v1",
    }
    params.update(overrides)
    return FlowConfig(**params)


@contextmanager
def _component_classes(classes: dict[str, Callable[..., Any]]):
    """Patch ``load_component_class`` to resolve from ``classes`` by type name."""
    with patch(
        f"{_MODULE}.load_component_class", side_effect=lambda name: classes[name]
    ):
        yield


class TestFlowGraphBuilder:
    @pytest.fixture(name="flow_type")
    def flow_type_fixture(self) -> GLReportingEventContext:
        return GLReportingEventContext.from_workflow_definition("chat")

    @pytest.fixture(name="mock_tools_registry")
    def mock_tools_registry_fixture(self) -> Mock:
        tools_registry = Mock(spec=ToolsRegistry)
        tools_registry.toolset.return_value = Mock(name="toolset")
        tools_registry.mcp_tool_names.return_value = ["mcp_tool"]
        tools_registry.ask_listed_tool_names.return_value = set()
        return tools_registry

    @pytest.fixture(name="mock_prompt_registry")
    def mock_prompt_registry_fixture(self) -> Mock:
        return Mock(spec=BasePromptRegistry)

    @pytest.fixture(name="mock_schema_registry")
    def mock_schema_registry_fixture(self) -> Mock:
        return Mock(spec=BaseResponseSchemaRegistry)

    @pytest.fixture(name="mock_internal_event_client")
    def mock_internal_event_client_fixture(self) -> Mock:
        return Mock(spec=InternalEventsClient)

    @pytest.fixture(name="builder")
    def builder_fixture(
        self,
        mock_tools_registry,
        mock_prompt_registry,
        mock_schema_registry,
        mock_internal_event_client,
        flow_type,
        user,
    ) -> FlowGraphBuilder:
        return FlowGraphBuilder(
            tools_registry=mock_tools_registry,
            prompt_registry=mock_prompt_registry,
            schema_registry=mock_schema_registry,
            workflow_id="test-workflow-123",
            workflow_type=flow_type,
            user=user,
            internal_event_client=mock_internal_event_client,
        )

    @pytest.fixture(name="mock_state_graph_class")
    def mock_state_graph_class_fixture(self):
        with patch(f"{_MODULE}.StateGraph") as state_graph_class:
            yield state_graph_class

    @pytest.fixture(name="mock_graph")
    def mock_graph_fixture(self, mock_state_graph_class) -> Mock:
        graph = Mock(spec=StateGraph)
        mock_state_graph_class.return_value = graph
        return graph

    def test_build_raises_when_entry_point_is_missing(self, builder, mock_graph):
        config = _config(flow={})

        with pytest.raises(
            ValueError, match="entry_point is not defined in the flow config"
        ):
            builder.build(config)

        mock_graph.set_entry_point.assert_not_called()

    def test_build_returns_the_graph_with_the_entry_component_hooked_up(
        self, builder, mock_graph
    ):
        agent = _component_stub("agent")

        with (
            _component_classes({"AgentComponent": _ComponentClassStub(agent)}),
            patch(f"{_MODULE}.Router"),
        ):
            graph = builder.build(_config())

        assert graph is mock_graph
        mock_graph.set_entry_point.assert_called_once_with("agent_entry_node")

    def test_it_builds_a_real_graph_that_compiles(self, builder):
        """The extraction's actual claim, with nothing patched out.

        Every other test here patches ``StateGraph`` and ``Router`` and asserts call
        shapes, which cannot catch a node name or an edge that LangGraph itself
        rejects. This one lets both run for real, so "a config becomes a graph
        without a ``Flow``" is verified rather than assumed.
        """
        with _component_classes({"NodeComponent": _NodeComponent}):
            graph = builder.build(
                _config(components=[{"name": "agent", "type": "NodeComponent"}])
            )

        assert {"agent_entry_node", "terminate_flow", "abort_flow"} <= set(graph.nodes)

        graph.compile()

    def test_build_attaches_the_terminal_components(self, builder, mock_graph):
        """``end`` and ``abort`` are implicit: no config declares them."""
        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(_config())

        attached_nodes = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "terminate_flow" in attached_nodes
        assert "abort_flow" in attached_nodes

    @pytest.mark.usefixtures("mock_graph")
    def test_component_receives_the_builders_dependencies(
        self,
        builder,
        mock_prompt_registry,
        mock_schema_registry,
        flow_type,
        user,
    ):
        agent_class = _ComponentClassStub(_component_stub("agent"))

        with (
            _component_classes({"AgentComponent": agent_class}),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(_config())

        assert agent_class.params["prompt_registry"] is mock_prompt_registry
        assert agent_class.params["schema_registry"] is mock_schema_registry
        assert agent_class.params["flow_id"] == "test-workflow-123"
        assert agent_class.params["flow_type"] == flow_type
        assert agent_class.params["user"] is user
        assert agent_class.params["environment"] == "ambient"
        # ``type`` selects the class; it is not a constructor param.
        assert "type" not in agent_class.params
        assert agent_class.params["name"] == "agent"

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "comp_type,expects_built_components",
        [("AgentComponent", True), ("DeterministicStepComponent", False)],
    )
    def test_only_agent_components_are_handed_the_shared_components_dict(
        self, builder, comp_type, expects_built_components
    ):
        """The ``AgentComponent`` factory needs it to resolve subagent references."""
        comp_class = _ComponentClassStub(_component_stub("agent"))

        with _component_classes({comp_type: comp_class}), patch(f"{_MODULE}.Router"):
            builder.build(_config(components=[{"name": "agent", "type": comp_type}]))

        assert ("_built_components" in comp_class.params) is expects_built_components

    @pytest.mark.usefixtures("mock_graph")
    def test_duplicate_component_name_raises(self, builder):
        config = _config(
            components=[
                {"name": "agent", "type": "AgentComponent"},
                {"name": "agent", "type": "DeterministicStepComponent"},
            ],
        )

        with (
            _component_classes(
                {
                    "AgentComponent": _ComponentClassStub(_component_stub("agent")),
                    "DeterministicStepComponent": _ComponentClassStub(
                        _component_stub("other")
                    ),
                }
            ),
            patch(f"{_MODULE}.Router"),
        ):
            with pytest.raises(ValueError, match="Duplicate component name: 'agent'"):
                builder.build(config)

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "ask_listed,expected",
        [
            ({"run_command"}, ["read_file"]),
            (set(), ["read_file", "run_command"]),
        ],
        ids=[
            "an_ask_rule_forces_a_listed_tool_to_prompt",
            "nothing_to_strip",
        ],
    )
    def test_pre_approved_tools_strip_ask_listed_entries(
        self, builder, mock_tools_registry, ask_listed, expected
    ):
        """The flow config's pre-approval list is reduced where it enters.

        Stripping here, rather than at each approval decision, is what stops a flow author pre-approving a tool an
        admin `ask` rule forces to prompt.
        """
        mock_tools_registry.ask_listed_tool_names.return_value = ask_listed

        comp_class = _ComponentClassStub(_component_stub("agent"))
        comp_config: dict[str, Any] = {
            "name": "agent",
            "type": "AgentComponent",
            "pre_approved_tools": ["read_file", "run_command"],
        }

        with (
            _component_classes({"AgentComponent": comp_class}),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(_config(components=[comp_config]))

        mock_tools_registry.ask_listed_tool_names.assert_called_once_with(
            ["read_file", "run_command"]
        )
        assert comp_class.params["pre_approved_tools"] == expected

    @pytest.mark.usefixtures("mock_graph")
    def test_a_component_without_pre_approved_tools_is_left_alone(
        self, builder, mock_tools_registry
    ):
        """Components that declare no pre-approval list are untouched."""
        comp_class = _ComponentClassStub(_component_stub("agent"))

        with (
            _component_classes({"AgentComponent": comp_class}),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(_config())

        mock_tools_registry.ask_listed_tool_names.assert_not_called()
        assert "pre_approved_tools" not in comp_class.params

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "toolset,tool_name,expected_call",
        [
            (None, "read_file", call(["read_file"])),
            (
                ["create_file_with_contents"],
                "read_file",
                call(["create_file_with_contents"], tool_options={}),
            ),
            (
                ["create_file_with_contents"],
                None,
                call(["create_file_with_contents"], tool_options={}),
            ),
            (None, None, None),
        ],
        ids=[
            "tool_name_without_toolset",
            "toolset_wins_over_tool_name",
            "toolset_without_tool_name",
            "neither_declared",
        ],
    )
    def test_a_toolset_is_resolved_from_toolset_or_tool_name(
        self,
        builder,
        mock_tools_registry,
        toolset,
        tool_name,
        expected_call,
    ):
        """The two are alternatives rather than additive, and ``toolset`` wins.

        The shapes differ on purpose: the ``tool_name`` shorthand cannot carry
        per-tool options, so it resolves without a ``tool_options`` argument.
        """
        comp_config: dict[str, Any] = {"name": "agent", "type": "AgentComponent"}
        if toolset is not None:
            comp_config["toolset"] = toolset
        if tool_name is not None:
            comp_config["tool_name"] = tool_name

        comp_class = _ComponentClassStub(_component_stub("agent"))

        with (
            _component_classes({"AgentComponent": comp_class}),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(_config(components=[comp_config]))

        if expected_call is None:
            mock_tools_registry.toolset.assert_not_called()
            assert "toolset" not in comp_class.params
        else:
            assert mock_tools_registry.toolset.call_args_list == [expected_call]
            assert (
                comp_class.params["toolset"] is mock_tools_registry.toolset.return_value
            )

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "toolset_config,expected_names,expected_options",
        [
            (["read_file", "write_file"], ["read_file", "write_file"], {}),
            (
                ["read_file", {"run_command": {"allowed": ["ls"]}}],
                ["read_file", "run_command"],
                {"run_command": {"allowed": ["ls"]}},
            ),
            # An entry with empty options contributes the name only.
            ([{"read_file": None}], ["read_file"], {}),
        ],
    )
    def test_toolset_entries_are_split_into_names_and_options(
        self,
        builder,
        mock_tools_registry,
        toolset_config,
        expected_names,
        expected_options,
    ):
        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(
                _config(
                    components=[
                        {
                            "name": "agent",
                            "type": "AgentComponent",
                            "toolset": toolset_config,
                        }
                    ]
                )
            )

        mock_tools_registry.toolset.assert_called_once_with(
            expected_names, tool_options=expected_options
        )

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "environment,mcp_names,expected_names",
        [
            ("chat", ["mcp_a", "mcp_b"], ["read_file", "mcp_a", "mcp_b"]),
            ("chat", [], ["read_file"]),
            ("chat-partial", ["mcp_a"], ["read_file", "mcp_a"]),
            ("ambient", ["mcp_a", "mcp_b"], ["read_file"]),
            ("ambient", [], ["read_file"]),
        ],
        ids=[
            "chat_with_mcp_tools",
            "chat_without_mcp_tools",
            "chat_partial_with_mcp_tools",
            "ambient_with_mcp_tools",
            "ambient_without_mcp_tools",
        ],
    )
    def test_mcp_tools_are_appended_only_for_chat_environments(
        self,
        builder,
        mock_tools_registry,
        environment,
        mcp_names,
        expected_names,
    ):
        mock_tools_registry.mcp_tool_names.return_value = mcp_names

        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(
                _config(
                    environment=environment,
                    components=[
                        {
                            "name": "agent",
                            "type": "AgentComponent",
                            "toolset": ["read_file"],
                        }
                    ],
                )
            )

        mock_tools_registry.toolset.assert_called_once_with(
            expected_names, tool_options={}
        )

    @pytest.mark.usefixtures("mock_graph")
    def test_a_component_whose_subagents_are_not_built_yet_is_deferred(self, builder):
        """The supervisor is declared first but must be built after its subagent."""
        build_order: list[str] = []

        def _record(name: str, component: Any) -> Callable[..., Any]:
            def factory(**_params: Any) -> Any:
                build_order.append(name)
                return component

            return factory

        supervisor = _component_stub("supervisor")
        supervisor.subagent_components = []

        config = _config(
            flow={"entry_point": "supervisor"},
            components=[
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "subagents": [{"name": "agent"}],
                },
                {"name": "agent", "type": "AgentComponent"},
            ],
            routers=[{"from": "supervisor", "to": "end"}],
        )

        with (
            _component_classes(
                {
                    "SupervisorAgentComponent": _record("supervisor", supervisor),
                    "AgentComponent": _record("agent", _component_stub("agent")),
                }
            ),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(config)

        assert build_order == ["agent", "supervisor"]

    @pytest.mark.usefixtures("mock_graph")
    def test_a_malformed_subagents_entry_names_the_offending_component(self, builder):
        config = _config(
            flow={"entry_point": "supervisor"},
            components=[
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "subagents": ["agent"],  # must be a dict with a "name" key
                },
            ],
            routers=[{"from": "supervisor", "to": "end"}],
        )

        with (
            _component_classes(
                {
                    "SupervisorAgentComponent": _ComponentClassStub(
                        _component_stub("supervisor")
                    )
                }
            ),
            patch(f"{_MODULE}.Router"),
        ):
            with pytest.raises(
                ValueError,
                match="Component 'supervisor' has a malformed subagents entry",
            ):
                builder.build(config)

    @pytest.mark.usefixtures("mock_graph")
    def test_subagents_consumed_by_a_supervisor_are_not_routable_components(
        self, builder
    ):
        """A consumed subagent is owned by its supervisor, so routing to it must fail.

        The ``KeyError`` is incidental: unresolved component references are bare dict
        lookups, carried over unchanged from ``Flow``. What is being pinned is that
        the name is gone from the routable pool, not the exception type. Giving those
        lookups a descriptive ``ValueError`` would be an improvement, not a regression
        against this test.
        """
        supervisor = _component_stub("supervisor")
        supervisor.subagent_components = ["agent"]

        config = _config(
            flow={"entry_point": "supervisor"},
            components=[
                {"name": "agent", "type": "AgentComponent"},
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "subagents": [{"name": "agent"}],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
        )

        with (
            _component_classes(
                {
                    "AgentComponent": _ComponentClassStub(_component_stub("agent")),
                    "SupervisorAgentComponent": _ComponentClassStub(supervisor),
                }
            ),
            patch(f"{_MODULE}.Router"),
        ):
            with pytest.raises(KeyError, match="agent"):
                builder.build(config)

    @pytest.mark.usefixtures("mock_graph")
    def test_conditional_router_carries_the_tracking_params(
        self, builder, flow_type, mock_internal_event_client
    ):
        config = _config(
            components=[
                {"name": "agent", "type": "AgentComponent"},
            ],
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": "status",
                        "routes": {"Execution": "agent", "default_route": "end"},
                    },
                }
            ],
        )

        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router") as mock_router_class,
        ):
            mock_router_class.return_value = Mock(spec=Router)

            builder.build(config)

        call_kwargs = mock_router_class.call_args.kwargs
        assert call_kwargs["flow_id"] == "test-workflow-123"
        assert call_kwargs["flow_type"] == flow_type
        assert call_kwargs["internal_event_client"] is mock_internal_event_client
        assert call_kwargs["input"] == "status"

    @pytest.mark.usefixtures("mock_graph")
    @pytest.mark.parametrize(
        "condition_input",
        ["status", {"from": "context:status", "optional": True}],
    )
    def test_conditional_router_accepts_a_string_or_a_mapping_input(
        self, builder, condition_input
    ):
        config = _config(
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": condition_input,
                        "routes": {"default_route": "end"},
                    },
                }
            ],
        )

        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router") as mock_router_class,
        ):
            mock_router_class.return_value = Mock(spec=Router)

            builder.build(config)

        assert mock_router_class.call_args.kwargs["input"] == condition_input

    @pytest.mark.usefixtures("mock_graph")
    def test_conditional_router_rejects_any_other_input_shape(self, builder):
        config = _config(
            routers=[
                {
                    "from": "agent",
                    "condition": {
                        "input": ["status"],
                        "routes": {"default_route": "end"},
                    },
                }
            ],
        )

        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router"),
        ):
            with pytest.raises(
                ValueError, match="Router input must be a string or a mapping"
            ):
                builder.build(config)

    @pytest.mark.usefixtures("mock_graph")
    def test_unconditional_router_is_built_without_tracking_params(self, builder):
        """A plain ``from``/``to`` router emits no routing events, so it needs none."""
        with (
            _component_classes(
                {"AgentComponent": _ComponentClassStub(_component_stub("agent"))}
            ),
            patch(f"{_MODULE}.Router") as mock_router_class,
        ):
            mock_router_class.return_value = Mock(spec=Router)

            builder.build(_config())

        call_kwargs = mock_router_class.call_args.kwargs
        assert set(call_kwargs) == {"from_component", "to_component"}

    def test_one_builder_builds_more_than_one_config(
        self, builder, mock_graph, mock_state_graph_class
    ):
        """The point of the extraction: a parent's builder can build a child's graph."""
        first = _config()
        second = _config(
            flow={"entry_point": "reviewer"},
            components=[{"name": "reviewer", "type": "AgentComponent"}],
            routers=[{"from": "reviewer", "to": "end"}],
        )

        with (
            _component_classes(
                {
                    "AgentComponent": lambda **params: _component_stub(params["name"]),
                }
            ),
            patch(f"{_MODULE}.Router"),
        ):
            builder.build(first)
            builder.build(second)

        assert mock_state_graph_class.call_count == 2
        assert [call.args[0] for call in mock_graph.set_entry_point.call_args_list] == [
            "agent_entry_node",
            "reviewer_entry_node",
        ]
