import re
from contextlib import contextmanager
from typing import Any, Optional, override
from unittest.mock import Mock, patch

import pytest
from langgraph.graph import StateGraph

from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas.base import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.catalog import CatalogItems
from duo_workflow_service.agent_platform.v1.chat_engine import (
    normalize_engine_owned_config,
)
from duo_workflow_service.agent_platform.v1.chat_engine.graph_builder import (
    ChatGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    AbortComponent,
    BaseComponent,
    EndComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    PartialFlowConfig,
)
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import WorkflowStatusEnum
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

_PARENT_MODULE = "duo_workflow_service.agent_platform.v1.flows.graph_builder"


class _NodeComponent(BaseComponent):
    """One real node per component; the router owns the edge out of it."""

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
    params: dict[str, Any] = {
        "flow": {"entry_point": "agent"},
        "components": [{"name": "agent", "type": "NodeComponent"}],
        "routers": [{"from": "agent", "to": "end"}],
        "environment": "chat-partial",
        "version": "v1",
    }
    params.update(overrides)
    return FlowConfig(**params)


def _components(*names: str, **overrides: Any) -> FlowConfig:
    # The builder never reads the environment; the one-component rule of
    # chat-partial lives in the registry. Multi-component shapes are built
    # under `ambient`.
    return _config(
        environment="ambient",
        components=[{"name": name, "type": "NodeComponent"} for name in names],
        **overrides,
    )


def _conditional(from_component: str, routes: dict[str, str]) -> dict[str, Any]:
    return {
        "from": from_component,
        "condition": {"input": "status", "routes": routes},
    }


@contextmanager
def _node_components():
    with patch(f"{_PARENT_MODULE}.load_component_class", return_value=_NodeComponent):
        yield


def _builder_params(user) -> dict[str, Any]:
    tools_registry = Mock(spec=ToolsRegistry)
    tools_registry.toolset.return_value = Mock(name="toolset")
    tools_registry.mcp_tool_names.return_value = []
    tools_registry.ask_listed_tool_names.return_value = set()
    return {
        "tools_registry": tools_registry,
        "prompt_registry": Mock(spec=BasePromptRegistry),
        "schema_registry": Mock(spec=BaseResponseSchemaRegistry),
        "workflow_id": "workflow-1",
        "workflow_type": GLReportingEventContext.from_workflow_definition("chat"),
        "user": user,
        "internal_event_client": Mock(spec=InternalEventsClient),
        "catalog_items": CatalogItems(),
    }


@pytest.fixture(name="builder")
def builder_fixture(user) -> ChatGraphBuilder:
    return ChatGraphBuilder(**_builder_params(user))


def test_end_writes_input_required_and_abort_is_unchanged(builder):
    with _node_components(), patch(f"{_PARENT_MODULE}.StateGraph") as graph_class:
        builder.build(_config())

    seeded = {
        call.args[0]: call.args[1].__self__
        for call in graph_class.return_value.add_node.call_args_list
        if hasattr(call.args[1], "__self__")
    }
    assert isinstance(seeded["terminate_flow"], EndComponent)
    assert seeded["terminate_flow"].status is WorkflowStatusEnum.INPUT_REQUIRED
    assert isinstance(seeded["abort_flow"], AbortComponent)


def test_declared_routers_attach_exactly_as_declared(builder):
    config = _components(
        "agent",
        "reviewer",
        routers=[
            {"from": "agent", "to": "reviewer"},
            _conditional("reviewer", {"retry": "agent", "default_route": "end"}),
        ],
    )

    with (
        _node_components(),
        patch(f"{_PARENT_MODULE}.StateGraph"),
        patch(f"{_PARENT_MODULE}.Router") as router_class,
    ):
        builder.build(config)

    assert [
        call.kwargs["from_component"].name for call in router_class.call_args_list
    ] == ["agent", "reviewer"]


@pytest.mark.parametrize(
    "config",
    [
        _components(
            "agent",
            "reviewer",
            routers=[
                {"from": "agent", "to": "reviewer"},
                {"from": "reviewer", "to": "end"},
            ],
        ),
        _components(
            "agent",
            "reviewer",
            routers=[
                {"from": "agent", "to": "reviewer"},
                _conditional("reviewer", {"retry": "agent", "default_route": "end"}),
            ],
        ),
        _components("agent", "orphan", routers=[{"from": "agent", "to": "end"}]),
    ],
    ids=[
        "declared route to end",
        "conditional route out of a loop",
        "an unreachable component is not checked",
    ],
)
def test_a_graph_where_every_reachable_component_can_end_is_built(builder, config):
    with (
        _node_components(),
        patch(f"{_PARENT_MODULE}.StateGraph"),
        patch(f"{_PARENT_MODULE}.Router"),
    ):
        builder.build(config)


@pytest.mark.parametrize(
    ("config", "stranded"),
    [
        (_config(routers=[]), "agent"),
        (_config(routers=[{"from": "agent", "to": "abort"}]), "agent"),
        (
            _components(
                "agent",
                "reviewer",
                routers=[
                    {"from": "agent", "to": "reviewer"},
                    _conditional(
                        "reviewer", {"retry": "agent", "default_route": "agent"}
                    ),
                ],
            ),
            "agent, reviewer",
        ),
        (
            _components(
                "agent",
                "reviewer",
                routers=[_conditional("agent", {"review": "reviewer", "done": "end"})],
            ),
            "reviewer",
        ),
        (
            _components(
                "agent",
                "reviewer",
                "checker",
                routers=[
                    _conditional("agent", {"review": "reviewer", "done": "end"}),
                    {"from": "reviewer", "to": "checker"},
                    {"from": "checker", "to": "reviewer"},
                ],
            ),
            "checker, reviewer",
        ),
    ],
    ids=[
        "an unrouted entry",
        "a component that can only abort",
        "a loop with no way out",
        "a dead end behind a conditional route",
        "a trap behind a conditional route",
    ],
)
def test_a_component_that_cannot_reach_end_is_rejected_before_any_router_attaches(
    builder, config, stranded
):
    with (
        _node_components(),
        patch(f"{_PARENT_MODULE}.StateGraph"),
        patch(f"{_PARENT_MODULE}.Router") as router_class,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(f"{stranded} cannot reach `end` from `agent`"),
        ):
            builder.build(config)

    router_class.assert_not_called()


def test_the_router_pass_needs_an_entry_point(builder):
    """``build`` refuses this first; the walk needs a start, so the pass refuses it too."""
    config = _config(flow={})

    with pytest.raises(ValueError, match="entry_point is not defined"):
        builder._build_routers(config, {}, Mock(spec=StateGraph))


@pytest.mark.asyncio
async def test_a_turn_ends_at_the_boundary_with_input_required(builder):
    """The graph LangGraph actually runs: entry component, declared hop, terminal write."""
    with _node_components():
        graph = builder.build(_config())

    assert {"agent_entry_node", "terminate_flow", "abort_flow"} <= set(graph.nodes)

    result = await graph.compile().ainvoke(
        FlowState(
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={},
            ui_chat_log=[],
            context={},
        )
    )

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED.value


@pytest.mark.asyncio
async def test_a_completed_chat_partial_config_builds_and_ends_its_turn(builder):
    """Completion writes the route the builder requires: omitting it would be rejected."""
    authored = PartialFlowConfig(
        version="v1",
        environment="chat-partial",
        components=[{"name": "agent", "type": "NodeComponent"}],
    )

    with _node_components():
        graph = builder.build(normalize_engine_owned_config(authored))

    result = await graph.compile().ainvoke(
        FlowState(
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={},
            ui_chat_log=[],
            context={},
        )
    )

    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED.value
