"""Compiling one component's body as a standalone graph.

A fanned-out branch runs a single component against its own ``FlowState``,
so that body has to be self-contained rather than wired into a parent flow.
"""

from typing import Annotated

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent as V1BaseComponent,
)

__all__ = ["TerminalRouter", "compile_as_unit"]


class TerminalRouter:
    """``RouterProtocol`` ending the graph: a standalone body has no successor."""

    def attach(self, graph: StateGraph) -> None:
        """Nothing to wire -- ``END`` is a node of every graph already."""

    def route(  # pylint: disable=unused-argument
        self, state: FlowState
    ) -> Annotated[str, "Next node"]:
        return END


def compile_as_unit(
    component: BaseComponent | V1BaseComponent,
) -> CompiledStateGraph:
    """Compile ``component``'s body as an independently invocable graph.

    Generic across component types: it touches only ``attach`` and
    ``__entry_hook__``, declared on both base classes of the union.
    """
    graph = StateGraph(FlowState)
    component.attach(graph, TerminalRouter())
    graph.set_entry_point(component.__entry_hook__())
    return graph.compile()
