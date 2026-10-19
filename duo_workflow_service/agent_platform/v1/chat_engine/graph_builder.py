from typing import override

from langgraph.graph import StateGraph

from duo_workflow_service.agent_platform.v1.components.base import (
    AbortComponent,
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import FlowGraphBuilder
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = ["ChatGraphBuilder"]

_TERMINAL_COMPONENT_NAMES = frozenset({"end", "abort"})


class ChatGraphBuilder(FlowGraphBuilder):
    """``FlowGraphBuilder`` with the chat engine's boundary: where a turn ends.

    ``end`` is seeded to write ``INPUT_REQUIRED``, so reaching it parks the
    session at the boundary instead of completing it. Declared routers attach
    exactly as they do in ambient and the builder adds none, so every turn has
    to be able to reach ``end`` through the routes the config declares.

    See ``docs/flow_registry/chat_engine.md``, "Where a turn ends".
    """

    @override
    def _seed_terminal_components(self, graph: StateGraph) -> dict[str, BaseComponent]:
        # ``abort`` is restated because the parent attaches both terminals in
        # one method and LangGraph does not allow replacing an attached node.
        end_component = EndComponent(
            status=WorkflowStatusEnum.INPUT_REQUIRED,
            **self._terminal_component_params("end"),
        )
        end_component.attach(graph)

        abort_component = AbortComponent(**self._terminal_component_params("abort"))
        abort_component.attach(graph)

        return {"end": end_component, "abort": abort_component}

    @override
    def _build_routers(
        self,
        flow_config: FlowConfig,
        components: dict[str, BaseComponent],
        graph: StateGraph,
    ) -> None:
        """Attach the declared routers once every turn is known to be able to end.

        A component reachable from the entry that cannot reach ``end`` would
        leave a turn with no boundary: a dead end, or a loop with no way out.
        The graph is rejected before any router attaches.
        """
        entry_point = flow_config.flow.entry_point
        if entry_point is None:
            raise ValueError(
                "Can not build flow graph: entry_point is not defined in the flow config."
            )

        edges = _declared_edges(flow_config.routers)
        can_end = _reachable("end", _reversed(edges))
        stranded = sorted(
            _reachable(entry_point, edges) - can_end - _TERMINAL_COMPONENT_NAMES
        )
        if stranded:
            raise ValueError(
                f"The chat flow graph cannot end every turn: {', '.join(stranded)} "
                f"cannot reach `end` from `{entry_point}`. Every component reachable "
                "from the entry has to route to `end`, directly or through other "
                "components."
            )

        super()._build_routers(flow_config, components, graph)


def _declared_edges(routers: list[dict]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for router in routers:
        if "condition" in router:
            targets = set(router["condition"]["routes"].values())
        else:
            targets = {router["to"]}
        edges.setdefault(router["from"], set()).update(targets)
    return edges


def _reversed(edges: dict[str, set[str]]) -> dict[str, set[str]]:
    reversed_edges: dict[str, set[str]] = {}
    for source, targets in edges.items():
        for target in targets:
            reversed_edges.setdefault(target, set()).add(source)
    return reversed_edges


def _reachable(start: str, edges: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier.extend(edges.get(name, ()))
    return seen
