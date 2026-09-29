from typing import Any, override

from duo_workflow_service.agent_platform.v1.chat_engine.graph_builder import (
    ChatGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import FlowGraphBuilder
from duo_workflow_service.components.tools_registry import ToolsRegistry

__all__ = ["ChatFlow"]


class ChatFlow(Flow):
    """The chat engine: a boundary policy over the shared executor for chat-partial flows.

    ``Flow`` and ``FlowGraphBuilder`` run the turn, and every component is the
    shared class ambient flows use. ``ChatFlow`` owns the conversational
    boundary and nothing else, through four seams of the parent: the graph
    builder (where a turn ends), the entry dispatch strategy (where it begins),
    the entry wiring (what crosses the line inbound), and the graph input
    (what happens after a bad crossing). Configs reach it already normalized
    with the chat-surface defaults, selected by the registry's engine table.

    See ``docs/flow_registry/chat_engine.md``.
    """

    def __init__(self, *, config: FlowConfig, **kwargs: Any):
        # Keyword-only: the registry binds ``config`` through ``functools.partial``
        # and the server passes every other argument by keyword.
        super().__init__(config=config, **kwargs)

    @override
    def _graph_builder(self, tools_registry: ToolsRegistry) -> FlowGraphBuilder:
        return ChatGraphBuilder(
            tools_registry=tools_registry,
            prompt_registry=self._flow_prompt_registry,
            schema_registry=self._flow_schema_registry,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            user=self._user,
            internal_event_client=self._internal_event_client,
            catalog_items=self._catalog_items,
        )
