from typing import Any

import structlog

from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    FlowConfigMetadata,
)

__all__ = ["ENGINE_FLOOR_UI_LOG_EVENTS", "normalize_engine_owned_config"]

logger = structlog.stdlib.get_logger(__name__)

AGENT_COMPONENT_TYPE = "AgentComponent"

# The user-facing events legacy chat emits on every turn: the final answer and
# the tool cards. An engine-owned config that declares nothing gets this floor.
ENGINE_FLOOR_UI_LOG_EVENTS: tuple[str, ...] = (
    UILogEventsAgent.ON_AGENT_FINAL_ANSWER.value,
    UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS.value,
    UILogEventsAgent.ON_TOOL_EXECUTION_FAILED.value,
)


def normalize_engine_owned_config(config: FlowConfig) -> FlowConfig:
    """Fill the chat-surface defaults into an engine-owned config before it reaches the engine.

    Authors write the v1 config they write today. The engine adds no fields and
    removes none from the schema; the defaults the chat surface guarantees are
    filled here, at load time, so ``ChatFlow`` and the shared graph builder see
    a complete config:

    ``ui_log_events`` on each ``AgentComponent`` defaults to the floor legacy
    chat emits. A declared list wins, including an empty one.
    ``require_tool_approval`` defaults to ``True``, and a declared value wins.
    ``pre_approved_tools`` is not applied on the engine, so a declared list is
    dropped and logged.

    ``routers`` defaults to an empty list and ``flow.entry_point`` to the single
    component. The graph builder requires both, and the chat-partial environment
    lets authors omit them.

    Args:
        config: A chat-partial config. Every ``AgentComponent`` in it receives
            the defaults; other component types pass through untouched.

    Returns:
        A new config. The input is not mutated.
    """
    components = [
        _normalize_agent_component(component) for component in config.components
    ]
    update: dict[str, Any] = {"components": components}

    if config.routers is None:
        update["routers"] = []

    if config.flow is None or config.flow.entry_point is None:
        update["flow"] = FlowConfigMetadata(
            entry_point=components[0]["name"],
            inputs=config.flow.inputs if config.flow else None,
        )

    return config.model_copy(update=update)


def _normalize_agent_component(component: dict[str, Any]) -> dict[str, Any]:
    if component.get("type") != AGENT_COMPONENT_TYPE:
        return component

    normalized = dict(component)
    normalized.setdefault("ui_log_events", list(ENGINE_FLOOR_UI_LOG_EVENTS))
    normalized.setdefault("require_tool_approval", True)

    if "pre_approved_tools" in normalized:
        dropped_tools = normalized.pop("pre_approved_tools")
        logger.info(
            "Dropping pre_approved_tools from an engine-owned config; the engine does not apply it",
            component=normalized.get("name"),
            pre_approved_tools=dropped_tools,
        )

    return normalized
