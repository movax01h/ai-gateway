"""Subsession attribution shared by agent nodes.

A module-level function: it is a pure lookup that needs no node state. ``BaseAgentComponentNode`` calls it for the
nodes that share that base, and the experimental components re-export it for theirs.
"""

from typing import Optional

from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey, NoneIOKey

__all__ = ["DEFAULT_SESSION_ID_KEY", "resolve_session_id"]

# For components that are not subagents. Safe to share: BaseIOKey is frozen.
DEFAULT_SESSION_ID_KEY: BaseIOKey = NoneIOKey(alias="session_id")


def resolve_session_id(key: BaseIOKey, state: FlowState) -> Optional[str]:
    """Active subsession ID as a string, or ``None`` when not a subagent.

    Subsession IDs are ints assigned by the supervisor and ``0`` is a real ID,
    so absence is tested against ``None`` rather than falsiness.
    """
    value = key.value_from_state(state)
    return str(value) if value is not None else None
