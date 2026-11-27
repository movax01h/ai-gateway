"""What the nodes of an agent component share: who they are, and whose history they read.

``run`` is the one entry point LangGraph calls, and it is the same for every node. It does nothing but call the node's
``_run``, so that one place can decide what a node does around its work, and each node stays what it was.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langgraph.runtime import Runtime

from duo_workflow_service.agent_platform.core.inbox import RunInbox
from duo_workflow_service.agent_platform.v1.components.agent.nodes._session import (
    DEFAULT_SESSION_ID_KEY,
    resolve_session_id,
)
from duo_workflow_service.agent_platform.v1.state import FlowState, IOKey
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey, RuntimeIOKey

__all__ = ["BaseAgentComponentNode"]


class BaseAgentComponentNode(ABC):
    """A node of an agent component: it names itself and reads one component's history.

    Subclasses implement ``_run``. ``run`` is the one LangGraph calls.

    Args:
        name: LangGraph node name.
        conversation_history_key: ``RuntimeIOKey`` that resolves the conversation-history ``IOKey`` at runtime.
        session_id_key: ``IOKey`` resolving the active subsession ID, so what the node writes is attributed to the
            subagent it runs for.
    """

    def __init__(
        self,
        *,
        name: str,
        conversation_history_key: RuntimeIOKey,
        session_id_key: BaseIOKey = DEFAULT_SESSION_ID_KEY,
    ):
        self.name = name
        self._conversation_history_key = conversation_history_key
        self._session_id_key = session_id_key

    async def run(
        self,
        state: FlowState,
        runtime: Optional[Runtime[Optional[RunInbox]]] = None,
    ) -> Any:
        """Run the node: the one entry point LangGraph calls."""
        return await self._run(state, runtime)

    @abstractmethod
    async def _run(
        self, state: FlowState, runtime: Optional[Runtime[Optional[RunInbox]]]
    ) -> Any:
        """Do the node's work."""

    def _history_iokey(self, state: FlowState) -> IOKey:
        return self._conversation_history_key.to_iokey(state)

    def _session_id(self, state: FlowState) -> Optional[str]:
        return resolve_session_id(self._session_id_key, state)
