"""What the nodes of an agent component share: who they are, whose history they read, and what they do on a recovery.

A turn that stopped or failed is recovered with ``status = RECOVERY``, and every node but the one that can end the
recovery steps aside until it does: the tool node runs no tool, the approval nodes ask for nothing, and a final response
doesn't finish a turn nobody has answered. The agent node is the one that can end it, because it claims the message
that recovers the turn. So the base decides when a node steps aside, in one place, and each node says what stepping
aside means for it.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langgraph.runtime import Runtime

from duo_workflow_service.agent_platform.core.inbox import RunInbox
from duo_workflow_service.agent_platform.v1.components.agent.nodes._recovery import (
    is_recovering,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes._session import (
    DEFAULT_SESSION_ID_KEY,
    resolve_session_id,
)
from duo_workflow_service.agent_platform.v1.state import FlowState, IOKey
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey, RuntimeIOKey

__all__ = ["BaseAgentComponentNode"]


class BaseAgentComponentNode(ABC):
    """A node of an agent component: it names itself, reads one component's history, and steps aside on a recovery.

    Subclasses implement ``_run`` and ``_skip_for_recovery``. ``run`` is the one LangGraph calls, and it is the same for
    all of them, so the rule for when a node steps aside can't differ from one node to the next.

    Args:
        name: LangGraph node name.
        conversation_history_key: ``RuntimeIOKey`` that resolves the conversation-history ``IOKey`` at runtime.
        session_id_key: ``IOKey`` resolving the active subsession ID, so what the node writes is attributed to the
            subagent it runs for.

    Attributes:
        is_subagent: Whether a supervisor bound or compiled the component for itself, which the component says once it
            knows. A node whose behaviour differs for a subagent reads it.
    """

    is_subagent: bool = False

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

    @property
    def _ends_recovery(self) -> bool:
        """Whether this node can end a recovery, which is why it doesn't step aside for one.

        Only the agent node claims the message that recovers the turn. A subagent's agent node leaves that message to
        its supervisor, so it steps aside too.
        """
        return False

    async def run(
        self,
        state: FlowState,
        runtime: Optional[Runtime[Optional[RunInbox]]] = None,
    ) -> Any:
        """Step aside while the turn is being recovered, unless this node is the one that ends the recovery."""
        if is_recovering(state) and not self._ends_recovery:
            return self._skip_for_recovery(state)
        return await self._run(state, runtime)

    @abstractmethod
    async def _run(
        self, state: FlowState, runtime: Optional[Runtime[Optional[RunInbox]]]
    ) -> Any:
        """Do the node's work.

        The turn isn't being recovered, or this node ends the recovery.
        """

    @abstractmethod
    def _skip_for_recovery(self, state: FlowState) -> Any:
        """What the node writes instead of its work, while the turn is being recovered.

        May be nothing.
        """

    def _history_iokey(self, state: FlowState) -> IOKey:
        return self._conversation_history_key.to_iokey(state)

    def _session_id(self, state: FlowState) -> Optional[str]:
        return resolve_session_id(self._session_id_key, state)
