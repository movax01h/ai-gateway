"""How an agent's nodes know that a turn which stopped or failed is being recovered.

The engine writes ``status = RECOVERY`` when the user's next message arrives, and the nodes that would act on what the
user interrupted yield while it holds: the tool node runs no tool, the approval nodes don't ask, and a subagent hands
control back to its supervisor. The agent that claims the message sets ``EXECUTION`` again, so the status is exactly as
long as the recovery.

Module-level functions rather than methods of the base node: ``BaseAgentComponentNode`` calls ``is_recovering`` to
decide when a node steps aside, and the routers of the component call it too, and they are not nodes.
"""

from duo_workflow_service.agent_platform.v1.state import FlowState, IOKey
from duo_workflow_service.entities import WorkflowStatusEnum

__all__ = ["RECOVERED_STATUS", "is_recovering"]

_STATUS_KEY = IOKey(target="status", optional=True)

# What the agent that claims the message sets, which ends the recovery.
RECOVERED_STATUS = WorkflowStatusEnum.EXECUTION


def is_recovering(state: FlowState) -> bool:
    """Whether the engine is recovering a turn that stopped or failed, and nothing has claimed the message yet.

    Args:
        state: The flow state the node runs on.

    Returns:
        True while ``status`` is ``RECOVERY``, and False for a state that holds no status.
    """
    try:
        return _STATUS_KEY.value_from_state(state) == WorkflowStatusEnum.RECOVERY
    except KeyError:
        # `optional` covers a missing subkey, not a missing channel: a state that never wrote a status is not recovering.
        return False
