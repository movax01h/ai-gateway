from enum import auto

from duo_workflow_service.agent_platform.experimental.ui_log import BaseUILogEvents

__all__ = ["UILogEventsSupervisor"]


class UILogEventsSupervisor(BaseUILogEvents):
    """UI log events for the SupervisorAgentComponent.

    Mirrors the standard agent events.  Supervisor-specific events for delegation and subagent return visibility
    (ON_DELEGATION, ON_SUBAGENT_RETURN) are intentionally omitted until DelegationNode and SubagentReturnNode have
    UIHistory wiring to emit them.
    """

    ON_AGENT_FINAL_ANSWER = auto()
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()
