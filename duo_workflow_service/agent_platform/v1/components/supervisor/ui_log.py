from enum import auto

from duo_workflow_service.agent_platform.v1.ui_log import (
    BaseUILogEvents,
)

__all__ = ["UILogEventsSupervisor"]


class UILogEventsSupervisor(BaseUILogEvents):
    """UI log events for the SupervisorAgentComponent."""

    ON_AGENT_FINAL_ANSWER = auto()
    ON_AGENT_REASONING = auto()
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()
    ON_DELEGATION = auto()
    ON_DELEGATION_RETURNS = auto()
    ON_DELEGATION_ERROR = auto()
