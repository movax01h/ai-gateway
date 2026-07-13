# Re-export UILogEventsSupervisor from v1 to prevent code duplication.
from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)

__all__ = ["UILogEventsSupervisor"]
