# Re-export delegate_task entities from v1 to prevent code duplication.
# The implementation has been migrated to v1; experimental re-exports for backward compatibility.
from duo_workflow_service.agent_platform.v1.components.supervisor.delegate_task import (
    DelegateTask,
    SubagentDescriptor,
    build_delegate_task_model,
)

__all__ = [
    "DelegateTask",
    "SubagentDescriptor",
    "build_delegate_task_model",
]
