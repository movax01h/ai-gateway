from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from duo_workflow_service.gitlab.schema import PromptInjectionProtectionLevel

__all__ = ["MonitoringContext", "current_monitoring_context"]


class MonitoringContext(BaseModel):
    workflow_id: Optional[str] = None
    workflow_definition: Optional[str] = None
    workflow_stop_reason: Optional[str] = None
    workflow_last_gitlab_status: Optional[str] = None
    tracing_enabled: Optional[str] = None
    use_ai_prompt_scanning: bool = False
    prompt_injection_protection_level: PromptInjectionProtectionLevel = (
        PromptInjectionProtectionLevel.LOG_ONLY
    )


current_monitoring_context: ContextVar[MonitoringContext] = ContextVar(
    "current_monitoring_context", default=MonitoringContext()
)
