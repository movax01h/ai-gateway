from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

__all__ = ["MonitoringContext", "current_monitoring_context"]


class MonitoringContext(BaseModel):
    workflow_id: Optional[str] = None
    workflow_definition: Optional[str] = None


current_monitoring_context: ContextVar[MonitoringContext] = ContextVar(
    "current_monitoring_context", default=MonitoringContext()
)
