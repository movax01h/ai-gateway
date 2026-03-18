from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from duo_workflow_service.audit_events.collector import AuditEventCollector

audit_collector_context: ContextVar[Optional["AuditEventCollector"]] = ContextVar(
    "audit_collector", default=None
)


def get_audit_collector() -> Optional["AuditEventCollector"]:
    return audit_collector_context.get()
