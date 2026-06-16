from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from duo_workflow_service.gitlab.schema import PromptInjectionProtectionLevel

__all__ = ["MonitoringContext", "current_monitoring_context"]


class MonitoringContext(BaseModel):
    workflow_id: Optional[str] = None
    workflow_definition: Optional[str] = None
    flow_id: Optional[str] = None
    flow_version: Optional[str] = None
    schema_version: Optional[str] = None
    workflow_stop_reason: Optional[str] = None
    workflow_last_gitlab_status: Optional[str] = None
    tracing_enabled: Optional[str] = None
    use_ai_prompt_scanning: bool = False
    prompt_injection_protection_level: PromptInjectionProtectionLevel = (
        PromptInjectionProtectionLevel.LOG_ONLY
    )

    def set_flow_identity(
        self,
        *,
        flow_id: Optional[str] = None,
        flow_version: Optional[str] = None,
        schema_version: Optional[str] = None,
    ) -> None:
        """Record resolved flow identity fields.

        Only truthy values are stored, so empty strings and None are ignored.
        The keyword-only signature is a contract with ``ResolvedFlow.tracking_fields``:
        an unknown key raises ``TypeError`` on ``**`` unpacking.
        """
        if flow_id:
            self.flow_id = flow_id
        if flow_version:
            self.flow_version = flow_version
        if schema_version:
            self.schema_version = schema_version

    def flow_versioning_fields(self) -> dict[str, str]:
        """Return the flow identity fields for tracing and observability.

        Unset or empty values are omitted so that legacy flows (which carry no versioning information) contribute
        nothing.
        """
        fields = {
            "flow_id": self.flow_id,
            "flow_version": self.flow_version,
            "schema_version": self.schema_version,
        }
        return {key: value for key, value in fields.items() if value}


current_monitoring_context: ContextVar[MonitoringContext] = ContextVar(
    "current_monitoring_context", default=MonitoringContext()
)
