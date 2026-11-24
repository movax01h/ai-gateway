from dependency_injector import providers

from ai.platform.container import ContainerPlatform
from duo_workflow_service.audit_events.container import ContainerAuditEvent

__all__ = [
    "ContainerApplication",
]


class ContainerApplication(ContainerPlatform):
    """The Duo Workflow Service container: the platform core plus the providers only DWS uses."""

    audit_event = providers.Container(
        ContainerAuditEvent, config=ContainerPlatform.config.audit_event
    )
