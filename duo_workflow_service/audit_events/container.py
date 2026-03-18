from dependency_injector import containers, providers

__all__ = ["ContainerAuditEvent"]


class ContainerAuditEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)
