from dependency_injector import containers, providers

from ai_gateway.response_schemas.registry import ResponseSchemaRegistry

__all__ = ["ContainerSchemas"]


class ContainerSchemas(containers.DeclarativeContainer):
    schema_registry = providers.Singleton(ResponseSchemaRegistry)
