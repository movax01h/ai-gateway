"""Flow-scoped registry for inline (in-config) response schema definitions."""

from typing import Any, Type

from ai_gateway.response_schemas.base import BaseAgentOutput, BaseResponseSchemaRegistry
from ai_gateway.response_schemas.converter import json_schema_to_pydantic

__all__ = ["InlineResponseSchemaRegistry"]


class InlineResponseSchemaRegistry(BaseResponseSchemaRegistry):
    """Flow-scoped registry combining inline schema dicts with a shared file-based registry.

    Mirrors the role of ``InMemoryPromptRegistry`` for prompts: created once per
    flow instance, pre-converts any inline JSON Schema dicts supplied in the flow
    config, and falls back to the shared ``ResponseSchemaRegistry`` singleton for
    registry-based lookups (``response_schema_id`` + ``response_schema_version``).

    Routing logic: a ``schema_id`` registered via ``register_schema()`` returns the
    pre-converted ``BaseAgentOutput`` subclass (``schema_version`` is ignored).
    All other ``schema_id`` values are delegated to the shared registry.

    See ``docs/flow_registry/v1.md`` for full YAML usage examples.
    """

    def __init__(self, shared_registry: BaseResponseSchemaRegistry) -> None:
        self._shared_registry = shared_registry
        self._schemas: dict[str, Type[BaseAgentOutput]] = {}

    def register_schema(self, schema_id: str, schema_dict: dict[str, Any]) -> None:
        """Pre-convert and cache an inline JSON Schema dict.

        The ``schema_id`` (typically the component ``name``) is used as the tool
        title fallback when the schema has no ``title`` field of its own.

        Args:
            schema_id: Identifier for this schema, used as key in the cache and
                as the tool-title fallback when the schema omits ``title``.
            schema_dict: Raw JSON Schema dict to convert to a ``BaseAgentOutput``
                subclass.

        Raises:
            ValueError: If the schema is invalid (bad JSON Schema, unsupported
                nesting depth, missing ``type: object``, etc.), or if
                ``schema_id`` has already been registered.
        """
        if schema_id in self._schemas:
            raise ValueError(
                f"Inline response schema '{schema_id}' is already registered. "
                f"Each schema_id must be unique within a flow's 'response_schemas' block."
            )
        self._schemas[schema_id] = json_schema_to_pydantic(
            schema_dict, title_fallback=schema_id
        )

    def get(self, schema_id: str, schema_version: str) -> Type[BaseAgentOutput]:
        """Return the Pydantic model for *schema_id*.

        For inline schemas (registered via ``register_schema``), ``schema_version``
        is ignored — there is only one version of an inline definition.

        For all other IDs, the call is delegated to the shared registry where
        ``schema_version`` is honoured as a semver constraint.

        Args:
            schema_id: Schema identifier.  For inline schemas this is the
                component ``name``; for registry schemas it is the path-based ID
                (e.g. ``"fix_pipeline_decide_approach"``).
            schema_version: Semver constraint string.  Ignored for inline schemas.
                Must be non-empty when ``schema_id`` is not registered inline —
                an empty version string cannot be resolved by the shared registry
                and most likely means a ``response_schema_version`` was omitted for
                a registry-based schema.

        Returns:
            A ``BaseAgentOutput`` subclass representing the schema.

        Raises:
            ValueError: If the schema_id is not registered inline and the shared
                registry cannot resolve it, or if ``schema_id`` is not registered
                inline and ``schema_version`` is empty (missing version for a
                registry-based schema).
        """
        if schema_id in self._schemas:
            return self._schemas[schema_id]
        if not schema_version:
            raise ValueError(
                f"response_schema_id '{schema_id}' is not defined in the flow's "
                f"'response_schemas' block. To use a registry-based schema, also "
                f"set 'response_schema_version'."
            )
        return self._shared_registry.get(schema_id, schema_version)
