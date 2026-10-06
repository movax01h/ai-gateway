"""Model metadata context variable shared between services.

Provides access to the current model metadata within request context.
Note: Only the context variable is here. The actual ModelMetadata classes
remain in ai_gateway due to their complex dependencies.
"""

from contextvars import ContextVar
from typing import Any, Optional

# The ModelMetadataByTag type is defined in ai_gateway.model_metadata
# We use Any here to avoid circular dependencies
current_model_metadata_with_size_context: ContextVar[Optional[Any]] = ContextVar(
    "current_model_metadata_with_size_context", default=None
)

# Backward-compatible alias used by ai_gateway HTTP middleware and API endpoints
current_model_metadata_context: ContextVar[Optional[Any]] = ContextVar(
    "current_model_metadata_context", default=None
)

# Flow config id this request is authorized for; a restricted model (one with
# `restricted_to_flows`) is usable only when it is in that list. None means no
# restricted model may be used. Nothing sets it yet, so restricted models are denied.
restricted_access_ctx: ContextVar[Optional[str]] = ContextVar(
    "restricted_access_ctx", default=None
)


def get_model_metadata(model_tags: list[str] | str | None = None) -> Optional[Any]:
    """Return model metadata for the given model tags, or None if no context is set."""
    if (models_metadata := current_model_metadata_with_size_context.get()) is not None:
        return models_metadata.get(model_tags)
    return None
