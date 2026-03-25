"""Model metadata context variable shared between services.

Provides access to the current model metadata within request context.
Note: Only the context variable is here. The actual ModelMetadata classes
remain in ai_gateway due to their complex dependencies.
"""

from contextvars import ContextVar
from typing import Any, Literal, Optional

# The ModelMetadataBySize type is defined in ai_gateway.model_metadata
# We use Any here to avoid circular dependencies
current_model_metadata_with_size_context: ContextVar[Optional[Any]] = ContextVar(
    "current_model_metadata_with_size_context", default=None
)

# Backward-compatible alias used by ai_gateway HTTP middleware and API endpoints
current_model_metadata_context: ContextVar[Optional[Any]] = ContextVar(
    "current_model_metadata_context", default=None
)

ModelSizeBucket = Literal["small", "large"]


def get_model_metadata(model_size: ModelSizeBucket | None = None) -> Optional[Any]:
    """Return model metadata for the given model size, or None if no context is set."""
    if (models_metadata := current_model_metadata_with_size_context.get()) is not None:
        return models_metadata.get(model_size)
    return None
