"""Model metadata context variable shared between services.

Provides access to the current model metadata within request context.
Note: Only the context variable is here. The actual ModelMetadata classes
remain in ai_gateway due to their complex dependencies.
"""

from contextvars import ContextVar
from typing import Any, Optional

# The TypeModelMetadata type is defined in ai_gateway.model_metadata
# We use Any here to avoid circular dependencies
current_model_metadata_context: ContextVar[Optional[Any]] = ContextVar(
    "current_model_metadata_context", default=None
)
