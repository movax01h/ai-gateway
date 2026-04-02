from contextvars import ContextVar
from typing import Any, Optional

response_schema_tracking_results: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "response_schema_tracking_results", default=None
)
