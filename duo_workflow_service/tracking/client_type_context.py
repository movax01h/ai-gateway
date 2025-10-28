from contextvars import ContextVar
from typing import Optional

__all__ = ["client_type"]

# Context variable to store client type
client_type: ContextVar[Optional[str]] = ContextVar("client_type", default=None)
