from contextvars import ContextVar
from typing import Optional

from ai_gateway.code_suggestions.language_server import LanguageServerVersion

__all__ = ["language_server_version"]

# Context variable to store language server version
language_server_version: ContextVar[Optional[LanguageServerVersion]] = ContextVar(
    "language_server_version", default=None
)
