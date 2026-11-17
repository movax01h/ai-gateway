import re
from contextvars import ContextVar
from typing import Optional

__all__ = [
    "get_session_owner_username",
    "set_session_owner_username",
]

# Only GitLab's username characters and length. Rails owns the other rules, and older usernames predate the current ones.
# Anything else could break the prompt template.
_USERNAME_PATTERN = re.compile(r"\A[A-Za-z0-9_.-]{1,255}\Z")

_session_owner_username: ContextVar[Optional[str]] = ContextVar(
    "session_owner_username", default=None
)


def set_session_owner_username(value: Optional[str]) -> None:
    """Store the session owner's username, or None when the value is missing or not a GitLab username."""
    if isinstance(value, str) and _USERNAME_PATTERN.match(value):
        _session_owner_username.set(value)
    else:
        _session_owner_username.set(None)


def get_session_owner_username() -> Optional[str]:
    return _session_owner_username.get()
