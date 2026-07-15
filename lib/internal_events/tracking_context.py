"""Parsing for the client-supplied ``x-gitlab-tracking-context`` header.

Clients can send an optional ``x-gitlab-tracking-context`` header carrying
a plain-JSON object of environment context. The GitLab monolith forwards it
as-is to the AI Gateway (HTTP) and Duo Workflow Service (gRPC metadata).

The payload is client-supplied and open-ended, so parsing is best-effort and
fails open: a missing, oversized, or malformed value yields ``None`` and never
raises.
"""

import json
from typing import Any, Dict, Optional

import structlog

__all__ = [
    "parse_tracking_context",
]

log = structlog.stdlib.get_logger("internal_events_tracking_context")


def parse_tracking_context(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the raw ``x-gitlab-tracking-context`` header value.

    Args:
        raw: The raw header value, or ``None`` when the header is absent.

    Returns:
        A dict of the parsed context when the value is a valid JSON object,
        otherwise ``None``. Never raises.
    """
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("Dropping malformed x-gitlab-tracking-context header")
        return None

    if not isinstance(parsed, dict):
        log.warning(
            "Dropping non-object x-gitlab-tracking-context header",
            parsed_type=type(parsed).__name__,
        )
        return None

    return {str(key): value for key, value in parsed.items()}
