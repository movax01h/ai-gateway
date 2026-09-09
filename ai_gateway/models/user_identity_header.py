"""Per-request user identity header for upstream model endpoints on a self-hosted AI Gateway.

Operators can attribute upstream LLM usage to individual GitLab users by configuring
``AIGW_CUSTOM_MODELS__USER_ID_HEADER``. The value forwarded is the instance-local user ID GitLab Rails sends on every
request (``x-gitlab-user-id``), captured into the ``gitlab_user_id`` ContextVar by ``RequestMetadataMiddleware`` (AI
Gateway) and ``MetadataContextInterceptor`` (Duo Workflow Service). Both run unconditionally, so forwarding does not
depend on internal events being enabled.

The header value is taken from the request header as sent by the client and is not bound to the access token, so it
is suitable for attribution and chargeback but must never be used for authorization upstream. Only plain numeric IDs
are forwarded, which is what Rails sends; anything else is dropped.

The header is injected at LLM call time rather than at model construction time so that the value is always the current
request's user and never leaks between users through cached model objects.
"""

import re
from typing import Any, MutableMapping, Optional

import structlog

from lib.context import gitlab_user_id

__all__ = ["inject_user_identity_header"]

log = structlog.stdlib.get_logger("user_identity_header")

_USER_ID_RE = re.compile(r"^[0-9]{1,20}$")


def inject_user_identity_header(
    params: MutableMapping[str, Any], header_name: Optional[str]
) -> None:
    """Merge the current user's identity header into ``params["extra_headers"]`` in place.

    No-op when the feature is not configured or the current request carries no usable instance-local user ID (e.g.
    non-user traffic). Existing headers such as operator-configured ``extra_headers`` and ``x-session-affinity`` are
    preserved; an existing header with the same name (compared case-insensitively) is replaced so the forwarded value
    is always the current user's.
    """
    if not header_name:
        return

    user_id = gitlab_user_id.get()
    if not user_id:
        return

    if not _USER_ID_RE.match(user_id):
        log.debug(
            "Ignoring non-numeric user ID for identity header", header_name=header_name
        )
        return

    existing = params.get("extra_headers") or {}
    params["extra_headers"] = {
        **{k: v for k, v in existing.items() if k.lower() != header_name.lower()},
        header_name: user_id,
    }
    log.debug(
        "Forwarding user identity header to model endpoint", header_name=header_name
    )
