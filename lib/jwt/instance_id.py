"""The GitLab instance identity of an authenticated request, taken from signed JWT claims only."""

import os
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from gitlab_cloud_connector import CloudConnectorConfig

__all__ = ["instance_uid_claim", "trusted_instance_id"]

_DEFAULT_CUSTOMERS_DOT_URL = "https://customers.gitlab.com"


def _normalise_url(url: str) -> str:
    """Strip trailing slashes and lowercase the scheme and host, so equivalent URLs compare equal."""
    parts = urlsplit(url.strip().rstrip("/"))
    return urlunsplit(
        parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower())
    )


def _customers_dot_urls() -> set[str]:
    """Every URL this process may treat as CustomersDot.

    The Duo Workflow Service verifier reads DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL and the AI gateway reads
    AIGW_CUSTOMER_PORTAL_URL, so both are included, together with the production default. Empty values are skipped.
    Every source counts, not just the first one set, and the test-only mock usage quota server URL is never used.
    """
    candidates = (
        # pylint: disable-next=direct-environment-variable-reference
        os.environ.get("DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL", ""),
        # pylint: disable-next=direct-environment-variable-reference
        os.environ.get("AIGW_CUSTOMER_PORTAL_URL", ""),
        _DEFAULT_CUSTOMERS_DOT_URL,
    )
    return {_normalise_url(url) for url in candidates if url and url.strip()}


def _is_customers_dot_issuer(issuer: Any) -> bool:
    """Whether the token's issuer matches any URL in _customers_dot_urls."""
    if not isinstance(issuer, str) or not issuer.strip():
        return False
    return _normalise_url(issuer) in _customers_dot_urls()


def trusted_instance_id(claims: Optional[Any]) -> Optional[str]:
    """Return the instance ID the request's verified JWT vouches for, or None.

    - ``gitlab_instance_uid``: assigned by CustomersDot, or copied into the tokens this gateway mints.
    - ``sub``, for tokens the GitLab instance signed itself (self-hosted models): the instance UUID.

    ``sub`` is never read from a token this gateway issued, because there it is a user, nor from a
    CustomersDot token, because there it is reported by the instance and CustomersDot does not vouch for it.

    ``gitlab_instance_id`` is deliberately not used: CustomersDot documents it as reported by the
    instance, so it can be spoofed, and request headers are not signed at all.
    """
    if claims is None:
        return None

    instance_uid = getattr(claims, "gitlab_instance_uid", None)
    if instance_uid:
        return instance_uid

    issuer = getattr(claims, "issuer", None)
    if issuer == CloudConnectorConfig().service_name or _is_customers_dot_issuer(
        issuer
    ):
        return None

    return getattr(claims, "subject", None) or None


def instance_uid_claim(claims: Optional[Any]) -> dict[str, Optional[str]]:
    """The ``gitlab_instance_uid`` claim to copy into a token this gateway mints for the caller.

    Self-signed tokens carry no ``gitlab_instance_uid``, and the minted token's own ``sub`` is a user,
    so the verified instance ID has to travel in the claim for later checks to find it.
    """
    return {"gitlab_instance_uid": trusted_instance_id(claims)}
