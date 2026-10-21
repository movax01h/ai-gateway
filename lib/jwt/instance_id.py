"""The GitLab instance identity of an authenticated request, taken from signed JWT claims only."""

from typing import Any, Optional

from gitlab_cloud_connector import CloudConnectorConfig

__all__ = ["instance_uid_claim", "trusted_instance_id"]


def trusted_instance_id(claims: Optional[Any]) -> Optional[str]:
    """Return the instance ID the request's verified JWT vouches for, or None.

    - ``gitlab_instance_uid``: assigned by CustomersDot, or copied into the tokens this gateway mints.
    - ``sub``, for tokens the GitLab instance signed itself (self-hosted models): the instance UUID.

    ``sub`` is never read from a token this gateway issued, because there it is a user.

    ``gitlab_instance_id`` is deliberately not used: CustomersDot documents it as reported by the
    instance, so it can be spoofed, and request headers are not signed at all.
    """
    if claims is None:
        return None

    instance_uid = getattr(claims, "gitlab_instance_uid", None)
    if instance_uid:
        return instance_uid

    if getattr(claims, "issuer", None) == CloudConnectorConfig().service_name:
        return None

    return getattr(claims, "subject", None) or None


def instance_uid_claim(claims: Optional[Any]) -> dict[str, Optional[str]]:
    """The ``gitlab_instance_uid`` claim to copy into a token this gateway mints for the caller.

    Self-signed tokens carry no ``gitlab_instance_uid``, and the minted token's own ``sub`` is a user,
    so the verified instance ID has to travel in the claim for later checks to find it.
    """
    return {"gitlab_instance_uid": trusted_instance_id(claims)}
