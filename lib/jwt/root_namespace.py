from collections.abc import Mapping
from typing import Any

GITLAB_ROOT_NAMESPACE_ID_CLAIM = "gitlab_root_namespace_id"


def parse_root_namespace_id(value: object) -> int | None:
    """Parse and validate a raw value as a positive namespace ID.

    Accepts only plain `int` (positive) or decimal `str` (positive). Booleans,
    floats, negative values, zero, and all other types are rejected.

    Args:
        value: The raw claim value to validate.

    Returns:
        A positive `int` if the value is valid, otherwise `None`.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, str) and value.isdecimal():
        root_namespace_id = int(value)
        return root_namespace_id if root_namespace_id > 0 else None

    return None


def root_namespace_id_from_claims_extra(
    extra: Mapping[str, Any] | None,
) -> int | None:
    """Read `gitlab_root_namespace_id` from verified JWT claims.

    This is the authoritative strategy: `extra` must be sourced from claims that
    have already been cryptographically verified (for example
    `CloudConnectorUser.claims.extra`). Use `root_namespace_id_from_header` for
    the realms that carry no such claim.
    """
    jwt_root_namespace_id = extra.get(GITLAB_ROOT_NAMESPACE_ID_CLAIM) if extra else None
    root_namespace_id = parse_root_namespace_id(jwt_root_namespace_id)

    return root_namespace_id


def root_namespace_id_from_header(value: str | None) -> int | None:
    """Read the root namespace ID from the `X-Gitlab-Root-Namespace-Id` header.

    The header is client-controlled, so this is only the correct strategy for
    realms that carry no `gitlab_root_namespace_id` JWT claim (self-managed,
    dedicated, and auth-bypass paths). SaaS must use
    `root_namespace_id_from_claims_extra`.
    """
    return parse_root_namespace_id(value)
