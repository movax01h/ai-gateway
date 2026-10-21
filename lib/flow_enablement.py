"""Per-instance enablement of gated flows for callers other than GitLab.com, set by GitLab, not by tenants."""

import json
import os
import uuid
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Optional

import structlog
from gitlab_cloud_connector import CloudConnectorConfig

from lib.jwt import trusted_instance_id

# JSON object mapping a gated flow to the instance UUIDs it is enabled for, e.g. {"bl_security": ["<uuid>"]}.
ENABLED_FLOW_INSTANCE_UIDS_ENV = "DUO_WORKFLOW_AUTH__ENABLED_FLOW_INSTANCE_UIDS"
# Off by default: gated flows are for GitLab.com only and the allowlist is not read. GenerateToken refuses other
# callers; ExecuteWorkflow refuses other callers using their own token (see refused_while_not_enforced). Debug users
# (Duo Workflow Service authentication disabled) are exempt from both.
ENFORCE_FLOW_ENABLEMENT_ENV = "DUO_WORKFLOW_AUTH__ENFORCE_FLOW_ENABLEMENT"
# Flows refused to callers other than GitLab.com unless their instance UID is listed for the flow in
# ENABLED_FLOW_INSTANCE_UIDS_ENV (when enforcement is on). Every other instance is refused, Self-Managed included,
# so only flows that run only on GitLab.com belong here. To gate another flow, add its id here and set the
# flow's `available_on` in GitLab to use `:dedicated_enabled_instances` (e.g. `%i[gitlab_com dedicated_enabled_instances]`).
GATED_FLOWS = frozenset({"bl_security"})
# Set only by this service, in tokens it mints for a caller that is_gitlab_com_caller accepted.
GITLAB_COM_VERIFIED_CLAIM = "gitlab_com_verified"
FLOW_NOT_ENABLED_MESSAGE = "Flow '{flow}' is not enabled for this instance"
FLOW_GITLAB_COM_ONLY_MESSAGE = "Flow '{flow}' is only available on GitLab.com"
# The gitlab_realm claim value of GitLab.com tokens.
SAAS_REALM = "saas"

log = structlog.stdlib.get_logger("flow_enablement")


def _canonical_uid(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


@lru_cache(maxsize=4)
def _parse_allowlist(raw: str) -> Mapping[str, frozenset[str]]:
    """Parse the allowlist; malformed JSON enables nothing, malformed entries are dropped."""
    if not raw.strip():
        return MappingProxyType({})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        log.error("Ignoring malformed flow allowlist")
        return MappingProxyType({})

    allowlist: dict[str, frozenset[str]] = {}
    for flow, uids in data.items():
        if flow not in GATED_FLOWS:
            log.warning("Flow allowlist entry is not a gated flow", flow=str(flow)[:64])
        if not isinstance(uids, list):
            log.error("Ignoring malformed flow allowlist entry", flow=str(flow)[:64])
            continue
        canonical = [_canonical_uid(uid) for uid in uids]
        if None in canonical:
            log.error(
                "Ignoring malformed instance UIDs in flow allowlist",
                flow=str(flow)[:64],
                dropped=canonical.count(None),
            )
        allowlist[flow] = frozenset(uid for uid in canonical if uid)
    return MappingProxyType(allowlist)


def is_gitlab_com_signed(claims: Any, gitlab_issuer: str) -> bool:
    """Whether a verified token is signed by GitLab.com's issuer and carries the SaaS realm claim.

    Issuers are compared with trailing slashes stripped on both sides. Only GitLab.com can sign a token with its own
    issuer and the SaaS realm claim; its CI id_tokens carry no realm claim.
    """
    return (
        bool(claims)
        and bool(claims.issuer)
        and bool(gitlab_issuer)
        and claims.issuer.rstrip("/") == gitlab_issuer.rstrip("/")
        and claims.gitlab_realm == SAAS_REALM
    )


def is_gitlab_com_caller(claims: Any, gitlab_issuer: str) -> bool:
    """Whether a verified token belongs to GitLab.com.

    Either it passes is_gitlab_com_signed, or this service minted it with GITLAB_COM_VERIFIED_CLAIM. The realm claim of
    a token this service mints is copied from an unsigned header, so it is not used for those.

    Unlike server._is_gitlab_com_instance_token, which also requires an allowlisted instance UUID, this does not read an
    instance UID env var, so an empty env cannot turn GitLab.com off.
    """
    if not claims or not claims.issuer:
        return False
    if claims.issuer == CloudConnectorConfig().service_name:
        return (claims.extra or {}).get(GITLAB_COM_VERIFIED_CLAIM) is True
    return is_gitlab_com_signed(claims, gitlab_issuer)


def flow_enablement_enforced() -> bool:
    # pylint: disable-next=direct-environment-variable-reference
    value = os.environ.get(ENFORCE_FLOW_ENABLEMENT_ENV, "")
    return value.strip().lower() in ("true", "1")


def is_flow_enabled(user: Any, flow_config_id: str, *, gitlab_issuer: str) -> bool:
    """Whether the caller may run the flow.

    Flows outside GATED_FLOWS are always enabled. Gated flows are enabled for GitLab.com, for instances whose signed
    instance UUID is in the allowlist, and for debug users (authentication disabled). The allowlist is read only while
    flow enablement is enforced.
    """
    if flow_config_id not in GATED_FLOWS or user.is_debug:
        return True
    if is_gitlab_com_caller(user.claims, gitlab_issuer):
        return True
    if not flow_enablement_enforced():
        return False
    instance_uid = _canonical_uid(trusted_instance_id(user.claims))
    # pylint: disable-next=direct-environment-variable-reference
    allowlist = _parse_allowlist(os.environ.get(ENABLED_FLOW_INSTANCE_UIDS_ENV, ""))
    return instance_uid is not None and instance_uid in allowlist.get(
        flow_config_id, frozenset()
    )


def refused_while_not_enforced(
    user: Any, flow_config_id: str, *, gitlab_issuer: str
) -> bool:
    """Whether a gated flow is refused to a caller using its own token while enforcement is off.

    Tokens this service minted are left to GenerateToken's check, so GitLab.com workflows whose tokens were minted
    before the switch existed keep running.
    """
    if flow_config_id not in GATED_FLOWS or user.is_debug or not user.claims:
        return False
    if user.claims.issuer == CloudConnectorConfig().service_name:
        return False
    return not is_gitlab_com_caller(user.claims, gitlab_issuer)


def enabled_gated_flows(user: Any, *, gitlab_issuer: str) -> list[str]:
    return sorted(
        f for f in GATED_FLOWS if is_flow_enabled(user, f, gitlab_issuer=gitlab_issuer)
    )
