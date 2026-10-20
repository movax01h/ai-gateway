"""Anthropic Workload Identity Federation (WIF) backed by Google Cloud ADC.

Anthropic WIF exchanges an OIDC JWT from a trusted issuer for a short-lived Anthropic access
token (RFC 7523 ``jwt-bearer`` grant against ``POST /v1/oauth/token``). Here the JWT is a
Google-signed identity token minted from Application Default Credentials, so a model can be
served from a dedicated Anthropic organization/workspace without shipping a static API key.

See https://platform.claude.com/docs/en/manage-claude/wif-providers/gcp
"""

from functools import lru_cache

from anthropic.lib.credentials import TokenCache, WorkloadIdentityCredentials

from ai_gateway.auth import gcp
from ai_gateway.model_selection.models import AnthropicWorkloadIdentityParams

# Defined locally because the SDK only exposes it from a private module (`anthropic.lib.credentials._constants`).
OAUTH_API_BETA_HEADER = "oauth-2025-04-20"

__all__ = [
    "OAUTH_API_BETA_HEADER",
    "access_token",
    "credentials",
    "merge_oauth_beta_header",
]


@lru_cache(maxsize=None)
def credentials(params: AnthropicWorkloadIdentityParams) -> WorkloadIdentityCredentials:
    """Build the SDK credentials provider for `params`.

    Cached per distinct params so every `AsyncAnthropic` client (and `with_options()` copy) built
    from the same model definition shares one provider and therefore one token cache.
    """

    def identity_token() -> str:
        return gcp.id_token(params.audience)

    return WorkloadIdentityCredentials(
        identity_token_provider=identity_token,
        federation_rule_id=params.federation_rule_id,
        organization_id=params.organization_id,
        service_account_id=params.service_account_id,
        workspace_id=params.workspace_id,
    )


@lru_cache(maxsize=None)
def _token_cache(params: AnthropicWorkloadIdentityParams) -> TokenCache:
    return TokenCache(credentials(params))


def access_token(params: AnthropicWorkloadIdentityParams) -> str:
    """Return a valid Anthropic access token for `params`, exchanging a fresh identity token only when the cached one is
    near expiry.

    Blocking: the exchange is a synchronous HTTP call. Callers on the event loop should run it
    in a thread.
    """
    return _token_cache(params).get_token()


def merge_oauth_beta_header(existing: str | None) -> str:
    """Add the OAuth beta flag that unlocks `Authorization: Bearer` auth to an `anthropic-beta` header value, preserving
    any flags already present."""
    flags = [flag.strip() for flag in (existing or "").split(",") if flag.strip()]
    if OAUTH_API_BETA_HEADER not in flags:
        flags.append(OAUTH_API_BETA_HEADER)
    return ",".join(flags)
