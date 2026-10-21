import asyncio
import os
import re
from typing import Any, Mapping, override

import fastapi
import structlog
from anthropic.lib.credentials import WorkloadIdentityError
from fastapi import status
from google.auth.exceptions import GoogleAuthError

from ai_gateway.auth import anthropic_wif
from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.model_selection.models import AnthropicWorkloadIdentityParams
from ai_gateway.models.base import KindModelProvider, anthropic_facilitator_headers
from ai_gateway.proxy.clients.base import (
    BaseProxyModelFactory,
    ProxyModel,
    enforce_model_allowlist,
    extract_json_body,
)

_ALLOWED_UPSTREAM_PATHS = [
    "/v1/messages",
    "/v1/messages?beta=true",
    "/v1/messages/count_tokens?beta=true",
]

_ALLOWED_HEADERS_TO_UPSTREAM = [
    "accept",
    "content-type",
    "anthropic-version",
    "anthropic-beta",
]

_ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type", "anthropic-beta"]

_UPSTREAM_SERVICE = KindModelProvider.ANTHROPIC.value

log = structlog.stdlib.get_logger("proxy")


def _extract_upstream_path(
    request_path: str, upstream_service: str, allowed_upstream_paths: list[str]
) -> str:
    path = re.sub(f"^(.*?)/{upstream_service}/", "/", request_path)

    if path not in allowed_upstream_paths:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
        )

    return path


def _extract_model_name(json_body: Any) -> str:
    try:
        return json_body["model"]
    except KeyError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract model name",
        )


def _extract_stream_flag(json_body: Any) -> bool:
    return json_body.get("stream", False)


def _load_allowed_upstream_models(provider_name: str = "anthropic") -> list[str]:
    config = ModelSelectionConfig.instance()
    return config.get_proxy_models_for_provider(provider_name)


def _resolve_api_key(model_name: str) -> str:
    """Resolve the Anthropic API key for a model.

    Prefers a per-model `api_key` configured via `AIGW_MODEL_SELECTION__MODEL_PARAMS`
    (which enables routing to a specific Anthropic organization/workspace, e.g. for
    models that require data retention to be enabled). Falls back to the shared
    `ANTHROPIC_API_KEY` environment variable.
    """
    config = ModelSelectionConfig.instance()
    for llm_def in config.get_llm_definitions().values():
        if (
            llm_def.proxy_provider == _UPSTREAM_SERVICE
            and llm_def.params.model == model_name
        ):
            api_key = getattr(llm_def.params, "api_key", None)
            if api_key:
                return api_key
            break

    try:
        return os.environ["ANTHROPIC_API_KEY"]  # pylint: disable=direct-environment-variable-reference
    except KeyError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="API key not found"
        )


def _build_headers_to_upstream(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, **anthropic_facilitator_headers()}


async def _build_workload_identity_headers(
    params: AnthropicWorkloadIdentityParams, request_headers: Mapping[str, str]
) -> dict[str, str]:
    """Authenticate upstream with an Anthropic WIF access token instead of an API key.

    The token is sent as `Authorization: Bearer`, which the API only accepts alongside the OAuth beta flag, so that flag
    is merged into whatever `anthropic-beta` value the client sent (the proxy's own headers win over forwarded ones, so
    the merge has to happen here).
    """
    try:
        # The SDK's token exchange is synchronous; keep it off the event loop.
        token = await asyncio.to_thread(anthropic_wif.access_token, params)
    except (WorkloadIdentityError, GoogleAuthError, ValueError) as exc:
        log.error("Anthropic workload identity token exchange failed", exc_info=exc)
        raise fastapi.HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to authenticate with the upstream provider",
        ) from exc

    return {
        "Authorization": f"Bearer {token}",
        "anthropic-beta": anthropic_wif.merge_oauth_beta_header(
            request_headers.get("anthropic-beta")
        ),
        **anthropic_facilitator_headers(),
    }


def _find_workload_identity(model_name: str) -> AnthropicWorkloadIdentityParams | None:
    config = ModelSelectionConfig.instance()
    for llm_def in config.get_llm_definitions().values():
        if (
            llm_def.proxy_provider == _UPSTREAM_SERVICE
            and llm_def.params.model == model_name
        ):
            return getattr(llm_def.params, "workload_identity", None)
    return None


async def _resolve_upstream_auth_headers(
    model_name: str, request_headers: Mapping[str, str]
) -> dict[str, str]:
    """Pick the upstream credential for a model: per-model workload identity when configured,
    otherwise the per-model or shared API key."""
    workload_identity = _find_workload_identity(model_name)
    if workload_identity:
        return await _build_workload_identity_headers(
            workload_identity, request_headers
        )

    return _build_headers_to_upstream(_resolve_api_key(model_name))


class AnthropicProxyModelFactory(BaseProxyModelFactory):
    @override
    async def factory(self, request: fastapi.Request) -> ProxyModel:
        upstream_path = _extract_upstream_path(
            request.url.__str__(), _UPSTREAM_SERVICE, _ALLOWED_UPSTREAM_PATHS
        )

        json_body = await extract_json_body(request)

        stream = _extract_stream_flag(json_body)
        allowed_upstream_models = _load_allowed_upstream_models()
        model_name = _extract_model_name(json_body)

        if model_name not in allowed_upstream_models:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model"
            )

        enforce_model_allowlist(request, model_name)

        auth_headers = await _resolve_upstream_auth_headers(model_name, request.headers)

        return ProxyModel(
            base_url="https://api.anthropic.com",
            model_name=model_name,
            upstream_path=upstream_path,
            stream=stream,
            upstream_service=_UPSTREAM_SERVICE,
            headers_to_upstream=auth_headers,
            allowed_upstream_models=allowed_upstream_models,
            allowed_headers_to_upstream=_ALLOWED_HEADERS_TO_UPSTREAM,
            allowed_headers_to_downstream=_ALLOWED_HEADERS_TO_DOWNSTREAM,
        )
