import os
import re
from typing import Any

import fastapi
from fastapi import status

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients.base import (
    BaseProxyModelFactory,
    ProxyModel,
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


def _build_headers_to_upstream() -> dict[str, str]:
    try:
        return {
            "x-api-key": os.environ[
                "ANTHROPIC_API_KEY"
            ]  # pylint: disable=direct-environment-variable-reference
        }
    except KeyError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="API key not found"
        )


class AnthropicProxyModelFactory(BaseProxyModelFactory):
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

        return ProxyModel(
            base_url="https://api.anthropic.com",
            model_name=model_name,
            upstream_path=upstream_path,
            stream=stream,
            upstream_service=_UPSTREAM_SERVICE,
            headers_to_upstream=_build_headers_to_upstream(),
            allowed_upstream_models=allowed_upstream_models,
            allowed_headers_to_upstream=_ALLOWED_HEADERS_TO_UPSTREAM,
            allowed_headers_to_downstream=_ALLOWED_HEADERS_TO_DOWNSTREAM,
        )
