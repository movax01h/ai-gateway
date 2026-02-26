import os
import re
from typing import Any, override

import fastapi
from fastapi import status

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.proxy.clients.base import (
    BaseProxyModelFactory,
    ProxyModel,
    extract_json_body,
)

_ALLOWED_UPSTREAM_PATHS = [
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/models",
    "/v1/responses",
]

_ALLOWED_HEADERS_TO_UPSTREAM = [
    "accept",
    "content-type",
    "user-agent",
]

_ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

_UPSTREAM_SERVICE = "openai"


def _extract_upstream_path(
    request_path: str, upstream_service: str, allowed_upstream_paths: list[str]
) -> str:
    path = re.sub(f"^(.*?)/{upstream_service}/", "/", request_path)

    if path not in allowed_upstream_paths:
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
        )

    return path


def _extract_stream_flag(json_body: Any) -> bool:
    return json_body.get("stream", False)


def _extract_model_name(json_body: Any) -> str:
    try:
        return json_body["model"]
    except KeyError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract model name",
        )


def _load_allowed_upstream_models() -> list[str]:
    config = ModelSelectionConfig.instance()

    return config.get_proxy_models_for_provider(_UPSTREAM_SERVICE)


def _build_headers_to_upstream() -> dict[str, str]:
    try:
        api_key = os.environ[
            "OPENAI_API_KEY"
        ]  # pylint: disable=direct-environment-variable-reference
        if not api_key:
            raise fastapi.HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error: API key not set",
            )

        return {
            "Authorization": (
                f"Bearer {os.environ['OPENAI_API_KEY']}"  # pylint: disable=direct-environment-variable-reference
            )
        }
    except KeyError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not found",
        )


class OpenAIProxyModelFactory(BaseProxyModelFactory):
    @override
    async def factory(self, request: fastapi.Request) -> ProxyModel:
        upstream_path = _extract_upstream_path(
            request.url.__str__(), _UPSTREAM_SERVICE, _ALLOWED_UPSTREAM_PATHS
        )

        json_body = await extract_json_body(request)

        allowed_upstream_models = _load_allowed_upstream_models()
        model_name = _extract_model_name(json_body)
        if model_name not in allowed_upstream_models:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model"
            )

        stream = _extract_stream_flag(json_body)

        return ProxyModel(
            base_url="https://api.openai.com",
            model_name=model_name,
            upstream_path=upstream_path,
            stream=stream,
            upstream_service=_UPSTREAM_SERVICE,
            headers_to_upstream=_build_headers_to_upstream(),
            allowed_upstream_models=allowed_upstream_models,
            allowed_headers_to_upstream=_ALLOWED_HEADERS_TO_UPSTREAM,
            allowed_headers_to_downstream=_ALLOWED_HEADERS_TO_DOWNSTREAM,
        )
