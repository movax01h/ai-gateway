import os
import typing

import fastapi
from fastapi import status

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.proxy.clients.base import BaseProxyClient


class OpenAIProxyClient(BaseProxyClient):
    ALLOWED_UPSTREAM_PATHS = [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/models",
        "/v1/responses",
    ]

    ALLOWED_HEADERS_TO_UPSTREAM = [
        "accept",
        "content-type",
        "user-agent",
    ]

    ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

    PROVIDER_NAME = "openai"

    def _base_url(self) -> str:
        return "https://api.openai.com"

    def _allowed_upstream_paths(self) -> list[str]:
        return OpenAIProxyClient.ALLOWED_UPSTREAM_PATHS

    def _allowed_headers_to_upstream(self):
        return OpenAIProxyClient.ALLOWED_HEADERS_TO_UPSTREAM

    def _allowed_headers_to_downstream(self):
        return OpenAIProxyClient.ALLOWED_HEADERS_TO_DOWNSTREAM

    def _allowed_upstream_models(self) -> list[str]:
        config = ModelSelectionConfig.instance()
        return config.get_proxy_models_for_provider(self.PROVIDER_NAME)

    def _upstream_service(self):
        return OpenAIProxyClient.PROVIDER_NAME

    def _extract_model_name(self, upstream_path: str, json_body: typing.Any) -> str:
        try:
            return json_body["model"]
        except KeyError:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract model name",
            )

    def _extract_stream_flag(self, upstream_path: str, json_body: typing.Any) -> bool:
        return json_body.get("stream", False)

    def _update_headers_to_upstream(self, headers_to_upstream: typing.Any) -> None:
        try:
            api_key = os.environ[
                "OPENAI_API_KEY"
            ]  # pylint: disable=direct-environment-variable-reference
            if not api_key:
                raise fastapi.HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Server configuration error: API key not set",
                )

            headers_to_upstream["Authorization"] = (
                f"Bearer {os.environ['OPENAI_API_KEY']}"  # pylint: disable=direct-environment-variable-reference
            )
        except KeyError:
            raise fastapi.HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key not found",
            )
