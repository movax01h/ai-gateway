import os
import typing

import fastapi
from fastapi import status

from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients.base import BaseProxyClient
from ai_gateway.proxy.clients.token_usage import TokenUsage


class AnthropicProxyClient(BaseProxyClient):
    ALLOWED_UPSTREAM_PATHS = [
        "/v1/messages",
        "/v1/messages?beta=true",
        "/v1/messages/count_tokens?beta=true",
    ]

    ALLOWED_HEADERS_TO_UPSTREAM = [
        "accept",
        "content-type",
        "anthropic-version",
    ]

    ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

    PROVIDER_NAME = "anthropic"

    def _allowed_upstream_paths(self) -> list[str]:
        return self.ALLOWED_UPSTREAM_PATHS

    def _allowed_headers_to_upstream(self) -> list[str]:
        return self.ALLOWED_HEADERS_TO_UPSTREAM

    def _allowed_headers_to_downstream(self) -> list[str]:
        return self.ALLOWED_HEADERS_TO_DOWNSTREAM

    def _allowed_upstream_models(self) -> list[str]:
        return [el.value for el in KindAnthropicModel]

    def _upstream_service(self) -> str:
        return KindModelProvider.ANTHROPIC.value

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

    def _extract_token_usage(
        self, upstream_path: str, json_body: typing.Any
    ) -> TokenUsage:
        """Extract token usage from Anthropic response.

        Anthropic response format:
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 3
            }
        }
        """
        try:
            if not isinstance(json_body, dict):
                raise ValueError("Response body must be a dictionary")

            usage = json_body.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Usage field must be a dictionary")

            return TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise fastapi.HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract token usage from response",
            )

    def _update_headers_to_upstream(self, headers_to_upstream: typing.Any) -> None:
        try:
            headers_to_upstream["x-api-key"] = os.environ[
                "ANTHROPIC_API_KEY"
            ]  # pylint: disable=direct-environment-variable-reference
        except KeyError:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="API key not found"
            )
