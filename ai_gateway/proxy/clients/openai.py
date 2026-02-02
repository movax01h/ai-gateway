import os
import typing

import fastapi
from fastapi import status

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.proxy.clients.base import BaseProxyClient
from ai_gateway.proxy.clients.token_usage import TokenUsage


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

    def _extract_token_usage(
        self, upstream_path: str, json_body: typing.Any
    ) -> TokenUsage:
        """Extract token usage from OpenAI response.

        OpenAI response format:
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 5
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 15
                }
            }
        }

        Note: For reasoning models (e.g., gpt-5), completion_tokens includes
        reasoning_tokens, which are also broken out separately in
        completion_tokens_details.reasoning_tokens.
        """
        try:
            if not isinstance(json_body, dict):
                raise ValueError("Response body must be a dictionary")

            usage = json_body.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Usage field must be a dictionary")

            prompt_tokens_details = usage.get("prompt_tokens_details", {})
            cached_tokens = 0
            if isinstance(prompt_tokens_details, dict):
                cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

            completion_tokens_details = usage.get("completion_tokens_details", {})
            reasoning_tokens = 0
            if isinstance(completion_tokens_details, dict):
                reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)

            return TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens"),
                cache_read_input_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise fastapi.HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract token usage from response",
            )

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
