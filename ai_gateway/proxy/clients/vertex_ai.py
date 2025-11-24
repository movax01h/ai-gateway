import re
import typing

import fastapi
from fastapi import status

from ai_gateway.auth.gcp import access_token
from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.vertex_text import KindVertexTextModel
from ai_gateway.proxy.clients.base import BaseProxyClient
from ai_gateway.proxy.clients.token_usage import TokenUsage


class VertexAIProxyClient(BaseProxyClient):
    def __init__(self, project: str, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self.location = location

    ALLOWED_HEADERS_TO_UPSTREAM = [
        "content-type",
    ]

    ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

    def _extract_upstream_path(self, request_path: str) -> str:
        model, action, sse_flag = self._extract_params_from_path(request_path)
        return (
            f"/v1/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{model}:{action}{sse_flag}"
        )

    def _allowed_upstream_paths(self) -> list[str]:
        return []  # No-op. _extract_upstream_path is overridden instead.

    def _allowed_headers_to_upstream(self):
        return VertexAIProxyClient.ALLOWED_HEADERS_TO_UPSTREAM

    def _allowed_headers_to_downstream(self):
        return VertexAIProxyClient.ALLOWED_HEADERS_TO_DOWNSTREAM

    def _allowed_upstream_models(self) -> list[str]:
        """Return all allowed models including Google and Anthropic models.

        Note: The check for textmodels will be removed in the future.
        See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/906
        """
        google_models = [el.value for el in KindVertexTextModel]
        anthropic_models = [
            el.value
            for el in KindAnthropicModel
            if el.value.endswith("-vertex") or "@" in el.value
        ]
        return google_models + anthropic_models

    def _upstream_service(self):
        return KindModelProvider.VERTEX_AI.value

    def _extract_model_name(self, upstream_path: str, json_body: typing.Any) -> str:
        model, _, _ = self._extract_params_from_path(upstream_path)
        return model

    def _extract_stream_flag(self, upstream_path: str, json_body: typing.Any) -> bool:
        _, action, _ = self._extract_params_from_path(upstream_path)
        return action == "serverStreamingPredict"

    def _extract_token_usage(
        self, upstream_path: str, json_body: typing.Any
    ) -> TokenUsage:
        """Extract token usage from Vertex AI response.

        Example embedding response format:
        {
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {
                            "truncated": false,
                            "token_count": 6
                        },
                        "values": [ ... ]
                    }
                }
            ]
        }
        """
        # pylint: disable=too-many-nested-blocks
        try:
            if not isinstance(json_body, dict):
                raise ValueError("Response body must be a dictionary")

            # Check if this is an embedding model response
            # Embedding models return token count in predictions[0].embeddings.statistics
            if "predictions" in json_body and isinstance(
                json_body["predictions"], list
            ):
                if len(json_body["predictions"]) > 0:
                    prediction = json_body["predictions"][0]
                    if isinstance(prediction, dict) and "embeddings" in prediction:
                        embeddings = prediction["embeddings"]
                        if isinstance(embeddings, dict) and "statistics" in embeddings:
                            statistics = embeddings["statistics"]
                            if isinstance(statistics, dict):
                                token_count = statistics.get("token_count", 0)
                                # For embeddings, tokens are input tokens
                                return TokenUsage(
                                    input_tokens=token_count,
                                    output_tokens=0,
                                )

            # For generative models, check metadata.tokenMetadata
            metadata = json_body.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("Metadata field must be a dictionary")

            token_metadata = metadata.get("tokenMetadata", {})
            if not isinstance(token_metadata, dict):
                raise ValueError("Token metadata field must be a dictionary")

            input_token_count = token_metadata.get("inputTokenCount", {})
            output_token_count = token_metadata.get("outputTokenCount", {})

            input_tokens = (
                input_token_count.get("totalTokens", 0)
                if isinstance(input_token_count, dict)
                else 0
            )
            output_tokens = (
                output_token_count.get("totalTokens", 0)
                if isinstance(output_token_count, dict)
                else 0
            )

            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except (KeyError, TypeError, ValueError, AttributeError):
            raise fastapi.HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract token usage from response",
            )

    def _update_headers_to_upstream(self, headers_to_upstream: typing.Any) -> None:
        headers_to_upstream["Authorization"] = f"Bearer {access_token()}"

    def _extract_params_from_path(self, path: str) -> tuple[str, str, str]:
        match = re.search(
            "/v1/projects/.*/locations/.*/publishers/google/models/(.*):(predict|serverStreamingPredict)(\\?alt=sse)?",
            path,
        )

        try:
            assert match is not None

            model = match.group(1)
            action = match.group(2)
            sse_flag = match.group(3) or ""
        except (IndexError, AssertionError):
            raise fastapi.HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
            )

        return model, action, sse_flag
