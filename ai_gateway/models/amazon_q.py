from enum import StrEnum
from typing import AsyncIterator, Optional

import structlog

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.integrations.amazon_q.errors import AWSException
from ai_gateway.models.base import ModelMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes

__all__ = [
    "AmazonQModel",
    "KindAmazonQModel",
]

log = structlog.stdlib.get_logger("amazon_q")


class KindAmazonQModel(StrEnum):
    AMAZON_Q = "amazon_q"


class AmazonQModel(TextGenModelBase):
    def __init__(
        self,
        current_user: StarletteUser,
        role_arn: str,
        client_factory: AmazonQClientFactory,
    ):
        self._current_user = current_user
        self._role_arn = role_arn
        self._client_factory = client_factory
        self._metadata = ModelMetadata(
            name=KindAmazonQModel.AMAZON_Q,
            engine=KindAmazonQModel.AMAZON_Q,
        )

    @property
    def input_token_limit(self) -> int:
        return 20480

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(  # type: ignore[override]
        self,
        prefix: str,
        suffix: Optional[str],
        filename: str,
        language: str,
        stream: bool,
        **_kwargs,
    ) -> (
        TextGenModelOutput | list[TextGenModelOutput] | AsyncIterator[TextGenModelChunk]
    ):

        request_payload = {
            "fileContext": {
                "leftFileContent": prefix,
                "rightFileContent": suffix or "",
                "filename": filename,
                "programmingLanguage": {
                    "languageName": language,
                },
            },
            "maxResults": 1,
        }
        try:
            q_client = self._client_factory.get_client(
                current_user=self._current_user,
                role_arn=self._role_arn,
            )
            response = q_client.generate_code_recommendations(request_payload)
        except AWSException as e:
            raise e.to_http_exception()

        recommendations = response.get("CodeRecommendations", [])
        recommendation = recommendations[0] if recommendations else {}
        content = recommendation.get("content", "")

        if stream:

            async def _handle_stream() -> AsyncIterator[TextGenModelChunk]:
                yield TextGenModelChunk(text=content)

            return _handle_stream()

        return TextGenModelOutput(
            text=content,
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )
