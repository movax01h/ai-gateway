from enum import StrEnum
from typing import Optional

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.models.base import ModelMetadata
from ai_gateway.models.base_text import TextGenModelBase, TextGenModelOutput
from ai_gateway.safety_attributes import SafetyAttributes

__all__ = [
    "AmazonQModel",
    "KindAmazonQModel",
]


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
        **kwargs,
    ) -> TextGenModelOutput:
        q_client = self._client_factory.get_client(
            current_user=self._current_user,
            role_arn=self._role_arn,
        )

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

        response = q_client.generate_code_recommendations(request_payload)

        recommendations = response.get("CodeRecommendations", [])
        recommendation = recommendations[0] if recommendations else {}

        return TextGenModelOutput(
            text=recommendation.get("content", ""),
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )
