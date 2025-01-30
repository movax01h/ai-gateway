from enum import StrEnum

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
        auth_header: str,
        client_factory: AmazonQClientFactory,
    ):
        self._current_user = current_user
        self._auth_header = auth_header
        self._role_arn = role_arn
        self._client_factory = client_factory
        self._metadata = ModelMetadata(
            name=KindAmazonQModel.AMAZON_Q,
            engine=KindAmazonQModel.AMAZON_Q,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    async def generate(  # type: ignore[override]
        self,
        prefix: str,
        suffix: str,
        filename: str,
        language: str,
        **kwargs,
    ) -> TextGenModelOutput:
        q_client = self._client_factory.get_client(
            current_user=self._current_user,
            auth_header=self._auth_header,
            role_arn=self._role_arn,
        )

        request_payload = {
            "fileContext": {
                "leftFileContent": prefix,
                "rightFileContent": suffix,
                "filename": filename,
                "programmingLanguage": {
                    "languageName": language,
                },
            },
            "maxResults": 1,
        }

        response = q_client.generate_code_recommendations(request_payload)

        recommendations = response.get("CodeRecommendations") or []
        recommendation = recommendations[0] or {}

        return TextGenModelOutput(
            text=recommendation["content"],
            # Give a high value, the model doesn't return scores.
            score=10**5,
            safety_attributes=SafetyAttributes(),
        )
