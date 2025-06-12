# pylint: disable=direct-environment-variable-reference

import os
from typing import Literal, Optional, Union

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langsmith import tracing_context
from pydantic import BaseModel, Field, field_validator

from ai_gateway.models import KindAnthropicModel
from lib.feature_flags.context import FeatureFlag, is_feature_enabled


class ModelConfig(BaseModel):
    max_retries: int = 6
    model_name: str
    provider: str


class AnthropicConfig(ModelConfig):
    provider: Literal["anthropic"] = "anthropic"

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model_name matches a value from KindAnthropicModel."""
        valid_models = [model.value for model in KindAnthropicModel]
        if v not in valid_models:
            raise ValueError(
                f"model_name '{v}' is not valid. Must be one of: {', '.join(valid_models)}"
            )
        return v


class VertexConfig(ModelConfig):
    provider: Literal["vertex"] = "vertex"

    @staticmethod
    def _get_model_name() -> str:
        if is_feature_enabled(FeatureFlag.DUO_WORKFLOW_CLAUDE_SONNET_4):
            return KindAnthropicModel.CLAUDE_SONNET_4.value

        return KindAnthropicModel.CLAUDE_3_7_SONNET_VERTEX.value

    @staticmethod
    def _get_project_id() -> str:
        project_id = os.environ.get("DUO_WORKFLOW__VERTEX_PROJECT_ID")
        if not project_id or len(project_id) < 1:
            raise RuntimeError("DUO_WORKFLOW__VERTEX_PROJECT_ID needs to be set")
        return project_id

    @staticmethod
    def _get_location() -> str:
        # This is where we'll need to add support for multi-region access to Anthropic
        # on Vertex.
        # Supported locations:
        # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
        location = os.environ.get("DUO_WORKFLOW__VERTEX_LOCATION")
        if not location or len(location) < 1:
            raise RuntimeError("DUO_WORKFLOW__VERTEX_LOCATION needs to be set")
        return location

    model_name: str = Field(default_factory=_get_model_name)
    project_id: str = Field(default_factory=_get_project_id)
    location: str = Field(default_factory=_get_location)


def create_chat_model(
    config: Union[AnthropicConfig, VertexConfig],
    **kwargs,
) -> BaseChatModel:

    if isinstance(config, AnthropicConfig):
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key and len(anthropic_api_key) > 1:
            return ChatAnthropic(
                model_name=config.model_name,
                **kwargs,
                max_retries=config.max_retries,
            )
        raise RuntimeError("ANTHROPIC_API_KEY needs to be set for Anthropic provider")

    if isinstance(config, VertexConfig):
        return ChatAnthropicVertex(
            model_name=config.model_name,
            project=config.project_id,
            location=config.location,
            max_retries=config.max_retries,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported config type: {type(config).__name__}. "
        "Must be either AnthropicConfig or VertexConfig"
    )


def validate_llm_access(config: Optional[Union[AnthropicConfig, VertexConfig]] = None):
    if config is None:
        try:
            config = VertexConfig()
        except RuntimeError:
            config = AnthropicConfig(
                model_name=KindAnthropicModel.CLAUDE_3_7_SONNET.value
            )

    log = structlog.stdlib.get_logger("server")
    anthropic_client = create_chat_model(config=config)

    with tracing_context(enabled=False):
        anthropic_response = anthropic_client.invoke(
            "Answer in under 80 characters: What LLM am I talking to?"
        )

    content = anthropic_response.content
    # feature flags are not yet loaded, so logging the model name here could be misleading if the model name depends on
    # feature flags.
    log.info(str(content))
