# pylint: disable=direct-environment-variable-reference

import os

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langsmith import tracing_context

from duo_workflow_service.interceptors.feature_flag_interceptor import (
    current_feature_flag_context,
)


class VertexConfig:
    @property
    def model_name(self) -> str:
        feature_flags = current_feature_flag_context.get()
        if "duo_workflow_claude_sonnet_4" in feature_flags:
            return "claude-sonnet-4@20250514"
        if "duo_workflow_claude_3_7" in feature_flags:
            return "claude-3-7-sonnet@20250219"

        return "claude-3-5-sonnet-v2@20241022"

    @property
    def project_id(self) -> str:
        project_id = os.environ.get("DUO_WORKFLOW__VERTEX_PROJECT_ID")
        if not project_id or len(project_id) < 1:
            raise RuntimeError("DUO_WORKFLOW__VERTEX_PROJECT_ID needs to be set")
        return project_id

    @property
    def location(self) -> str:
        # This is where we'll need to add support for multi-region access to Anthropic
        # on Vertex.
        # Supported locations:
        # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions
        location = os.environ.get("DUO_WORKFLOW__VERTEX_LOCATION")
        if not location or len(location) < 1:
            raise RuntimeError("DUO_WORKFLOW__VERTEX_LOCATION needs to be set")
        return location

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts in case of failure."""
        return 6


def new_chat_client(config: VertexConfig = VertexConfig(), **kwargs) -> BaseChatModel:
    vertex_project_id = os.environ.get("DUO_WORKFLOW__VERTEX_PROJECT_ID")

    if vertex_project_id and len(vertex_project_id) > 1:
        return ChatAnthropicVertex(
            model_name=config.model_name,
            project=config.project_id,
            location=config.location,
            max_retries=config.max_retries,
            **kwargs,
        )

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_model_name = get_anthropic_model_name()
    if anthropic_api_key and len(anthropic_api_key) > 1:
        return ChatAnthropic(
            model_name=anthropic_model_name, **kwargs, max_retries=config.max_retries
        )

    raise RuntimeError(
        "Either Vertex needs to be configured, or an ANTHROPIC_API_KEY needs to be set"
    )


def validate_llm_access(config: VertexConfig = VertexConfig()):
    log = structlog.stdlib.get_logger("server")
    anthropic_client = new_chat_client(config=config)

    with tracing_context(enabled=False):
        anthropic_response = anthropic_client.invoke(
            "Answer in under 80 characters: What LLM am I talking to?"
        )

    content = anthropic_response.content
    # feature flags are not yet loaded, so logging the model name here could be misleaeding if the model name depends on feature flags.
    log.info(str(content))


def get_anthropic_model_name() -> str:
    feature_flags = current_feature_flag_context.get()

    if "duo_workflow_claude_sonnet_4" in feature_flags:
        return "claude-sonnet-4-20250514"
    if "duo_workflow_claude_3_7" in feature_flags:
        return "claude-3-7-sonnet-20250219"

    return "claude-3-5-sonnet-20241022"
