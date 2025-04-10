import os

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langsmith import tracing_context


class VertexConfig:
    model = "claude-3-5-sonnet-v2@20241022"

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
            model_name=config.model,
            project=config.project_id,
            location=config.location,
            max_retries=config.max_retries,
            **kwargs,
        )

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    # Too bad these names are different across vertex-anthropic and regular-anthropic
    anthropic_model_name = "claude-3-5-sonnet-20241022"
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
    model_name = anthropic_response.response_metadata["model"]
    log.info("Connected to model: %s: %s", model_name, content)
