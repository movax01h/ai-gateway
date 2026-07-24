from typing import Any

import litellm
from dependency_injector import containers, providers
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from ai_gateway.config import ConfigDuoWorkflow
from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.models import mock
from ai_gateway.models.base import init_anthropic_client, log_request
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.models.v2.chat_google_genai import (
    ChatGoogleGenerativeAI,
    connect_google_gen_vertex_ai,
)
from ai_gateway.models.v2.chat_litellm import ChatLiteLLM
from ai_gateway.models.v2.completion_litellm import CompletionLiteLLM
from ai_gateway.models.v2.embedding_litellm import EmbeddingLiteLLM
from ai_gateway.models.v2.openai import ChatOpenAI

__all__ = [
    "ContainerModels",
]

litellm.module_level_aclient = AsyncHTTPHandler(event_hooks={"request": [log_request]})


def _compute_fireworks_allowed_api_bases(
    fireworks_api_base_url: str,
) -> frozenset[str]:
    """Compute the set of allowed Fireworks API base URLs from operator configuration.

    Fireworks regional endpoints have been discontinued. The single ``fireworks_api_base_url``
    (configured via ``AIGW_FIREWORKS_API_BASE_URL``) is the only endpoint that needs to be
    allowlisted so that the SSRF protection in ``validate_custom_endpoint`` permits outgoing
    requests to Fireworks without requiring custom models to be enabled.
    """
    if fireworks_api_base_url.strip():
        return frozenset([fireworks_api_base_url.rstrip("/")])
    return frozenset()


def _init_google_chat_gen_vertex_ai_global_client(config: dict[str, Any]):
    client = connect_google_gen_vertex_ai(config["project"], "global")
    yield client
    client.close()


def _mock_selector(mock_model_responses: bool, use_agentic_mock: bool) -> str:
    if mock_model_responses and use_agentic_mock:
        return "agentic"

    if mock_model_responses:
        return "mocked"

    return "original"


class ContainerModels(containers.DeclarativeContainer):
    # We need to resolve the model based on the model name provided by the upstream container.
    # Hence, `ChatAnthropic` etc. are only partially applied here.

    config = providers.Configuration(strict=True)
    integrations = providers.DependenciesContainer()

    _mock_selector = providers.Callable(
        _mock_selector,
        config.mock_model_responses,
        config.use_agentic_mock,
    )

    http_async_client_anthropic = providers.Singleton(init_anthropic_client)

    _fireworks_allowed_api_bases = providers.Singleton(
        _compute_fireworks_allowed_api_bases,
        fireworks_api_base_url=config.fireworks_api_base_url,
    )

    _duo_workflow = providers.Callable(
        ConfigDuoWorkflow.model_validate, config.duo_workflow
    )

    anthropic_claude_chat_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            ChatAnthropic,
            async_client=http_async_client_anthropic,
            anthropic_api_url=_duo_workflow.provided.caching_proxy_url.call(),
            betas=[
                "context-1m-2025-08-07",
            ],
            custom_models_enabled=config.custom_models.enabled,
        ),
        mocked=providers.Factory(mock.FakeModel),
        agentic=providers.Factory(
            mock.AgenticFakeModel,
            auto_tool_approval=config.agentic_mock.auto_tool_approval,
            use_last_human_message=config.agentic_mock.use_last_human_message,
        ),
    )

    openai_chat_fn = providers.Factory(
        ChatOpenAI,
        output_version="responses/v1",
        custom_models_enabled=config.custom_models.enabled,
    )

    google_chat_gen_vertex_ai_global_fn = providers.Factory(
        ChatGoogleGenerativeAI,
        client=providers.Resource(
            _init_google_chat_gen_vertex_ai_global_client,
            config.google_cloud_platform,
        ),
        custom_models_enabled=config.custom_models.enabled,
    )

    lite_llm_chat_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            ChatLiteLLM,
            custom_models_enabled=config.custom_models.enabled,
            bedrock_guardrail_config=config.bedrock_guardrail_config,
            allowed_api_bases=_fireworks_allowed_api_bases,
            # Without this, litellm falls back to its default (~600s) timeout instead
            # of failing fast when a connection stalls.
            request_timeout=config.duo_chat.model_request_timeout,
        ),
        mocked=providers.Factory(mock.FakeModel),
        agentic=providers.Factory(
            mock.AgenticFakeModel,
            auto_tool_approval=config.agentic_mock.auto_tool_approval,
            use_last_human_message=config.agentic_mock.use_last_human_message,
        ),
    )

    amazon_q_chat_fn = providers.Factory(
        ChatAmazonQ,
        amazon_q_client_factory=integrations.amazon_q_client_factory,
    )

    lite_llm_completion_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            CompletionLiteLLM,
            custom_models_enabled=config.custom_models.enabled,
            allowed_api_bases=_fireworks_allowed_api_bases,
            bedrock_guardrail_config=config.bedrock_guardrail_config,
        ),
        mocked=providers.Factory(mock.FakeCompletionModel),
        agentic=providers.Factory(mock.AgenticFakeModel),
    )

    lite_llm_embedding_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            EmbeddingLiteLLM,
            custom_models_enabled=config.custom_models.enabled,
        ),
        mocked=providers.Factory(mock.FakeEmbeddingModel),
    )
