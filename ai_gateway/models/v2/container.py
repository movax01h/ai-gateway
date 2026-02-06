import litellm
from dependency_injector import containers, providers
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.models import mock
from ai_gateway.models.base import init_anthropic_client, log_request
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.models.v2.chat_litellm import ChatLiteLLM
from ai_gateway.models.v2.completion_litellm import CompletionLiteLLM
from ai_gateway.models.v2.openai import ChatOpenAI

__all__ = [
    "ContainerModels",
]

litellm.module_level_aclient = AsyncHTTPHandler(event_hooks={"request": [log_request]})


def _litellm_factory(*args, **kwargs) -> ChatLiteLLM:
    if kwargs.get("custom_llm_provider", "") == "vertex_ai":
        if kwargs.get("model", "").lower().startswith("claude"):
            kwargs["model_kwargs"] = kwargs.get("model_kwargs", {}) or {}
            kwargs["model_kwargs"]["extra_headers"] = {
                **kwargs["model_kwargs"].get("extra_headers", {}),
                "anthropic-beta": "fine-grained-tool-streaming-2025-05-14,context-1m-2025-08-07",
            }

    return ChatLiteLLM(*args, **kwargs)


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

    anthropic_claude_chat_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            ChatAnthropic,
            async_client=http_async_client_anthropic,
            betas=[
                "extended-cache-ttl-2025-04-11",
                "context-1m-2025-08-07",
                "fine-grained-tool-streaming-2025-05-14",
            ],
        ),
        mocked=providers.Factory(mock.FakeModel),
        agentic=providers.Factory(
            mock.AgenticFakeModel,
            auto_tool_approval=config.agentic_mock.auto_tool_approval,
            use_last_human_message=config.agentic_mock.use_last_human_message,
        ),
    )

    openai_chat_fn = providers.Factory(ChatOpenAI, output_version="responses/v1")

    lite_llm_chat_fn = providers.Selector(
        _mock_selector,
        original=providers.Factory(
            _litellm_factory,
            model_keys=config.model_keys,
            model_endpoints=config.model_endpoints,
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
            model_keys=config.model_keys,
            model_endpoints=config.model_endpoints,
        ),
        mocked=providers.Factory(mock.FakeCompletionModel),
        agentic=providers.Factory(mock.AgenticFakeModel),
    )
