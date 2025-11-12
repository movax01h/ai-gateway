from typing import Any, AsyncIterator, List, Mapping, Optional

import langchain_community.chat_models.litellm
from langchain_community.chat_models.litellm import ChatLiteLLM as _LChatLiteLLM
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk

__all__ = ["ChatLiteLLM"]


class ChatLiteLLM(_LChatLiteLLM):
    """A wrapper around `langchain_community.chat_models.litellm.ChatLiteLLM` that adds custom stream_options."""

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Always include usage metrics when streaming. See https://docs.litellm.ai/docs/completion/usage#streaming-usage
        # Respect other possible values that may have been passed.
        kwargs["stream_options"] = {
            **kwargs.get("stream_options", {}),
            "include_usage": True,
        }

        async for chunk in super()._astream(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


def _create_usage_metadata(token_usage: Mapping[str, Any]) -> UsageMetadata:
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    extra_kwargs = {}

    if (
        "cache_creation_input_tokens" in token_usage
        and "cache_read_input_tokens" in token_usage
    ):
        cache_creation_input_tokens = token_usage.get("cache_creation_input_tokens", 0)
        cache_read_input_tokens = token_usage.get("cache_read_input_tokens", 0)

        extra_kwargs["input_token_details"] = InputTokenDetails(
            cache_creation=cache_creation_input_tokens,
            cache_read=cache_read_input_tokens,
        )

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        **extra_kwargs,  # type: ignore[typeddict-item]
    )


# Overriding `_create_usage_metadata` method in `langchain_community.chat_models.litellm` module
# to return usage metadata with prompt caching info.
# See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1642
# for more info.
langchain_community.chat_models.litellm._create_usage_metadata = _create_usage_metadata
