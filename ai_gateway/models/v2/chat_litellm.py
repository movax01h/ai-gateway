from typing import Any, AsyncIterator, List, Mapping, Optional, Type, cast

import langchain_community.chat_models.litellm
import structlog
from langchain_community.chat_models.litellm import ChatLiteLLM as _LChatLiteLLM
from langchain_community.chat_models.litellm import (
    _convert_delta_to_message_chunk,
    acompletion_with_retry,
)
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk

__all__ = ["ChatLiteLLM"]

log = structlog.getLogger(__name__)


class ChatLiteLLM(_LChatLiteLLM):
    """Patch https://github.com/langchain-ai/langchain-community/blob/libs/community/v0.3.27/libs/community/
    langchain_community/chat_models/litellm.py#L490
    This patch adds custom stream_options and correctly handles finish_reason support.
    NOTE: langchain_community's litellm adapter is deprecated
    (https://github.com/langchain-ai/langchain-community/pull/11) and no longer maintained.
    """

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

        # We need to intercept the raw chunks to extract finish_reason before LangChain processes them
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        added_model_name = False
        async for raw_chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(raw_chunk, dict):
                raw_chunk = raw_chunk.model_dump()
            if len(raw_chunk["choices"]) == 0:
                continue

            # Extract finish_reason from the raw chunk BEFORE processing
            finish_reason = raw_chunk["choices"][0].get("finish_reason")

            delta = raw_chunk["choices"][0]["delta"]
            usage = raw_chunk.get("usage", {})
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            if isinstance(chunk, AIMessageChunk):
                if not added_model_name:
                    chunk.response_metadata = {
                        "model_name": self.model_name or self.model
                    }
                    added_model_name = True

                # Add finish_reason to response_metadata
                # LiteLLM normalizes all provider responses to OpenAI format
                if finish_reason:
                    chunk.response_metadata["finish_reason"] = finish_reason

                chunk.usage_metadata = _create_usage_metadata(usage)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                # Extract string content for callback
                # Note: chunk.content can be str | list[str | dict[Any, Any]] per LangChain types,
                # but on_llm_new_token expects only str. The original LangChain code doesn't handle
                # this type mismatch. We explicitly handle both cases for type safety.
                content = chunk.content
                if isinstance(content, list):
                    # Handle list content by joining text parts
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                await run_manager.on_llm_new_token(cast(str, content), chunk=cg_chunk)
            yield cg_chunk


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
