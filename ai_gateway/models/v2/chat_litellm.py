from typing import Any, AsyncIterator, List, Optional

from langchain_community.chat_models.litellm import ChatLiteLLM as _LChatLiteLLM
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk

__all__ = ["ChatLiteLLM"]


class ChatLiteLLM(_LChatLiteLLM):
    """
    A wrapper around `langchain_community.chat_models.litellm.ChatLiteLLM`
    that adds custom stream_options
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

        async for chunk in super()._astream(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk
