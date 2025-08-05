import json
import re
from typing import Any, AsyncIterator, Callable, List, Optional, TypeVar
from unittest.mock import AsyncMock

import fastapi
from fastapi import status
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from ai_gateway.models.base import ModelMetadata
from ai_gateway.models.base_chat import ChatModelBase, Message
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes

__all__ = [
    "AsyncStream",
    "LLM",
    "ChatModel",
]

_T = TypeVar("_T")


class AsyncStream(AsyncIterator[_T]):
    def __init__(self, chunks: list[_T], callback_finish: Optional[Callable] = None):
        self.chunks = chunks
        self.callback_finish = callback_finish

    def __aiter__(self) -> "AsyncStream[_T]":
        return self

    async def __anext__(self) -> _T:
        if len(self.chunks) > 0:
            return self.chunks.pop(0)

        if self.callback_finish:
            self.callback_finish()

        raise StopAsyncIteration


class ProxyClient(AsyncMock):  # pylint: disable=too-many-ancestors
    async def proxy(self, *_args, **_kwargs):
        return fastapi.Response(
            content=json.dumps({"response": "mocked"}).encode("utf-8"),
            status_code=status.HTTP_200_OK,
            headers={"Content-Type": "application/json"},
        )


class LLM(TextGenModelBase):
    """Implementation of the stub model that inherits the `TextGenBaseModel` interface.

    Please, use this class if you require to mock such models as `AnthropicModel` or `PalmCodeGeckoModel`
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        super().__init__()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="llm-mocked", engine="llm-provider-mocked")

    async def generate(
        self,
        prefix: str,
        suffix: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> (
        TextGenModelOutput | list[TextGenModelOutput] | AsyncIterator[TextGenModelChunk]
    ):
        scope = {
            "prefix": prefix,
            "suffix": suffix,
            "stream": stream,
            "kwargs": dict(kwargs),
        }
        if temperature is not None:
            scope["temperature"] = str(temperature)
        if max_output_tokens is not None:
            scope["max_output_tokens"] = str(max_output_tokens)
        if top_p is not None:
            scope["top_p"] = str(top_p)
        if top_k is not None:
            scope["top_k"] = str(top_k)

        # echo the current scope's local variables
        # default=vars prevents object is not JSON serializable error
        suggestion = f"echo: {json.dumps(scope, default=vars)}"

        with self.instrumentator.watch(stream=stream) as watcher:
            if stream:
                chunks = [
                    TextGenModelChunk(text=chunk)
                    for chunk in re.split(r"(\s)", suggestion)
                ]
                return AsyncStream(chunks, watcher.finish)

        return TextGenModelOutput(
            text=suggestion,
            score=0,
            safety_attributes=SafetyAttributes(),
        )


class FakeModel(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-model"}

    def _call(
        self,
        messages: List[BaseMessage],  # pylint: disable=unused-argument
        stop: Optional[List[str]] = None,  # pylint: disable=unused-argument
        # pylint: disable=unused-argument
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **_kwargs: Any,
    ) -> str:
        return "mock"

    def bind_tools(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> Any:
        return self


class ChatModel(ChatModelBase):
    """Implementation of the stub model that inherits the `ChatModelBase` interface.

    Please, use this class if you require to mock such models as `AnthropicChatModel`
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        super().__init__()

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="chat-model-mocked", engine="chat-model-provider-mocked"
        )

    async def generate(
        self,
        messages: list[Message],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> TextGenModelOutput | AsyncIterator[TextGenModelChunk]:

        serialized_messages = [message.model_dump(mode="json") for message in messages]
        scope = {
            "messages": serialized_messages,
            "stream": stream,
            "kwargs": dict(kwargs),
        }
        if temperature is not None:
            scope["temperature"] = str(temperature)
        if max_output_tokens is not None:
            scope["max_output_tokens"] = str(max_output_tokens)
        if top_p is not None:
            scope["top_p"] = str(top_p)
        if top_k is not None:
            scope["top_k"] = str(top_k)
        suggestion = (
            f"echo: {json.dumps(scope)}"  # echo the current scope's local variables
        )

        with self.instrumentator.watch(stream=stream) as watcher:
            if stream:
                chunks = [
                    TextGenModelChunk(text=chunk)
                    for chunk in re.split(r"(\s)", suggestion)
                ]
                return AsyncStream(chunks, watcher.finish)

        return TextGenModelOutput(
            text=suggestion,
            score=0,
            safety_attributes=SafetyAttributes(),
        )
