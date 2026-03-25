"""LangChain Runnable wrapper for embeddings endpoints via LiteLLM.

This module provides an embeddings adapter that integrates with the prompt registry for embeddings requests.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional

import litellm
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig, RunnableSerializable

__all__ = ["EmbeddingLiteLLM", "EmbeddingBadRequestError", "EmbeddingRateLimitError"]

logger = logging.getLogger(__name__)


class EmbeddingBadRequestError(Exception):
    pass


class EmbeddingRateLimitError(Exception):
    pass


class EmbeddingLiteLLM(RunnableSerializable[Dict[str, Any], AIMessage]):
    """Runnable wrapper for embeddings endpoints via LiteLLM.

    This model is designed to work with the prompt registry system. It accepts inputs with 'contents' keys, and returns
    an AIMessage for compatibility with the existing Prompt chain architecture.
    """

    model: str
    custom_llm_provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout: Optional[float] = 60.0
    max_retries: int = 1

    # define unused attribute to satisfy the LLMModelProtocol interface
    disable_streaming: bool = False

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("client", None)
        kwargs.pop("streaming", None)
        kwargs.pop("model_kwargs", None)

        super().__init__(**kwargs)

    @property
    def _default_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "custom_llm_provider": self.custom_llm_provider,
            "model": self.model,
        }
        return {k: v for k, v in params.items() if v is not None}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
        }

    @property
    def _llm_type(self) -> str:
        return "litellm-embedding"

    def _build_embedding_args(
        self,
        contents: list[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Get api_base and api_key from kwargs (bound from model_metadata) or fall back to instance
        api_base = kwargs.pop("api_base", None) or self.api_base
        api_key = kwargs.pop("api_key", None) or self.api_key
        vertex_location = kwargs.pop("vertex_location", None)

        embedding_args = {
            **self._default_params,
            "input": contents,
        }

        if api_base:
            embedding_args["api_base"] = api_base
        if api_key:
            embedding_args["api_key"] = api_key

        if vertex_location:
            embedding_args["vertex_ai_location"] = vertex_location

        return embedding_args

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        raise NotImplementedError("Sync invocation not implemented. Use ainvoke.")

    def stream(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        raise NotImplementedError(
            "Sync or async streaming not implemented. Use ainvoke."
        )

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,  # pylint: disable=unused-argument
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the embedding model.

        Args:
            contents: list of strings to embed
            config: Optional runnable config
            **kwargs: Additional arguments passed to completion

        Returns:
            AIMessage containing the embeddings
        """
        contents = input.get("contents", [])

        embedding_args = self._build_embedding_args(
            contents,
            **kwargs,
        )

        try:
            response = await litellm.aembedding(**embedding_args)
        except litellm.BadRequestError as e:
            raise EmbeddingBadRequestError(str(e)) from e
        except litellm.RateLimitError as e:
            raise EmbeddingRateLimitError(str(e)) from e

        predictions = self._extract_predictions(response)

        return AIMessage(content=predictions)

    async def astream(  # type: ignore[override]
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        raise NotImplementedError(
            "Sync or async streaming not implemented. Use ainvoke."
        )

    def _extract_predictions(self, response: Any) -> list[str | dict[Any, Any]]:
        if not hasattr(response, "data") or not response.data:
            raise ValueError(
                "Unexpected response format: missing or empty response data"
            )

        return [
            {
                "embedding": data.get("embedding", []),
                "index": data.get("index"),
            }
            for data in response.data
        ]
