"""LangChain Runnable wrapper for embeddings endpoints via LiteLLM.

This module provides an embeddings adapter that integrates with the prompt registry for embeddings requests.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, override

import litellm
from langchain_core.messages import AIMessage, AIMessageChunk, UsageMetadata
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable

from ai_gateway.models.base import validate_custom_endpoint
from ai_gateway.models.user_identity_header import inject_user_identity_header

__all__ = ["EmbeddingBadRequestError", "EmbeddingLiteLLM", "EmbeddingRateLimitError"]

logger = logging.getLogger(__name__)


class EmbeddingBadRequestError(Exception):
    pass


class EmbeddingRateLimitError(Exception):
    pass


class EmbeddingAuthenticationError(Exception):
    pass


class EmbeddingTimeoutError(Exception):
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
    custom_models_enabled: bool = False
    user_id_header: Optional[str] = None

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
            # litellm.embedding() defaults timeout to 600s, so an omitted key silently widens
            # the bound. Only `timeout` reaches the HTTP layer, not `request_timeout`.
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
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
        dimensions: Optional[int] = None,
        drop_params: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        embedding_args: dict[str, Any] = {**self._default_params, "input": contents}

        if dimensions:
            embedding_args["dimensions"] = dimensions

        if drop_params:
            embedding_args["drop_params"] = drop_params

        # Override model from default params if it is set in kwargs (bound from model_metadata)
        if model := kwargs.pop("model", None):
            embedding_args["model"] = model

        # Get api_base and api_key from kwargs (bound from model_metadata) or fall back to instance
        api_base = kwargs.pop("api_base", None) or self.api_base
        api_key = kwargs.pop("api_key", None) or self.api_key
        if api_base:
            embedding_args["api_base"] = api_base
        if api_key:
            embedding_args["api_key"] = api_key

        if vertex_location := kwargs.pop("vertex_location", None):
            embedding_args["vertex_ai_location"] = vertex_location

        inject_user_identity_header(embedding_args, self.user_id_header)

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
        dimensions = input.get("dimensions", None)
        drop_params = input.get("drop_params", None)

        embedding_args = self._build_embedding_args(
            contents,
            dimensions,
            drop_params,
            **kwargs,
        )

        try:
            response = await litellm.aembedding(**embedding_args)
        except litellm.BadRequestError as e:
            raise EmbeddingBadRequestError(str(e)) from e
        except litellm.RateLimitError as e:
            raise EmbeddingRateLimitError(str(e)) from e
        except litellm.AuthenticationError as e:
            raise EmbeddingAuthenticationError(str(e)) from e
        except litellm.Timeout as e:
            raise EmbeddingTimeoutError(str(e)) from e

        predictions = self._extract_predictions(response)
        usage_metadata = self._extract_usage_metadata(response)

        return AIMessage(content=predictions, usage_metadata=usage_metadata)

    @override
    def bind(self, **kwargs: Any) -> "Runnable[Dict[str, Any], AIMessage]":
        validate_custom_endpoint(
            self.custom_models_enabled,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
        )
        return super().bind(**kwargs)

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

    def _extract_usage_metadata(self, response: Any) -> UsageMetadata:
        usage = getattr(response, "usage", None)
        if not usage:
            return UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)

        # completion_tokens_details->output_token_details is not mapped because the
        # field is always None for embeddings responses
        usage_metadata = UsageMetadata(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        if input_token_details := self._extract_input_token_details(usage):
            usage_metadata["input_token_details"] = input_token_details  # type: ignore[typeddict-item]

        return usage_metadata

    def _extract_input_token_details(self, usage: Any) -> dict[str, int]:
        details = getattr(usage, "prompt_tokens_details", None)
        if not details:
            return {}

        cache_creation_details = getattr(details, "cache_creation_token_details", None)
        mapping = {
            "cache_read": getattr(details, "cached_tokens", None),
            "cache_creation": getattr(details, "cache_creation_tokens", None),
            "ephemeral_5m_input_tokens": getattr(
                cache_creation_details, "ephemeral_5m_input_tokens", None
            ),
            "ephemeral_1h_input_tokens": getattr(
                cache_creation_details, "ephemeral_1h_input_tokens", None
            ),
        }

        # The keys are NotRequired, so omit unset ones rather than report them as None
        return {key: value for key, value in mapping.items() if value is not None}
