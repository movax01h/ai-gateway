"""LangChain Runnable wrapper for FiM/text completion endpoints via LiteLLM.

This module provides a completion model that integrates with the prompt registry and AgentModel for code completion use
cases.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    cast,
    override,
)

import litellm
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ai_gateway.model_selection.models import CompletionType
from ai_gateway.models.fireworks_retry import (
    DEFAULT_FIREWORKS_ERRORS,
    create_fireworks_retry_decorator,
)
from ai_gateway.vendor.langchain_litellm.litellm import _create_usage_metadata

__all__ = ["CompletionLiteLLM"]

logger = logging.getLogger(__name__)


MODEL_STOP_TOKENS: Dict[str, List[str]] = {
    "codestral-2501": [
        "\n\n",
        "\n+++++",
        "[PREFIX]",
        "</s>[SUFFIX]",
        "[MIDDLE]",
    ],
    "codestral-2508": [
        "\n\n",
        "\n+++++",
        "[PREFIX]",
        "</s>[SUFFIX]",
        "[MIDDLE]",
    ],
    "qwen2p5-coder-7b": [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|im_start|>",
        "<|im_end|>",
        "\n\n",
    ],
}


class CompletionLiteLLM(BaseChatModel):
    """Chat model wrapper for FiM and text completion endpoints via LiteLLM.

    Supports two completion types:
    - FIM: Format prefix/suffix into prompt string using fim_format template
    - TEXT: Pass suffix as native parameter to LiteLLM with text_completion=True

    This model is designed to work with the prompt registry system. It accepts
    inputs with 'prefix' and 'suffix' keys, formats them appropriately based on
    the completion_type, and returns an AIMessage for compatibility with the
    existing Prompt chain architecture.
    """

    model: str
    completion_type: CompletionType
    fim_format: Optional[str] = None
    custom_llm_provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    request_timeout: Optional[float] = 60.0
    max_retries: int = 1
    disable_streaming: bool = False

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_keys", None)
        kwargs.pop("client", None)
        kwargs.pop("streaming", None)
        kwargs.pop("model_kwargs", None)

        completion_type = kwargs.get("completion_type")
        fim_format = kwargs.get("fim_format")
        if completion_type == CompletionType.FIM and not fim_format:
            raise ValueError("fim_format is required when completion_type is 'fim'")

        super().__init__(**kwargs)

    @property
    def _default_params(self) -> Dict[str, Any]:
        # Note: 'model' is intentionally excluded here as it's set in _build_completion_args
        # to allow model_identifier from FireworksModelMetadata to override
        params: Dict[str, Any] = {
            "force_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "custom_llm_provider": self.custom_llm_provider,
        }
        return {k: v for k, v in params.items() if v is not None}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "completion_type": str(self.completion_type),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @property
    def _llm_type(self) -> str:
        return "litellm-completion"

    def _get_stop_tokens(self, model_name: str) -> List[str]:
        for key, tokens in MODEL_STOP_TOKENS.items():
            if key in model_name.lower():
                return tokens
        return []

    def _format_fim_prompt(self, prefix: str, suffix: str) -> str:
        if not self.fim_format:
            raise ValueError("fim_format is required for FIM completion type")
        return self.fim_format.format(prefix=prefix, suffix=suffix or "")

    def _build_completion_args(
        self,
        prompt: str,
        suffix: Optional[str],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Get api_base and api_key from kwargs (bound from model_metadata) or fall back to instance
        api_base = kwargs.pop("api_base", None) or self.api_base
        api_key = kwargs.pop("api_key", None) or self.api_key
        # model from kwargs overrides instance model (e.g., model_identifier from FireworksModelMetadata)
        model = kwargs.pop("model", None) or self.model

        completion_args = {
            **self._default_params,
            "model": model,
            "stream": stream,
        }

        if api_base:
            completion_args["api_base"] = api_base
        if api_key:
            completion_args["api_key"] = api_key

        model_stop_tokens = self._get_stop_tokens(model)
        if stop:
            completion_args["stop"] = list(set(stop + model_stop_tokens))
        elif model_stop_tokens:
            completion_args["stop"] = model_stop_tokens

        completion_args["prompt"] = prompt
        if self.completion_type == CompletionType.TEXT and suffix:
            completion_args["suffix"] = suffix

        if self.custom_llm_provider == "fireworks_ai":
            completion_args["logprobs"] = 1
            using_cache = kwargs.get("using_cache")
            if using_cache is not None:
                cache_str = str(using_cache).lower()
                if cache_str == "false":
                    completion_args["prompt_cache_max_len"] = 0
            session_id = kwargs.get("session_id")
            if session_id:
                completion_args.setdefault("extra_headers", {})
                completion_args["extra_headers"]["x-session-affinity"] = session_id

        if self.custom_llm_provider == "vertex_ai":
            completion_args["vertex_ai_location"] = kwargs.get(
                "vertex_ai_location", "us-central1"
            )

        return completion_args

    @override
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat result from messages.

        This method is not implemented for completion models. Use _agenerate instead.
        """
        raise NotImplementedError("Sync generation not implemented. Use _agenerate.")

    @override
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat generation from messages.

        This method is not implemented for completion models. Use _astream instead.
        """
        raise NotImplementedError("Sync streaming not implemented. Use _astream.")

    async def _acompletion_with_retry(self, **completion_args: Any) -> Any:
        """Execute text completion with retry logic for Fireworks provider."""
        if self.custom_llm_provider == "fireworks_ai":

            @create_fireworks_retry_decorator(logger, DEFAULT_FIREWORKS_ERRORS)
            async def _completion_with_retry() -> Any:
                return await litellm.atext_completion(**completion_args)

            return await _completion_with_retry()

        return await litellm.atext_completion(**completion_args)

    def _extract_prefix_suffix_from_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[str, str]:
        """Extract prefix and suffix from messages.

        For completion models, we expect the last message to contain prefix/suffix info. This is a compatibility layer
        for the BaseChatModel interface.
        """
        if not messages:
            return "", ""

        last_message = messages[-1]
        content = last_message.content

        # If content is a non-empty list, and the first item is a dict, use that
        if isinstance(content, list) and content:
            data = content[0]
            if isinstance(data, dict):
                return data.get("prefix", ""), data.get("suffix", "")

        # Otherwise, treat the entire content as prefix
        return str(content), ""

    @override
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat result from messages.

        Args:
            messages: List of messages
            stop: Optional stop sequences
            run_manager: Optional async callback manager
            **kwargs: Additional arguments passed to completion

        Returns:
            ChatResult containing the completion
        """
        prefix, suffix = self._extract_prefix_suffix_from_messages(messages)
        stop_sequences = kwargs.pop("stop", stop)

        if self.completion_type == CompletionType.FIM:
            formatted_prompt = self._format_fim_prompt(prefix, suffix)
            completion_suffix = None
        else:
            formatted_prompt = prefix
            completion_suffix = suffix

        completion_args = self._build_completion_args(
            formatted_prompt,
            completion_suffix,
            stop_sequences,
            False,
            **kwargs,
        )

        response = await self._acompletion_with_retry(**completion_args)
        text = self._extract_text(response)
        message = AIMessage(
            content=text,
            response_metadata=(
                {"model_name": response.model} if response.model else {}
            ),
            usage_metadata=(
                _create_usage_metadata(response.usage) if response.usage else None
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @override
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the completion model.

        Args:
            messages: List of messages
            stop: Optional stop sequences
            run_manager: Optional async callback manager
            **kwargs: Additional arguments passed to completion

        Yields:
            AIMessageChunk containing completion text chunks
        """
        if self.disable_streaming:
            result = await self._agenerate(messages, stop, run_manager, **kwargs)
            if result.generations:
                message = cast(AIMessage, result.generations[0].message)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=message.content,
                        response_metadata=message.response_metadata,
                        usage_metadata=message.usage_metadata,
                    )
                )
            return

        prefix, suffix = self._extract_prefix_suffix_from_messages(messages)
        stop_sequences = kwargs.pop("stop", stop)

        if self.completion_type == CompletionType.FIM:
            formatted_prompt = self._format_fim_prompt(prefix, suffix)
            completion_suffix = None
        else:
            formatted_prompt = prefix
            completion_suffix = suffix

        completion_args = self._build_completion_args(
            formatted_prompt,
            completion_suffix,
            stop_sequences,
            True,
            **kwargs,
        )

        response = await self._acompletion_with_retry(**completion_args)

        async for chunk in response:
            text = self._extract_chunk_text(chunk)
            if text:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=text,
                        response_metadata=(
                            {"model_name": chunk.model} if chunk.model else {}
                        ),
                        usage_metadata=(
                            _create_usage_metadata(chunk.usage) if chunk.usage else None
                        ),
                    )
                )

    def _extract_text(self, response: Any) -> str:
        return response.choices[0].text

    def _extract_chunk_text(self, chunk: Any) -> str:
        if not hasattr(chunk, "choices") or not chunk.choices:
            return ""
        choice = chunk.choices[0]
        return getattr(choice, "text", "") or ""
