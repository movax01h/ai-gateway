from collections.abc import Callable, Sequence
from typing import Any, List, Mapping, Optional, Self

from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic as _LChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import model_validator

__all__ = ["ChatAnthropic"]


class ChatAnthropic(_LChatAnthropic):
    """A wrapper around `langchain_anthropic.ChatAnthropic` that accepts the Anthropic asynchronous client as an input
    parameter."""

    async_client: AsyncAnthropic
    """Anthropic async HTTP client."""

    default_request_timeout: float | None = 60.0
    """Timeout for requests to Anthropic Completion API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 1
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

    default_headers: Mapping[str, str] = {"anthropic-version": "2023-06-01"}
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    betas: Optional[list[str]] = None
    """Beta features to enable for the Anthropic client."""

    def _get_combined_headers(self) -> Optional[dict[str, str]]:
        """Get combined headers including default headers and beta features.

        This method doesn't modify any instance variables, it just computes and returns
        the appropriate headers based on the current state.

        Returns:
            Optional[dict[str, str]]: The combined headers or None if no headers.
        """
        headers = dict(self.default_headers) if self.default_headers else {}

        # Add beta header if beta features are specified
        if self.betas:
            headers["anthropic-beta"] = ",".join(self.betas)

        return headers or None

    @model_validator(mode="after")
    def post_init(self) -> Self:
        client_options: dict[str, Any] = {
            "api_key": self.anthropic_api_key.get_secret_value(),
            "base_url": self.anthropic_api_url,
        }

        headers = self._get_combined_headers()
        if headers:
            client_options["default_headers"] = headers

        client_options.update(
            {
                "max_retries": self.max_retries,
            }
        )

        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_options["timeout"] = self.default_request_timeout

        async_client: AsyncAnthropic = self.async_client
        self._async_client = async_client.with_options(**client_options)

        # hack: we don't use sync methods in the AIGW,
        # so to avoid unnecessary initialization, set None for the sync client
        self._client = None  # type: ignore[assignment]

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError()

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Optional[dict[str, str] | str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        tools_list = list(tools)
        web_search_options = kwargs.pop("web_search_options", None)
        if web_search_options is not None:
            tools_list.append({"type": "web_search_20250305", "name": "web_search"})

        return super().bind_tools(tools_list, tool_choice=tool_choice, **kwargs)
