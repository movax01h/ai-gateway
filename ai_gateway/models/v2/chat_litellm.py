from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, override

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ai_gateway.models.base import validate_custom_endpoint
from ai_gateway.models.v2._model_compat import (
    remove_trailing_assistant_message,
    supports_assistant_prefill,
)
from ai_gateway.models.v2.litellm_model_registry import register_external_models
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM as _LChatLiteLLM

__all__ = ["ChatLiteLLM"]

# Register any additional model metadata supplied by operators via an external
# JSON file (path set by AIGW_LITELLM__MODEL_METADATA_FILE). It allows operators
# to enable parameters like `tool_choice` for models that are missing from
# LiteLLM's built-in registry, without running a LiteLLM proxy.
register_external_models()


class ChatLiteLLM(_LChatLiteLLM):
    custom_models_enabled: bool = False
    allowed_api_bases: frozenset[str] = frozenset()

    @override
    def bind(self, **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        validate_custom_endpoint(
            self.custom_models_enabled,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
            allowed_api_bases=self.allowed_api_bases,
            custom_llm_provider=kwargs.get("custom_llm_provider")
            or self.custom_llm_provider,
        )
        return super().bind(**kwargs)

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        kwargs.pop("web_search_options", None)  # Not yet supported for LiteLLM

        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    @override
    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts, params = super()._create_message_dicts(messages, stop)
        model_name = self.model_name or self.model
        if not supports_assistant_prefill(model_name):
            payload = remove_trailing_assistant_message({"messages": message_dicts})
            message_dicts = payload["messages"]
        return message_dicts, params

    @property
    @override
    def _client_params(self) -> dict[str, Any]:
        """Ensure api_key is passed as a kwarg to async litellm.acompletion.

        Upstream _client_params mutates self.client.api_key but does not include api_key in the returned kwargs. That
        works for the sync path (module-level litellm.api_key) but fails for async acompletion, which reads api_key from
        kwargs.
        """
        params = super()._client_params
        if self.api_key and "api_key" not in params:
            params["api_key"] = self.api_key
        return params
