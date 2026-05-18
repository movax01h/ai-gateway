from collections.abc import Callable, Sequence
from typing import Any, override

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import _ChatModelBinding
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI as _LChatOpenAI

from ai_gateway.models.base import validate_custom_endpoint

__all__ = ["ChatOpenAI"]


class ChatOpenAI(_LChatOpenAI):
    custom_models_enabled: bool = False
    """Whether custom model endpoints are allowed."""

    @override
    def bind(self, **kwargs: Any) -> _ChatModelBinding:
        validate_custom_endpoint(
            self.custom_models_enabled,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
        )
        return super().bind(**kwargs)

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        tools_list = list(tools)
        web_search_options = kwargs.pop("web_search_options", None)
        if web_search_options is not None:
            tools_list.append({"type": "web_search"})

        return super().bind_tools(tools_list, **kwargs)
