from collections.abc import Callable, Sequence
from typing import Any, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI as _LChatOpenAI

__all__ = ["ChatOpenAI"]


class ChatOpenAI(_LChatOpenAI):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Optional[dict[Any, Any] | str | bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        tools_list = list(tools)
        web_search_options = kwargs.pop("web_search_options", None)
        if web_search_options is not None:
            tools_list.append({"type": "web_search"})

        return super().bind_tools(tools_list, tool_choice=tool_choice, **kwargs)
