from collections.abc import Callable, Sequence
from typing import Any, override

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import _ChatModelBinding
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI as _LChatOpenAI

from ai_gateway.models.base import validate_custom_endpoint
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

__all__ = ["ChatOpenAI"]

_WEB_SEARCH_MAX_TOOL_CALLS = 5
"""Cap on built-in tool calls per request, matching Anthropic's `web_search` `max_uses`.

The Responses API has no per-tool limit. `max_tool_calls` counts every built-in call, and `web_search` is the only
built-in tool bound here, so it caps searches. Page reads (`open_page`, `find_in_page`) count against it too.
Past it, the API ignores further calls and the model
answers from what it already found.
"""

_WEB_SEARCH_INCLUDES = (
    # Which search matched what; citations alone do not say.
    "web_search_call.action.sources",
    # The only titles for sources the answer never cited; `action.sources` has none.
    "web_search_call.results",
)


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
        # Without `strict`, the Responses API makes each tool schema strict when it can. That makes every optional
        # argument required, so GPT fills them with made-up defaults such as 0, "" or the first enum value.
        if (
            is_feature_enabled(FeatureFlag.DAP_OPENAI_NON_STRICT_TOOLS)
            and kwargs.get("strict") is None
        ):
            kwargs["strict"] = False

        web_search_options = kwargs.pop("web_search_options", None)
        if web_search_options is not None:
            tools_list.append({"type": "web_search"})
            kwargs["max_tool_calls"] = _WEB_SEARCH_MAX_TOOL_CALLS
            # A bound `include` wins over the field; carry the field's entries along.
            bound = kwargs["include"] if "include" in kwargs else self.include
            kwargs["include"] = list(
                dict.fromkeys([*(bound or []), *_WEB_SEARCH_INCLUDES])
            )

        return super().bind_tools(tools_list, **kwargs)
