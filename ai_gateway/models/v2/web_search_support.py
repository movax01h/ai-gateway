"""Which models run web search themselves.

Web search reaches a model by one of two routes, and they are mutually exclusive:

- Provider-hosted: the model class turns `web_search_options` into a server-side tool the provider
    executes. Nothing runs locally, and results and citations come back inside the response.
- Fallback: the agent is handed an ordinary client-executed search tool that we invoke ourselves.

Callers use this module to pick a route. Offering both at once gives the model two ways to search
the same query and bills the fallback provider for work the model provider would have done.
"""

from typing import Any, Optional, Sequence

import structlog

from ai_gateway.model_selection import LLMDefinition
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.models.v2.chat_litellm import litellm_supports_native_web_search
from ai_gateway.models.v2.web_search_policy import NATIVE_WEB_SEARCH_ALLOWED_PROVIDERS

log = structlog.stdlib.get_logger("web_search_support")

__all__ = [
    "NATIVE_WEB_SEARCH_PROVIDERS",
    "WEB_SEARCH_TOOL_NAME",
    "bind_web_search",
    "supports_native_web_search",
]

WEB_SEARCH_TOOL_NAME = "web_search"
"""The tool a caller lists to offer web search, whichever route the model then takes."""


NATIVE_WEB_SEARCH_PROVIDERS = frozenset(
    {
        # `ChatAnthropic.bind_tools` appends a real `web_search` server tool and returns
        # `web_search_tool_result` blocks with citations intact.
        ModelClassProvider.ANTHROPIC,
        # `ChatOpenAI` is wired with `output_version="responses/v1"`, so `bind_tools` appending
        # `{"type": "web_search"}` reaches the Responses API built-in tool.
        ModelClassProvider.OPENAI,
    }
)
"""Model class providers where every model executes web search itself.

`litellm` is deliberately absent, because there the answer is per-model rather than per-provider:
one `ModelClassProvider.LITE_LLM` entry covers Vertex, Bedrock and the rest at once, and only some
of those platforms run a hosted search. `supports_native_web_search` asks
`litellm_supports_native_web_search` about the individual model instead.
"""


def supports_native_web_search(definition: LLMDefinition) -> bool:
    """Whether this model executes web search itself.

    A model that could search but whose provider isn't cleared in `web_search_policy` counts as unable. Deciding what to
    do when no model has been resolved is the caller's policy call, not a fact about any model.
    """
    provider = definition.model_class_provider

    if provider in NATIVE_WEB_SEARCH_PROVIDERS:
        return provider in NATIVE_WEB_SEARCH_ALLOWED_PROVIDERS

    if provider is not ModelClassProvider.LITE_LLM:
        return False

    # Only some litellm-routed platforms run a hosted search, so this is decided per model.
    params = definition.params
    return litellm_supports_native_web_search(
        getattr(params, "model", None),
        getattr(params, "custom_llm_provider", None),
    )


def bind_web_search(
    tools: Sequence[Any],
    llm_definition: Optional[LLMDefinition],
    bind_tools_params: Optional[dict[str, Any]],
) -> tuple[list[Any], Optional[dict[str, Any]]]:
    """Pick the web search route for this model, given what a caller wants to bind.

    A caller offers web search in one of two ways:

    - A `web_search` tool among `tools`. A model that searches natively gets `web_search_options`
        instead, and the tool object is dropped so the model has one way to search. Any other model
        keeps the tool when it can run locally (`locally_available`, the fallback gateway is
        configured), and gets nothing otherwise.
    - `web_search_options` in `bind_tools_params`, for callers that offer native search only (the
        `enable_web_search` alias). A model that can't search natively has it dropped, so the
        prompt isn't told it can search.

    With no resolved model, native is assumed: the default models search natively, and it risks
    no search rather than a duplicated, billable one.
    """
    params = bind_tools_params
    native_only_offer = params is not None and "web_search_options" in params
    offer = next(
        (tool for tool in tools if getattr(tool, "name", None) == WEB_SEARCH_TOOL_NAME),
        None,
    )
    if offer is None and not native_only_offer:
        return list(tools), params

    # Every prompt with tools binds through here; only an offer needs the model's capabilities.
    native = llm_definition is None or supports_native_web_search(llm_definition)
    if params is not None and native_only_offer and not native:
        params = {k: v for k, v in params.items() if k != "web_search_options"}

    if offer is None:
        _log_route("native" if native else "none", "web_search_options", llm_definition)
        return list(tools), params

    others = [tool for tool in tools if tool is not offer]
    if native:
        _log_route("native", "tool", llm_definition)
        return others, {**(params or {}), "web_search_options": {}}
    if getattr(offer, "locally_available", False):
        _log_route("fallback", "tool", llm_definition)
        return list(tools), params
    _log_route("none", "tool", llm_definition)
    return others, params


def _log_route(route: str, offer: str, llm_definition: Optional[LLMDefinition]) -> None:
    """Record why a session did or didn't get web search; only called when a caller offered it."""
    log.info(
        "Bound web search route",
        route=route,
        offer=offer,
        model=llm_definition.gitlab_identifier if llm_definition else None,
    )
