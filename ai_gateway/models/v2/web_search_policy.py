"""Which providers Privacy cleared to run web search themselves.

This is a legal decision, separate from what a model can do (`web_search_support`). A model that can search natively
but whose provider isn't listed here gets the AgentCore fallback instead. Clear a new provider by adding it here.

Two sets, because LiteLLM's `custom_llm_provider` is its own namespace: there `openai` also covers self-hosted
OpenAI-compatible endpoints, which a shared set of strings would clear by accident.
"""

from ai_gateway.model_selection.models import ModelClassProvider

__all__ = [
    "NATIVE_WEB_SEARCH_ALLOWED_LITELLM_PROVIDERS",
    "NATIVE_WEB_SEARCH_ALLOWED_PROVIDERS",
]


NATIVE_WEB_SEARCH_ALLOWED_PROVIDERS: frozenset[ModelClassProvider] = frozenset(
    {ModelClassProvider.ANTHROPIC, ModelClassProvider.OPENAI}
)
"""Model class providers cleared for native web search."""

NATIVE_WEB_SEARCH_ALLOWED_LITELLM_PROVIDERS: frozenset[str] = frozenset()
"""LiteLLM `custom_llm_provider` values cleared for native web search.

Vertex (`vertex_ai`) is not cleared: it sends queries to Brave outside our Anthropic DPA.
"""
