"""Which models run web search themselves.

Web search reaches a model by one of two routes, and they are mutually exclusive:

- Provider-hosted: the model class turns `web_search_options` into a server-side tool the provider
    executes. Nothing runs locally, and results and citations come back inside the response.
- Fallback: the agent is handed an ordinary client-executed search tool that we invoke ourselves.

Callers use this module to pick a route. Offering both at once gives the model two ways to search
the same query and bills the fallback provider for work the model provider would have done.
"""

from ai_gateway.model_selection import LLMDefinition
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.models.v2.chat_litellm import litellm_supports_native_web_search

__all__ = ["NATIVE_WEB_SEARCH_PROVIDERS", "supports_native_web_search"]


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

    This answers only what the model is capable of. Deciding what to do when no model has been resolved is the caller's
    policy call, not a fact about any model.
    """
    provider = definition.model_class_provider

    if provider in NATIVE_WEB_SEARCH_PROVIDERS:
        return True

    if provider is not ModelClassProvider.LITE_LLM:
        return False

    # Only some litellm-routed platforms run a hosted search, so this is decided per model.
    params = definition.params
    return litellm_supports_native_web_search(
        getattr(params, "model", None),
        getattr(params, "custom_llm_provider", None),
    )
