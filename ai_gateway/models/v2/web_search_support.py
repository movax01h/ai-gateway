"""Which model classes run web search themselves.

Web search reaches a model by one of two routes, and they are mutually exclusive:

- Provider-hosted: the model class turns `web_search_options` into a server-side tool the provider
    executes. Nothing runs locally, and results and citations come back inside the response.
- Fallback: the agent is handed an ordinary client-executed search tool that we invoke ourselves.

Callers use this module to pick a route. Offering both at once gives the model two ways to search
the same query and bills the fallback provider for work the model provider would have done.
"""

from ai_gateway.model_selection.models import ModelClassProvider

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
"""Model class providers that execute web search themselves.

`litellm` is deliberately absent, including for the providers whose platforms could run a search of
their own. Two reasons:

- `ChatLiteLLM.bind_tools` drops `web_search_options` outright, so no litellm-routed model is asked
    to search in the first place.
- Were it forwarded, LiteLLM converts Anthropic's `server_tool_use` block into a client-style
    `tool_calls` entry prefixed `srvtoolu_`. The agent then tries to execute a `web_search` tool that
    exists nowhere locally and the turn fails.

So every litellm-routed model takes the fallback route today. Should a single litellm provider gain
a working native path, this needs to become per-model rather than per-provider, since one
`ModelClassProvider.LITE_LLM` entry covers Vertex, Bedrock and the rest at once.
"""


def supports_native_web_search(model_class_provider: ModelClassProvider) -> bool:
    """Whether this model class provider executes web search itself.

    This answers only what the provider is capable of. Deciding what to do when no model has been resolved is the
    caller's policy call, not a fact about any provider.
    """
    return model_class_provider in NATIVE_WEB_SEARCH_PROVIDERS
