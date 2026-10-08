"""The matrix: every litellm entry in models.yml, and the parameter sets sent through each one."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ai_gateway.model_metadata import (
    create_model_metadata,
    resolve_provider_aware_metadata,
)
from ai_gateway.model_selection import ModelSelectionConfig

CHAT, COMPLETION, EMBEDDING = "litellm", "litellm_completion", "litellm_embedding"

PROMPT_ID = {
    CHAT: "model_configuration/check",
    COMPLETION: "code_suggestions/completions",
    EMBEDDING: "embeddings_code",
}

# Entries that are not run. Each needs a reason, and the entry must still exist in models.yml.
SKIPS: dict[str, str] = {
    "gemini_2_5_flash_vertex": "deprecated, removal 19.5",
}

# (route, parameter set) pairs that are not run, with a reason. A route is the entry's custom_llm_provider, and None means
# a self-hosted family. Add a pair only when a provider legitimately rejects a set that production never sends to it.
# Keying by route means a new entry on an existing route needs no edit here.
SET_SKIPS: dict[tuple[str | None, str], str] = {
    ("bedrock", "tool_calls_without_tools"): (
        "litellm requires tools= for Bedrock when the history contains tool calls (UnsupportedParamsError)"
    ),
    ("bedrock_mantle", "temperature"): (
        "litellm accepts only temperature=1 for the GPT-5.x reasoning models on Mantle (UnsupportedParamsError)"
    ),
}

SELFHOSTED_BASE = "http://fake-selfhosted.invalid/v1"


@tool
def get_weather(city: str) -> str:
    """Get the weather."""
    return "sunny"


@dataclass(frozen=True)
class ParamSet:
    """One way of calling a model.

    Each set is backed by an incident or by a parameter every route must accept.
    """

    name: str
    why: str
    tools: bool = False  # pass tools and tool_choice through the registry
    history: bool = False  # send a conversation that contains tool calls
    bind: dict[str, Any] = field(default_factory=dict)


CHAT_SETS = [
    ParamSet("plain", "baseline request"),
    ParamSet(
        "tools_tool_choice", "tool_choice rejected for Fireworks (#2587)", tools=True
    ),
    ParamSet(
        "tool_calls_without_tools",
        "tool_calls sent without tools (#2121)",
        history=True,
    ),
    ParamSet(
        "max_tokens",
        "max_tokens rejected for GPT-5 on custom_openai (!5707)",
        bind={"max_tokens": 50},
    ),
    ParamSet(
        "temperature",
        "temperature deprecated by newer Anthropic models",
        bind={"temperature": 0.2},
    ),
]
COMPLETION_SETS = [
    ParamSet("plain", "prefix only"),
    ParamSet("fim", "fill in the middle with a suffix"),
]
EMBEDDING_SETS = [ParamSet("plain", "baseline request")]

SETS_BY_KIND = {CHAT: CHAT_SETS, COMPLETION: COMPLETION_SETS, EMBEDDING: EMBEDDING_SETS}


@dataclass(frozen=True)
class Entry:
    ident: str
    kind: str
    provider: (
        str | None
    )  # custom_llm_provider from models.yml. None means a self-hosted family.
    model: str | None  # params.model from models.yml


def load_entries() -> list[Entry]:
    """Read every litellm entry through the repo's own loader."""
    entries = []
    for ident, definition in (
        ModelSelectionConfig.instance().get_llm_definitions().items()
    ):
        kind = str(
            getattr(
                definition.model_class_provider,
                "value",
                definition.model_class_provider,
            )
        )
        if kind not in SETS_BY_KIND:
            continue
        params = definition.params
        entries.append(
            Entry(
                ident=ident,
                kind=kind,
                provider=getattr(params, "custom_llm_provider", None),
                model=getattr(params, "model", None),
            )
        )
    return sorted(entries, key=lambda entry: entry.ident)


def metadata_for(entry: Entry, config: Any) -> Any:
    """Build the model metadata the way production does for this kind of entry."""
    if entry.provider is None:
        # A self-hosted family. The route is defined by the request: provider, identifier, endpoint and key.
        return create_model_metadata(
            {
                "provider": "litellm",
                "name": entry.ident,
                "identifier": f"custom_openai/{entry.ident}",
                "endpoint": SELFHOSTED_BASE,
                "api_key": "fake",
            }
        )

    return resolve_provider_aware_metadata(
        ModelSelectionConfig.instance().get_model(entry.ident),
        provider_keys={"fireworks_provider_api_key": "fake"},
        fireworks_api_base_url=config.fireworks_api_base_url,
    )


def expected_route(entry: Entry) -> tuple[str, str, str]:
    """Return the (method, host, path pattern) litellm must call for this entry.

    An unknown combination raises, so a new provider cannot be added without saying where its request goes.
    """
    provider, kind = entry.provider, entry.kind
    routes = {
        ("fireworks_ai", CHAT): (
            "POST",
            "api.fireworks.ai",
            r"/inference/v1/chat/completions$",
        ),
        ("fireworks_ai", COMPLETION): (
            "POST",
            "api.fireworks.ai",
            r"/inference/v1/completions$",
        ),
        ("bedrock_mantle", CHAT): ("POST", "bedrock-mantle", r"/openai/v1/responses$"),
        ("openai", EMBEDDING): ("POST", "api.openai.com", r"/v1/embeddings$"),
        ("vertex_ai", EMBEDDING): ("POST", "aiplatform.googleapis.com", r":predict$"),
        ("vertex_ai", COMPLETION): (
            "POST",
            "aiplatform.googleapis.com",
            r"publishers/mistralai/models/[^/]+:rawPredict$",
        ),
        ("vertex_ai", CHAT): (
            "POST",
            "aiplatform.googleapis.com",
            r"publishers/anthropic/models/[^/]+:rawPredict$",
        ),
        # litellm sends some Bedrock Claude models to /invoke and the rest to /converse. Either is accepted on purpose,
        # so a release that moves a model between them stays green.
        ("bedrock", CHAT): ("POST", "bedrock-runtime", r"/(converse|invoke)$"),
        (None, CHAT): ("POST", "fake-selfhosted.invalid", r"/v1/chat/completions$"),
    }
    try:
        return routes[(provider, kind)]
    except KeyError:
        raise LookupError(
            f"No expected route for {entry.ident} (custom_llm_provider={provider!r}, kind={kind}). "
            "Add it to expected_route() in litellm_param_tests/cases.py."
        ) from None


async def run_case(harness: Any, entry: Entry, pset: ParamSet) -> Any:
    """Build the model through the production registry and send one parameter set through it."""
    selfhosted = entry.provider is None
    config = harness.selfhosted_config if selfhosted else harness.managed_config
    registry = harness.selfhosted_registry if selfhosted else harness.managed_registry
    metadata = metadata_for(entry, config)

    if entry.kind == COMPLETION:
        prompt = registry.get(PROMPT_ID[COMPLETION], "^1.0.0", model_metadata=metadata)
        inputs = {"prefix": "def f("}
        if pset.name == "fim":
            inputs["suffix"] = "):"
        return await prompt.ainvoke(inputs)

    if entry.kind == EMBEDDING:
        prompt = registry.get(PROMPT_ID[EMBEDDING], "^1.0.0", model_metadata=metadata)
        return await prompt.ainvoke({"contents": ["hello"]})

    kwargs: dict[str, Any] = (
        {"tools": [get_weather], "tool_choice": "auto"} if pset.tools else {}
    )
    prompt = registry.get(PROMPT_ID[CHAT], "^1.0.0", model_metadata=metadata, **kwargs)

    if pset.history:
        messages: list[Any] = [
            HumanMessage("hi"),
            AIMessage(
                "",
                tool_calls=[{"name": "get_weather", "args": {"city": "x"}, "id": "c1"}],
            ),
            ToolMessage("sunny", tool_call_id="c1"),
        ]
        return await prompt.bound.last.bind(**pset.bind).ainvoke(messages)

    if pset.bind:
        return await prompt.bound.last.bind(**pset.bind).ainvoke([HumanMessage("hi")])

    return await prompt.ainvoke({})
