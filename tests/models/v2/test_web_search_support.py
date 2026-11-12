from unittest.mock import MagicMock, patch

import pytest

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.models.v2.web_search_support import (
    bind_web_search,
    supports_native_web_search,
)


def _tool(name: str, locally_available: bool = False) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.locally_available = locally_available
    return tool


def _model(provider: ModelClassProvider, model: str = "", custom_llm_provider=None):
    definition = MagicMock()
    definition.model_class_provider = provider
    definition.params.model = model
    definition.params.custom_llm_provider = custom_llm_provider
    return definition


_BEDROCK = _model(
    ModelClassProvider.LITE_LLM, "bedrock/global.anthropic.claude-sonnet-4-6", "bedrock"
)
_VERTEX = _model(ModelClassProvider.LITE_LLM, "claude-sonnet-4-6", "vertex_ai")


@pytest.mark.parametrize(
    ("model", "locally_available", "bound", "native"),
    [
        (_model(ModelClassProvider.ANTHROPIC), False, ["read_file"], True),
        (_model(ModelClassProvider.OPENAI), False, ["read_file"], True),
        # Conservative default: risk no search rather than a duplicated, billable one.
        (None, False, ["read_file"], True),
        (_BEDROCK, True, ["read_file", "web_search"], False),
        # No native search and no gateway: nothing to bind.
        (_BEDROCK, False, ["read_file"], False),
        # Vertex could search natively but isn't cleared for it, so it gets the fallback.
        (_VERTEX, True, ["read_file", "web_search"], False),
    ],
    ids=[
        "anthropic",
        "openai",
        "no_model",
        "litellm_fallback",
        "litellm_no_gateway",
        "vertex_not_cleared",
    ],
)
def test_web_search_tool_takes_the_models_route(
    model, locally_available, bound, native
):
    tools = [_tool("read_file"), _tool("web_search", locally_available)]

    result, params = bind_web_search(tools, model, {"other": 1})

    assert [tool.name for tool in result] == bound
    assert params == (
        {"other": 1, "web_search_options": {}} if native else {"other": 1}
    )


def test_tools_without_web_search_are_untouched():
    tools = [_tool("read_file")]

    assert bind_web_search(tools, _model(ModelClassProvider.ANTHROPIC), None) == (
        tools,
        None,
    )


@pytest.mark.parametrize(
    ("model", "kept"),
    [
        (_model(ModelClassProvider.ANTHROPIC), True),
        (None, True),
        # A native-only offer (the `enable_web_search` alias) has no fallback.
        (_BEDROCK, False),
    ],
    ids=["native", "no_model", "no_native_search"],
)
def test_native_only_offer_is_dropped_for_models_that_cannot_search(model, kept):
    tools = [_tool("read_file")]

    result, params = bind_web_search(
        tools, model, {"other": 1, "web_search_options": {}}
    )

    assert result == tools
    assert params == ({"other": 1, "web_search_options": {}} if kept else {"other": 1})


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_model(ModelClassProvider.ANTHROPIC, "claude-sonnet-4-6"), True),
        (_model(ModelClassProvider.OPENAI, "gpt-5.2"), True),
        (_model(ModelClassProvider.GOOGLE_GENAI, "gemini-2.5-pro"), False),
        (_model(ModelClassProvider.AMAZON_Q), False),
        # LiteLLM is per model: Vertex runs Anthropic's server-side tools but isn't cleared for
        # them, Bedrock doesn't run them.
        (_VERTEX, False),
        (_BEDROCK, False),
    ],
    ids=[
        "anthropic",
        "openai",
        "google-genai",
        "amazon-q",
        "litellm-vertex",
        "litellm-bedrock",
    ],
)
def test_model_determines_native_support(model, expected):
    assert supports_native_web_search(model) is expected


def test_vertex_is_native_once_cleared():
    with patch(
        "ai_gateway.models.v2.chat_litellm.NATIVE_WEB_SEARCH_ALLOWED_LITELLM_PROVIDERS",
        frozenset({"vertex_ai"}),
    ):
        assert supports_native_web_search(_VERTEX) is True


def test_capable_provider_not_cleared_is_not_native():
    with patch(
        "ai_gateway.models.v2.web_search_support.NATIVE_WEB_SEARCH_ALLOWED_PROVIDERS",
        frozenset(),
    ):
        assert supports_native_web_search(_model(ModelClassProvider.ANTHROPIC)) is False
