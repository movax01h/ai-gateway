import pytest
from langchain_core.runnables import Runnable

from ai_gateway.models.v2.openai import _WEB_SEARCH_INCLUDES, ChatOpenAI

_DEFAULTS = list(_WEB_SEARCH_INCLUDES)
# Stands in for an include entry nothing to do with web search.
_UNRELATED_INCLUDE = "reasoning.encrypted_content"


@pytest.mark.parametrize(
    ("bind_tools_params", "expected_tools"),
    [
        (
            {"web_search_options": {}},
            [
                {
                    "type": "function",
                    "function": {"name": "get_issue", "parameters": []},
                },
                {"type": "web_search"},
            ],
        ),
        (
            {},
            [{"type": "function", "function": {"name": "get_issue", "parameters": []}}],
        ),
    ],
)
def test_bind_tools_with_web_search_options(bind_tools_params, expected_tools):
    """Test that web search tool is added when web_search_options is in bind_tools_params."""
    chat = ChatOpenAI(model="gpt-4", api_key="test")

    existing_tools = [{"name": "get_issue", "parameters": []}]
    result = chat.bind_tools(
        tools=existing_tools,
        **bind_tools_params,
    )

    assert isinstance(result, Runnable)
    assert result.kwargs["tools"] == expected_tools


@pytest.mark.parametrize(
    ("field_include", "bind_kwargs", "expected_include"),
    [
        pytest.param(None, {}, None, id="no-web-search-no-include"),
        pytest.param(
            None, {"web_search_options": {}}, _DEFAULTS, id="web-search-adds-defaults"
        ),
        pytest.param(
            [_UNRELATED_INCLUDE],
            {"web_search_options": {}},
            [_UNRELATED_INCLUDE, *_DEFAULTS],
            id="field-include-would-otherwise-be-dropped",
        ),
        pytest.param(
            _DEFAULTS,
            {"web_search_options": {}},
            _DEFAULTS,
            id="entries-already-asked-for-are-not-repeated",
        ),
        pytest.param(
            [_UNRELATED_INCLUDE],
            {"web_search_options": {}, "include": ["file_search_call.results"]},
            ["file_search_call.results", *_DEFAULTS],
            id="bound-include-takes-precedence-over-the-field",
        ),
    ],
)
def test_bind_tools_merges_include(field_include, bind_kwargs, expected_include):
    chat = ChatOpenAI(model="gpt-4", api_key="test", include=field_include)

    result = chat.bind_tools(tools=[], **bind_kwargs)

    assert result.kwargs.get("include") == expected_include
