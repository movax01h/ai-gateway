import pytest
from langchain_core.runnables import Runnable

from ai_gateway.models.v2.openai import ChatOpenAI


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
