from langchain_core.messages import HumanMessage

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter


def test_messages_with_string_content():
    messages = [
        HumanMessage(content="This is a single message"),
        HumanMessage(content="This is another single message"),
    ]

    result = TikTokenCounter("some_name").count_tokens(messages)

    assert result == 12


def test_messages_with_string_content_and_tools():
    messages = [
        HumanMessage(content="This is a single message"),
        HumanMessage(content="This is another single message"),
    ]

    result = TikTokenCounter("context_builder").count_tokens(messages)

    # context_builder has 4735 tool tokens and these messages have 12
    assert result == 4747


def test_messages_for_chat_agent():
    messages = [
        HumanMessage(content="This is a single message"),
        HumanMessage(content="This is another single message"),
    ]

    result = TikTokenCounter("Chat Agent").count_tokens(messages)

    # Chat Agent has 2500 tool tokens and these messages have 12
    assert result == 2512


def test_messages_with_mixed_content():
    messages = [
        HumanMessage(
            content="This is a single message"
        ),  # 9 content tokens + 2 role tokens = 11 Tokens
        HumanMessage(
            content="This is another single message"
        ),  # 10 content tokens + 2 role tokens = 12 Tokens
        HumanMessage(
            content=[
                {"type": "text", "text": "This is a text message"},  # 10 Tokens
                {"type": "other", "other": "Some value"},  # 5 tokens
                "This is a string message",  # 9 tokens
            ]  # 10 + 5 + 9 content tokens + 2 role tokens = 26 Tokens
        ),
    ]

    result = TikTokenCounter("some_name").count_tokens(messages)

    assert result == 27


def test_unicode_and_emojis():
    """Test that Unicode content is counted accurately."""
    messages = [
        HumanMessage(content="Hello! 你好！こんにちは！🚀✨🎉 Special chars: é à ü ñ"),
    ]
    result = TikTokenCounter("some_name").count_tokens(messages)
    assert result >= 20


def test_json_structured_data():
    """Test that JSON/structured data is counted accurately."""
    messages = [
        HumanMessage(
            content='{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
        ),
    ]
    result = TikTokenCounter("some_name").count_tokens(messages)
    assert result >= 28


def test_large_string_chunking():
    """Test that large strings are handled without stack overflow."""
    large_content = "x" * 500_000  # 500KB
    counter = TikTokenCounter("some_name")
    result = counter.count_string_content(large_content)
    # Should complete without error and return reasonable count
    assert result > 50_000
