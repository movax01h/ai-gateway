from langchain_core.messages import HumanMessage

from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)


def test_messages_with_string_content():
    messages = [
        HumanMessage(content="This is a single message"),
        HumanMessage(content="This is another single message"),
    ]

    result = ApproximateTokenCounter("some_name").count_tokens(messages)  # type: ignore

    assert result == 23


def test_messages_with_string_content_and_tools():
    messages = [
        HumanMessage(content="This is a single message"),
        HumanMessage(content="This is another single message"),
    ]

    result = ApproximateTokenCounter("context_builder").count_tokens(messages)  # type: ignore

    # context_builder has 4735 tool tokens and these messages have 23
    assert result == 4758


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

    result = ApproximateTokenCounter("some_name").count_tokens(messages)  # type: ignore

    assert result == 49
