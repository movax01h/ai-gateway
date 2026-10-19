import base64

from langchain_core.messages import HumanMessage

from duo_workflow_service.entities.image_blocks import IMAGE_BLOCK_TOKEN_ESTIMATE
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


class TestImageContentBlocks:
    """Image payloads must be estimated, never tiktoken-encoded as prose."""

    def test_image_block_uses_the_flat_estimate(self):
        counter = TikTokenCounter(agent_name="unknown_agent")
        # A base64 blob large enough that encoding it as text would dwarf the estimate.
        payload = base64.b64encode(b"x" * 200_000).decode()

        tokens = counter.count_tokens_in_list(
            [{"type": "image", "base64": payload, "mime_type": "image/png"}]
        )

        assert tokens == IMAGE_BLOCK_TOKEN_ESTIMATE

    def test_text_blocks_alongside_an_image_are_still_counted(self):
        counter = TikTokenCounter(agent_name="unknown_agent")
        payload = base64.b64encode(b"x" * 1000).decode()

        tokens = counter.count_tokens_in_list(
            [
                {"type": "text", "text": "describe this screenshot please"},
                {"type": "image", "base64": payload, "mime_type": "image/png"},
            ]
        )

        assert tokens > IMAGE_BLOCK_TOKEN_ESTIMATE

    def test_multimodal_human_message_is_bounded(self):
        counter = TikTokenCounter(agent_name="unknown_agent")
        payload = base64.b64encode(b"x" * 500_000).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": "hi"},
                {"type": "image", "base64": payload, "mime_type": "image/png"},
            ]
        )

        tokens = counter.count_tokens([message], include_tool_tokens=False)

        assert tokens < 2 * IMAGE_BLOCK_TOKEN_ESTIMATE

    def test_url_only_image_blocks_are_not_treated_as_payloads(self):
        counter = TikTokenCounter(agent_name="unknown_agent")

        tokens = counter.count_tokens_in_list(
            [{"type": "image", "url": "https://example.com/a.png"}]
        )

        assert tokens != IMAGE_BLOCK_TOKEN_ESTIMATE
