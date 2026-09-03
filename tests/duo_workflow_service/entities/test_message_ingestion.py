from langchain_core.messages import HumanMessage

from duo_workflow_service.entities.message_ingestion import (
    assemble_user_message,
    non_blank_text_block,
    split_text_blocks,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext


class TestAssembleUserMessage:
    def test_text_only_matches_legacy_shape(self):
        context = [AdditionalContext(category="file", content="def foo(): ...")]

        message = assemble_user_message("explain this", context)

        assert isinstance(message, HumanMessage)
        assert message.content == "explain this"
        assert message.additional_kwargs == {"additional_context": context}

    def test_text_only_with_no_context_keeps_kwargs_key(self):
        message = assemble_user_message("hello")

        assert message.content == "hello"
        assert message.additional_kwargs == {"additional_context": None}

    def test_content_blocks_follow_the_text_block(self):
        image_block = {"type": "image", "base64": "abc123", "mime_type": "image/png"}
        context = [AdditionalContext(category="file", content="ctx")]

        message = assemble_user_message("what is this?", context, [image_block])

        assert message.content == [
            {"type": "text", "text": "what is this?"},
            image_block,
        ]
        assert message.additional_kwargs == {"additional_context": context}

    def test_content_blocks_without_text_omit_the_text_block(self):
        image_block = {"type": "image", "base64": "abc123", "mime_type": "image/png"}

        message = assemble_user_message("", None, [image_block])

        assert message.content == [image_block]

    def test_content_blocks_with_whitespace_only_text_omit_the_text_block(self):
        image_block = {"type": "image", "base64": "abc123", "mime_type": "image/png"}

        message = assemble_user_message("  \n ", None, [image_block])

        assert message.content == [image_block]

    def test_empty_blocks_list_falls_back_to_string_content(self):
        message = assemble_user_message("hello", None, [])

        assert message.content == "hello"


class TestNonBlankTextBlock:
    def test_wraps_non_blank_text(self):
        assert non_blank_text_block("hello") == [{"type": "text", "text": "hello"}]

    def test_empty_text_yields_no_block(self):
        assert non_blank_text_block("") == []

    def test_whitespace_only_text_yields_no_block(self):
        assert non_blank_text_block(" \n\t ") == []


class TestSplitTextBlocks:
    def test_joins_text_pieces_and_preserves_passthrough_order(self):
        image_a = {"type": "image", "base64": "aaa", "mime_type": "image/png"}
        image_b = {"type": "image", "base64": "bbb", "mime_type": "image/jpeg"}

        text, passthrough = split_text_blocks(
            [
                {"type": "text", "text": "first"},
                image_a,
                "second",
                image_b,
            ]
        )

        assert text == "first\nsecond"
        assert passthrough == [image_a, image_b]

    def test_all_text_content_yields_no_passthrough(self):
        text, passthrough = split_text_blocks([{"type": "text", "text": "only"}])

        assert text == "only"
        assert passthrough == []

    def test_empty_content(self):
        text, passthrough = split_text_blocks([])

        assert text == ""
        assert passthrough == []

    def test_text_block_without_text_key_becomes_empty_string(self):
        text, passthrough = split_text_blocks([{"type": "text"}])

        assert text == ""
        assert passthrough == []
