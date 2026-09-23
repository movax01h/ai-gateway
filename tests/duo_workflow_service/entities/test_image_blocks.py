import base64
import json

import pytest
from langchain_core.messages import ToolMessage

from duo_workflow_service.entities.image_blocks import (
    IMAGE_BLOCK_TOKEN_ESTIMATE,
    block_text,
    content_as_text,
    image_content_block,
    is_image_block,
    is_image_content_block,
    strip_image_payloads,
    with_block_text,
)

PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\npixels").decode()


class TestImageContentBlock:
    def test_builds_a_langchain_standard_block(self):
        block = image_content_block(base64=PNG_B64, mime_type="image/png")

        assert block["type"] == "image"
        assert block["base64"] == PNG_B64
        assert block["mime_type"] == "image/png"

    def test_the_block_it_builds_is_recognised_as_one(self):
        """The constructor and the predicate must not drift apart."""
        assert is_image_content_block(
            image_content_block(base64=PNG_B64, mime_type="image/webp")
        )


class TestIsImageContentBlock:
    @pytest.mark.parametrize(
        "block,expected",
        [
            ({"type": "image", "base64": PNG_B64}, True),
            ({"type": "image", "url": "https://example.com/a.png"}, False),
            ({"type": "image", "base64": ""}, False),
            ({"type": "text", "text": "hi"}, False),
            ("plain string", False),
            (None, False),
        ],
    )
    def test_only_inline_payloads_match(self, block, expected):
        assert is_image_content_block(block) is expected


class TestIsImageBlock:
    """The two standard shapes and nothing else.

    ``is_image_content_block`` is the inline subset of this, so every block that carries a payload is also an image
    block, and a ``type: image`` dict with its payload under any other key is neither.
    """

    @pytest.mark.parametrize(
        "block,expected",
        [
            ({"type": "image", "base64": PNG_B64}, True),
            ({"type": "image", "url": "https://example.com/a.png"}, True),
            ({"type": "image", "base64": ""}, False),
            ({"type": "image", "url": None}, False),
            ({"type": "image", "url": ""}, False),
            ({"type": "image", "source": {"type": "base64", "data": PNG_B64}}, False),
            ({"type": "image"}, False),
            ({"type": "text", "text": "hi"}, False),
            ("plain string", False),
            (None, False),
        ],
    )
    def test_inline_or_url_and_nothing_else(self, block, expected):
        assert is_image_block(block) is expected

    @pytest.mark.parametrize(
        "block",
        [
            {"type": "image", "base64": PNG_B64},
            {"type": "image", "url": "https://example.com/a.png"},
            {"type": "image", "source": {"type": "base64", "data": PNG_B64}},
            {"type": "text", "text": "hi"},
        ],
    )
    def test_inline_blocks_are_a_subset_of_image_blocks(self, block):
        if is_image_content_block(block):
            assert is_image_block(block)


class TestStripImagePayloads:
    @pytest.mark.parametrize(
        "content",
        [
            "a plain string",
            [{"type": "text", "text": "no images here"}],
            [{"type": "image", "url": "https://example.com/a.png"}],
            [],
            None,
        ],
    )
    def test_content_without_inline_images_is_returned_unchanged(self, content):
        assert strip_image_payloads(content) is content

    def test_image_blocks_become_text_placeholders(self):
        content = [
            {"type": "text", "text": "look"},
            {"type": "image", "base64": PNG_B64, "mime_type": "image/png"},
        ]

        stripped = strip_image_payloads(content)

        assert stripped[0] == {"type": "text", "text": "look"}
        assert stripped[1] == {
            "type": "text",
            "text": "[image/png omitted from history]",
        }
        assert PNG_B64 not in json.dumps(stripped)

    def test_placeholder_falls_back_when_the_mime_type_is_missing(self):
        (block,) = strip_image_payloads([{"type": "image", "base64": PNG_B64}])

        assert block["text"] == "[image omitted from history]"

    @pytest.mark.parametrize(
        "word", ["attachment", "read_file", "upload", "tool", "user"]
    )
    def test_placeholder_names_no_producer(self, word):
        """A block carries no provenance, so wording that fits one producer misleads the model about every other one."""
        (block,) = strip_image_payloads(
            [{"type": "image", "base64": PNG_B64, "mime_type": "image/png"}]
        )

        assert word not in block["text"]

    def test_a_sibling_text_block_survives_so_producers_can_leave_a_recovery_hint(self):
        """A tool that can re-fetch the image says so in its own text block; that block is not an image block, so
        stripping leaves it intact."""
        content = [
            {"type": "text", "text": "Contents of diagram.png:"},
            {"type": "image", "base64": PNG_B64, "mime_type": "image/png"},
        ]

        stripped = strip_image_payloads(content)

        assert stripped[0] == {"type": "text", "text": "Contents of diagram.png:"}

    def test_original_content_is_not_mutated(self):
        content = [{"type": "image", "base64": PNG_B64, "mime_type": "image/png"}]

        strip_image_payloads(content)

        assert content[0]["base64"] == PNG_B64


def test_image_token_estimate_is_a_sane_constant():
    """Guards against a regression to tiktoken-encoding base64 as prose."""
    assert 0 < IMAGE_BLOCK_TOKEN_ESTIMATE < 10_000


class TestContentAsText:
    """One string out, whatever came in.

    The ui_chat_log tool card is the hard constraint: the CLI and the IDE validate `tool_response` as a string (or a
    message whose `content` is one) and discard the whole checkpoint's chat log otherwise. Base64 must not ride it,
    and nothing may leave a block list without a trace.
    """

    IMAGE_BLOCK = {
        "type": "image",
        "base64": "A" * 4096,
        "mime_type": "image/png",
    }

    def test_text_and_image_collapse_without_the_payload(self):
        content = [
            {"type": "text", "text": "Read image file: ./a.png (image/png, 3 KB)."},
            self.IMAGE_BLOCK,
        ]

        result = content_as_text(content)

        assert result == (
            "Read image file: ./a.png (image/png, 3 KB).\n"
            "[image/png omitted from history]"
        )
        assert "AAAA" not in result

    def test_image_placeholder_is_the_one_history_uses(self):
        # One voice for a lost image: the card, the summarizer and a resumed
        # session all say the same thing, because they all go through
        # strip_image_payloads.
        (placeholder,) = strip_image_payloads([self.IMAGE_BLOCK])

        assert content_as_text([self.IMAGE_BLOCK]) == placeholder["text"]

    def test_text_only_list_flattens(self):
        # The case the image gate used to miss: no image, still a list, still
        # rejected by the client. Thomas's reproduction on !6863.
        assert content_as_text([{"type": "text", "text": "x"}]) == "x"
        assert content_as_text([{"type": "text", "text": "a"}, "b"]) == "a\nb"

    def test_unrenderable_blocks_leave_a_marker(self):
        # A block type this helper does not know about must still be visible;
        # silently dropping it loses that the tool returned it. This is the
        # property BaseMessage.text does not have.
        content = [
            self.IMAGE_BLOCK,
            {"type": "audio", "base64": "B" * 32},
            {"no_type_key": True},
        ]

        result = content_as_text(content)

        assert result == (
            "[image/png omitted from history]\n[audio omitted]\n[dict omitted]"
        )
        assert "BBBB" not in result

    def test_a_block_naming_a_source_renders_title_and_url(self):
        """Web search results are the live case: they carry `url` and usually `title`, and no text.

        Only those two fields are shown. The snippet and the encrypted payload a provider adds stay off the card.
        """
        content = [
            {
                "type": "web_search_result",
                "url": "https://a.example/x",
                "title": "A doc",
                "encrypted_content": "Z" * 32,
                "page_age": "2026",
            },
            {"type": "web_search_result", "url": "https://b.example/y", "title": ""},
            {"type": "web_search_result", "snippet": "no url here"},
        ]

        result = content_as_text(content)

        assert result == (
            "A doc: https://a.example/x\nhttps://b.example/y\n[web_search_result omitted]"
        )
        assert "ZZZZ" not in result
        assert "no url here" not in result

    def test_dict_renders_as_a_single_block(self):
        assert content_as_text({"type": "text", "text": "hello"}) == "hello"
        assert content_as_text({"key": "value"}) == "[dict omitted]"
        assert content_as_text(self.IMAGE_BLOCK) == "[image/png omitted from history]"

    def test_string_passes_through_by_identity(self):
        content = "a plain string"

        assert content_as_text(content) is content

    def test_object_carrying_content_renders_by_it(self):
        """Not just messages: the redactor accepts any object with `content`, so this must too.

        Falling through to `str()` would put the object's repr, payload included, on the card.
        """

        # A stand-in for the duck-typed objects the redactor accepts.
        class Wrapper:
            def __init__(self, content):
                self.content = content

        result = content_as_text(
            Wrapper([{"type": "text", "text": "hi"}, self.IMAGE_BLOCK])
        )

        assert result == "hi\n[image/png omitted from history]"
        assert "AAAA" not in result

    def test_tuple_content_is_treated_as_a_block_list(self):
        assert content_as_text(({"type": "text", "text": "a"}, "b")) == "a\nb"

    def test_whole_message_renders_by_its_content(self):
        # Passing the message instead of message.content must not fall through
        # to str(message), which would put the payload's repr on the card.
        message = ToolMessage(
            content=[{"type": "text", "text": "Read"}, self.IMAGE_BLOCK],
            tool_call_id="c1",
        )

        result = content_as_text(message)

        assert result == "Read\n[image/png omitted from history]"
        assert "AAAA" not in result

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (None, ""),
            (42, "42"),
            (True, "True"),
        ],
    )
    def test_scalars_become_strings(self, content, expected):
        assert content_as_text(content) == expected


class TestBlockText:
    """The one reader of the block shape, shared by the card renderer and truncation."""

    def test_string_is_its_own_text(self):
        assert block_text("plain") == "plain"

    def test_text_block_yields_its_text(self):
        assert block_text({"type": "text", "text": "hello"}) == "hello"

    @pytest.mark.parametrize(
        "block",
        [
            {"type": "image", "base64": "QUFB", "mime_type": "image/png"},
            {"type": "audio", "data": "QUFB"},
            {"type": "text", "text": 42},
            {"no_type": True},
            42,
            None,
        ],
    )
    def test_blocks_without_text_yield_none(self, block):
        assert block_text(block) is None

    def test_with_block_text_keeps_the_shape(self):
        assert with_block_text("old", "new") == "new"

        block = {"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}}
        replaced = with_block_text(block, "new")

        assert replaced == {
            "type": "text",
            "text": "new",
            "cache_control": {"type": "ephemeral"},
        }
        assert block["text"] == "old"
