import base64
import json

import pytest

from duo_workflow_service.entities.image_blocks import (
    IMAGE_BLOCK_TOKEN_ESTIMATE,
    image_content_block,
    is_image_content_block,
    strip_image_payloads,
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
