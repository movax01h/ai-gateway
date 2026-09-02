import base64
import json

import pytest

from duo_workflow_service.entities.attachments import (
    ALLOWED_IMAGE_MIME_TYPES,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_TOTAL_ATTACHMENT_BYTES,
    Attachment,
    attachment_content_blocks,
    parse_attachments,
    split_attachment_envelopes,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"pixels"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def envelope(**payload) -> AdditionalContext:
    """Build an ``attachments`` AdditionalContext envelope from a payload dict."""
    return AdditionalContext(
        category="attachments",
        content=json.dumps(payload),
        metadata={"version": "1.0.0"},
    )


def valid_envelope(**overrides) -> AdditionalContext:
    payload = {
        "mime_type": "image/png",
        "data": PNG_B64,
        "filename": "screenshot.png",
    }
    payload.update(overrides)
    return envelope(**payload)


class TestParseAttachments:
    def test_empty_input_returns_empty_list(self):
        assert parse_attachments([]) == []

    def test_parses_a_valid_envelope(self):
        (attachment,) = parse_attachments([valid_envelope()])

        assert attachment.mime_type == "image/png"
        assert attachment.data == PNG_B64
        assert attachment.filename == "screenshot.png"
        assert attachment.byte_size == len(PNG_BYTES)

    def test_preserves_request_order(self):
        attachments = parse_attachments(
            [
                valid_envelope(filename="first.png"),
                valid_envelope(filename="second.png"),
            ]
        )

        assert [a.filename for a in attachments] == ["first.png", "second.png"]

    @pytest.mark.parametrize("mime_type", sorted(ALLOWED_IMAGE_MIME_TYPES))
    def test_accepts_every_allowed_media_type(self, mime_type):
        (attachment,) = parse_attachments([valid_envelope(mime_type=mime_type)])

        assert attachment.mime_type == mime_type

    def test_filename_is_optional(self):
        (attachment,) = parse_attachments(
            [envelope(mime_type="image/png", data=PNG_B64)]
        )

        assert attachment.filename is None

    @pytest.mark.parametrize(
        "item,expected",
        [
            (
                AdditionalContext(category="attachments", content=None),
                "attachment #1 is missing 'content'",
            ),
            (
                AdditionalContext(category="attachments", content="not json"),
                "attachment #1 has invalid JSON content",
            ),
            (
                AdditionalContext(category="attachments", content="[1, 2]"),
                "attachment #1 content must be a JSON object",
            ),
            (
                envelope(data=PNG_B64, filename="a.png"),
                "attachment 'a.png' is missing 'mime_type'",
            ),
            (
                envelope(mime_type="image/png", filename="a.png"),
                "attachment 'a.png' is missing 'data'",
            ),
            (
                # Falls back to the positional label when no filename is given.
                envelope(data=PNG_B64),
                "attachment '#1' is missing 'mime_type'",
            ),
        ],
    )
    def test_rejects_malformed_envelopes(self, item, expected):
        with pytest.raises(ValueError, match=expected):
            parse_attachments([item])

    def test_rejects_unsupported_media_type(self):
        with pytest.raises(
            ValueError, match="unsupported media type 'application/pdf'"
        ):
            parse_attachments([valid_envelope(mime_type="application/pdf")])

    def test_rejects_non_base64_data(self):
        with pytest.raises(ValueError, match="is not valid base64 data"):
            parse_attachments([valid_envelope(data="not base64!!")])

    def test_rejects_attachment_over_per_file_limit(self):
        oversized = base64.b64encode(b"x" * (MAX_ATTACHMENT_BYTES + 1)).decode()

        with pytest.raises(ValueError, match="per-attachment limit"):
            parse_attachments([valid_envelope(data=oversized)])

    def test_rejects_too_many_attachments(self):
        items = [valid_envelope() for _ in range(MAX_ATTACHMENTS + 1)]

        with pytest.raises(ValueError, match="exceeds the limit of"):
            parse_attachments(items)

    def test_rejects_total_over_combined_limit(self):
        # Two attachments each under the per-file cap but over the total cap.
        half = base64.b64encode(b"x" * (MAX_TOTAL_ATTACHMENT_BYTES // 2 + 1)).decode()

        with pytest.raises(ValueError, match="exceeds the .* byte limit"):
            parse_attachments([valid_envelope(data=half), valid_envelope(data=half)])


class TestSplitAttachmentEnvelopes:
    @pytest.mark.parametrize("items", [None, []])
    def test_empty_input_yields_two_empty_lists(self, items):
        assert split_attachment_envelopes(items) == ([], [])

    def test_context_without_attachments_passes_straight_through(self):
        context = [AdditionalContext(category="file", content="print(1)")]

        remaining, attachments = split_attachment_envelopes(context)

        assert remaining == context
        assert attachments == []

    def test_attachment_envelopes_are_claimed_out_of_the_context(self):
        file_context = AdditionalContext(category="file", content="print(1)")

        remaining, attachments = split_attachment_envelopes(
            [file_context, valid_envelope()]
        )

        assert remaining == [file_context]
        assert [a.filename for a in attachments] == ["screenshot.png"]

    def test_remaining_context_keeps_its_original_order(self):
        first = AdditionalContext(category="file", id="1", content="a")
        second = AdditionalContext(category="issue", id="2", content="b")

        remaining, _ = split_attachment_envelopes([first, valid_envelope(), second])

        assert remaining == [first, second]

    def test_invalid_attachments_reject_the_whole_turn(self):
        with pytest.raises(ValueError, match="unsupported media type"):
            split_attachment_envelopes([valid_envelope(mime_type="application/pdf")])


class TestContentBlocks:
    def test_builds_standard_image_blocks(self):
        (block,) = attachment_content_blocks(
            [Attachment(mime_type="image/png", data=PNG_B64)]
        )

        assert block["type"] == "image"
        assert block["base64"] == PNG_B64
        assert block["mime_type"] == "image/png"

    def test_one_block_per_attachment_in_order(self):
        blocks = attachment_content_blocks(
            [
                Attachment(mime_type="image/png", data=PNG_B64),
                Attachment(mime_type="image/webp", data=PNG_B64),
            ]
        )

        assert [block["mime_type"] for block in blocks] == ["image/png", "image/webp"]

    def test_no_attachments_produce_no_blocks(self):
        assert attachment_content_blocks([]) == []
