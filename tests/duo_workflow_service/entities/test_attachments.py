import base64
import json

import pytest

from duo_workflow_service.entities.attachments import (
    ALLOWED_IMAGE_MIME_TYPES,
    ATTACHMENTS_CATEGORY,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_TOTAL_ATTACHMENT_BYTES,
    Attachment,
    attachment_content_blocks,
    attachment_reference_envelopes,
    parse_attachments,
    partition_attachment_envelopes,
    split_attachment_envelopes,
    with_attachment_references,
)
from duo_workflow_service.entities.image_blocks import strip_image_payloads
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


class TestPartitionAttachmentEnvelopes:
    """The half of the split that cannot fail, so a caller can drop attachments it has no message to carry without a
    malformed one taking the turn down."""

    def test_separates_envelopes_from_the_rest_in_order(self):
        other = AdditionalContext(category="file", content="ctx")
        envelope_ = valid_envelope()

        remaining, envelopes = partition_attachment_envelopes([other, envelope_, other])

        assert remaining == [other, other]
        assert envelopes == [envelope_]

    def test_does_not_validate_what_it_removes(self):
        """`split_attachment_envelopes` would raise on each of these."""
        bad = [
            valid_envelope(mime_type="application/pdf"),
            valid_envelope(data="not base64!!"),
            AdditionalContext(category="attachments", content="not json"),
        ]

        remaining, envelopes = partition_attachment_envelopes(bad)

        assert remaining == []
        assert len(envelopes) == 3

    @pytest.mark.parametrize("items", [None, []])
    def test_empty_input(self, items):
        assert partition_attachment_envelopes(items) == ([], [])


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

    def test_a_named_attachment_is_labelled_before_its_image(self):
        blocks = attachment_content_blocks(
            [Attachment(mime_type="image/png", data=PNG_B64, filename="diagram.png")]
        )

        assert blocks[0] == {"type": "text", "text": "[attached file: diagram.png]"}
        assert blocks[1]["type"] == "image"

    def test_the_label_survives_payload_stripping(self):
        """The whole point: after a checkpoint the image is gone, but the model can still
        say which file it was shown."""
        blocks = attachment_content_blocks(
            [Attachment(mime_type="image/png", data=PNG_B64, filename="diagram.png")]
        )

        stripped = strip_image_payloads(blocks)

        assert {"type": "text", "text": "[attached file: diagram.png]"} in stripped
        assert not [
            block
            for block in stripped
            if isinstance(block, dict) and block.get("base64")
        ]

    def test_each_attachment_gets_its_own_label(self):
        blocks = attachment_content_blocks(
            [
                Attachment(mime_type="image/png", data=PNG_B64, filename="a.png"),
                Attachment(mime_type="image/webp", data=PNG_B64, filename="b.webp"),
            ]
        )

        assert [block.get("text") for block in blocks if block["type"] == "text"] == [
            "[attached file: a.png]",
            "[attached file: b.webp]",
        ]

    def test_an_unnamed_attachment_gets_no_label(self):
        """There is nothing useful to say, and an empty label would just cost tokens."""
        blocks = attachment_content_blocks(
            [Attachment(mime_type="image/png", data=PNG_B64)]
        )

        assert [block["type"] for block in blocks] == ["image"]


class TestAttachmentReferenceEnvelopes:
    def test_names_each_attachment_without_carrying_its_payload(self):
        (envelope_,) = attachment_reference_envelopes(
            [Attachment(mime_type="image/png", data=PNG_B64, filename="screenshot.png")]
        )

        assert envelope_.category == ATTACHMENTS_CATEGORY
        assert envelope_.metadata["title"] == "screenshot.png"
        # The whole point of the reference: the payload must not travel back out.
        assert envelope_.content is None
        assert PNG_B64 not in envelope_.model_dump_json()

    def test_satisfies_the_clients_context_item_contract(self):
        # duo-ui's `contextItemValidator` rejects an item without `id`, `category`,
        # or a `metadata` object whose `enabled` is a real bool.
        (envelope_,) = attachment_reference_envelopes(
            [Attachment(mime_type="image/png", data=PNG_B64, filename="a.png")]
        )

        assert envelope_.id
        assert envelope_.category
        assert isinstance(envelope_.metadata, dict)
        assert envelope_.metadata["enabled"] is True
        assert envelope_.metadata["icon"] == "paperclip"

    def test_ids_are_positional_and_unique_within_a_turn(self):
        envelopes = attachment_reference_envelopes(
            [
                Attachment(mime_type="image/png", data=PNG_B64, filename="a.png"),
                Attachment(mime_type="image/webp", data=PNG_B64, filename="b.webp"),
            ]
        )

        ids = [envelope_.id for envelope_ in envelopes]
        assert ids == ["attachment-1", "attachment-2"]
        # The client interpolates the id into a DOM id, so a filename (spaces, dots)
        # is deliberately not used.
        assert len(set(ids)) == len(ids)

    def test_preserves_order_and_reports_each_mime_type(self):
        envelopes = attachment_reference_envelopes(
            [
                Attachment(mime_type="image/png", data=PNG_B64, filename="a.png"),
                Attachment(mime_type="image/webp", data=PNG_B64, filename="b.webp"),
            ]
        )

        assert [e.metadata["title"] for e in envelopes] == ["a.png", "b.webp"]
        assert [e.metadata["secondaryText"] for e in envelopes] == [
            "image/png",
            "image/webp",
        ]

    def test_unnamed_attachment_falls_back_to_a_positional_title(self):
        # `filename` is optional on the wire, and a token with a blank label would
        # render as an empty chip.
        envelopes = attachment_reference_envelopes(
            [
                Attachment(mime_type="image/png", data=PNG_B64),
                Attachment(mime_type="image/png", data=PNG_B64, filename=""),
            ]
        )

        assert [e.metadata["title"] for e in envelopes] == ["image 1", "image 2"]

    def test_no_attachments_produce_no_envelopes(self):
        assert attachment_reference_envelopes([]) == []


class TestWithAttachmentReferences:
    def test_appends_references_after_the_turns_own_context(self):
        context = [AdditionalContext(category="file", content="def foo(): ...")]

        result = with_attachment_references(
            context,
            [Attachment(mime_type="image/png", data=PNG_B64, filename="a.png")],
        )

        assert [item.category for item in result] == ["file", ATTACHMENTS_CATEGORY]

    def test_does_not_mutate_the_context_handed_to_the_prompt_path(self):
        # The same list is passed to `assemble_user_message`; appending in place
        # would render the references into the prompt too.
        context = [AdditionalContext(category="file", content="ctx")]

        with_attachment_references(
            context, [Attachment(mime_type="image/png", data=PNG_B64)]
        )

        assert len(context) == 1

    def test_attachments_only_turn_yields_references_alone(self):
        result = with_attachment_references(
            None, [Attachment(mime_type="image/png", data=PNG_B64, filename="a.png")]
        )

        assert [item.category for item in result] == [ATTACHMENTS_CATEGORY]

    @pytest.mark.parametrize(
        "context", [None, [], [AdditionalContext(category="file")]]
    )
    def test_context_is_unchanged_when_there_are_no_attachments(self, context):
        assert with_attachment_references(context, []) == context
