"""Tests for the typed executor image-result consumer."""

import base64

import pytest

from duo_workflow_service.entities.attachments import ALLOWED_IMAGE_MIME_TYPES
from duo_workflow_service.entities.image_blocks import is_image_content_block
from duo_workflow_service.entities.image_response import (
    _FORMAT_DISPLAY_NAMES,
    MAX_IMAGE_DECODED_BYTES,
    SUPPORTED_IMAGE_MIME_TYPES,
    image_response_to_blocks,
    supported_image_formats_display,
)
from duo_workflow_service.executor.image_result import ImageActionResult

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FAKE_PNG_BYTES = PNG_MAGIC + b"not real pixels" * 4

# Real container headers per supported type, because conversion checks the
# payload against its declared mime type (same recognisers as attachments.py).
REAL_HEADERS = {
    "image/png": PNG_MAGIC,
    "image/jpeg": b"\xff\xd8\xff\xe0",
    "image/webp": b"RIFF\x10\x00\x00\x00WEBP",
}


def png_payload(total_bytes: int) -> bytes:
    """A PNG-signed payload padded with zeros to exactly *total_bytes*."""
    return PNG_MAGIC + b"\x00" * (total_bytes - len(PNG_MAGIC))


class TestValidImage:
    def test_converts_to_text_and_image_blocks(self):
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/png", data=FAKE_PNG_BYTES),
            file_path="./screenshot.png",
        )

        assert isinstance(result, list)
        assert len(result) == 2

        text_block, image_block = result
        assert text_block["type"] == "text"
        assert "./screenshot.png" in text_block["text"]
        assert "image/png" in text_block["text"]

        assert is_image_content_block(image_block)
        assert image_block["base64"] == base64.b64encode(FAKE_PNG_BYTES).decode()
        assert image_block["mime_type"] == "image/png"

    def test_text_block_reports_size(self):
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/png", data=png_payload(300 * 1024))
        )

        assert isinstance(result, list)
        assert "300 KB" in result[0]["text"]

    def test_without_file_path(self):
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/png", data=FAKE_PNG_BYTES)
        )

        assert isinstance(result, list)
        assert result[0]["text"].startswith("Read image file (")

    @pytest.mark.parametrize("mime_type", sorted(SUPPORTED_IMAGE_MIME_TYPES))
    def test_all_supported_mime_types(self, mime_type):
        data = REAL_HEADERS[mime_type] + b"not real pixels"
        result = image_response_to_blocks(
            ImageActionResult(mime_type=mime_type, data=data)
        )

        assert isinstance(result, list)
        assert result[1]["mime_type"] == mime_type

    def test_supported_formats_display_covers_the_allowlist(self):
        # The display map feeds the tool descriptions; a format the allowlist
        # carries but the map cannot name would silently never be advertised.
        assert set(_FORMAT_DISPLAY_NAMES) == set(SUPPORTED_IMAGE_MIME_TYPES)
        assert supported_image_formats_display() == "PNG, JPEG, WebP"

    def test_supported_set_is_the_shared_attachment_policy(self):
        # One format policy for both image entry points; GIF and HEIC are out
        # for the provider-intersection reasons documented in attachments.py.
        assert SUPPORTED_IMAGE_MIME_TYPES is ALLOWED_IMAGE_MIME_TYPES
        assert "image/gif" not in SUPPORTED_IMAGE_MIME_TYPES


class TestInvalidImage:
    """An image result with bad contents becomes a readable error string."""

    @pytest.mark.parametrize("mime_type", ["image/bmp", "image/gif", "image/heic"])
    def test_unsupported_mime_type(self, mime_type):
        result = image_response_to_blocks(
            ImageActionResult(mime_type=mime_type, data=b"GIF89a")
        )

        assert isinstance(result, str)
        assert "unsupported mime type" in result
        assert mime_type in result

    def test_empty_data(self):
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/png", data=b"")
        )

        assert isinstance(result, str)
        assert "missing image data" in result

    def test_content_not_matching_declared_type(self):
        # PNG bytes declared as JPEG: the declared type reaches the provider
        # verbatim, so the mismatch has to be caught here, readably.
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/jpeg", data=FAKE_PNG_BYTES)
        )

        assert isinstance(result, str)
        assert "does not match the declared type" in result
        assert "image/jpeg" in result


class TestSizeCap:
    """Oversized images are rejected at conversion so they never reach the 4 MiB gRPC egress guard (which cancels the
    whole session)."""

    def test_over_cap_returns_readable_error(self):
        result = image_response_to_blocks(
            ImageActionResult(
                mime_type="image/png", data=png_payload(MAX_IMAGE_DECODED_BYTES + 1)
            ),
            file_path="./huge.png",
        )

        assert isinstance(result, str)
        assert "too large" in result
        assert "./huge.png" in result
        assert "2 MiB" in result

    def test_exactly_at_cap_still_converts(self):
        data = png_payload(MAX_IMAGE_DECODED_BYTES)
        result = image_response_to_blocks(
            ImageActionResult(mime_type="image/png", data=data)
        )

        assert isinstance(result, list)
        assert result[1]["base64"] == base64.b64encode(data).decode()
