"""Typed executor image results: validation and conversion to content blocks.

When ``read_file`` targets an image, the client executor returns it as the
``ActionResponse.imageResponse`` oneof variant (``mime_type`` + encoded file
``bytes``), and ``executor/action.py`` surfaces that as an
:class:`~duo_workflow_service.executor.image_result.ImageActionResult`. This module
is the consumer: it validates the payload and converts it into standard
langchain content blocks (a short text lead-in plus an image block), so the
model receives pixels rather than nothing.

Validation notes:

* The format policy is shared with user attachments: the allowlist is
    ``ALLOWED_IMAGE_MIME_TYPES`` and the payload must match its declared type by
    magic bytes (``payload_matches_mime_type``), both from ``attachments.py``.
* An image result with invalid contents (unsupported mime type, empty or
    mismatched bytes) converts to a readable error string for the model instead
    of raising — the tool keeps its string/blocks contract either way.
"""

import base64
from typing import Any, Union

from duo_workflow_service.entities.attachments import (
    ALLOWED_IMAGE_MIME_TYPES,
    payload_matches_mime_type,
)
from duo_workflow_service.entities.image_blocks import image_content_block
from duo_workflow_service.executor.image_result import ImageActionResult

# One image-format policy for both entry points: this set is the attachment
# path's provider-intersection allowlist (see ``attachments.py`` for why GIF
# and HEIC are out), and the client executor mirrors it when deciding whether
# to emit an image response.
SUPPORTED_IMAGE_MIME_TYPES = ALLOWED_IMAGE_MIME_TYPES

# Human-readable names for the supported formats, in the order the tool
# descriptions present them. Keyed by the same mime types as the allowlist and
# asserted to cover it (test_supported_formats_display_covers_the_allowlist),
# so adding a format without naming it fails loudly instead of silently
# leaving the tool descriptions stale — the model only attempts image reads
# the description advertises.
_FORMAT_DISPLAY_NAMES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WebP",
}


def supported_image_formats_display() -> str:
    """The advertised format list (e.g. "PNG, JPEG, WebP"), derived from the allowlist."""
    return ", ".join(
        name
        for mime_type, name in _FORMAT_DISPLAY_NAMES.items()
        if mime_type in SUPPORTED_IMAGE_MIME_TYPES
    )


# Decoded-size ceiling for a single image. A tool result has to fit the 4 MiB
# gRPC message budget, and 2 MiB decoded is ~2.7 MiB once base64 inflates it
# on the protojson leg, which leaves room for the surrounding framing.
#
# Checkpoints are a second budget this ceiling does NOT cover: an unstripped
# image costs ~1.37x its raw bytes in every checkpoint from the tool result
# onward (measured), so persistence stays within budget only while the
# checkpoint encoder strips image payloads before serializing. Revisit this
# number together with downscaling.
MAX_IMAGE_DECODED_BYTES = 2 * 1024 * 1024


def _invalid_image_response(reason: str) -> str:
    return (
        f"read_file returned an invalid image response: {reason}. "
        "The file may be corrupted, or the client executor is out of date."
    )


def image_response_to_blocks(
    image: ImageActionResult, *, file_path: str | None = None
) -> Union[str, list[dict[str, Any]]]:
    """Convert a typed executor image result into content blocks.

    Args:
        image: The typed result surfaced from ``ActionResponse.imageResponse``.
        file_path: The path the tool read, included in the text lead-in.

    Returns:
        ``[text block, image block]`` for a valid image, where the text block
        anchors the image in history (path, mime type, size). An error string
        the model can act on for an invalid one (unsupported or mismatched
        type, empty payload, over the size cap).
    """
    if image.mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        return _invalid_image_response(f"unsupported mime type {image.mime_type!r}")

    if not image.data:
        return _invalid_image_response("missing image data")

    size = len(image.data)
    if size > MAX_IMAGE_DECODED_BYTES:
        which = f" {file_path}" if file_path else ""
        return (
            f"Image{which} is too large to attach: "
            f"{size / (1024 * 1024):.1f} MiB "
            f"(limit {MAX_IMAGE_DECODED_BYTES // (1024 * 1024)} MiB). "
            "Use a smaller or downscaled copy of the file."
        )

    # Same magic-byte sanity check the attachment path applies: the declared
    # mime type reaches the provider verbatim, so bytes that do not match it
    # would fail there with an opaque error instead of a readable one here.
    if not payload_matches_mime_type(image.data, image.mime_type):
        return _invalid_image_response(
            f"content does not match the declared type {image.mime_type!r}"
        )

    location = f": {file_path}" if file_path else ""
    lead_in = (
        f"Read image file{location} ({image.mime_type}, {size // 1024} KB). "
        "The image follows."
    )
    return [
        {"type": "text", "text": lead_in},
        image_content_block(
            base64=base64.b64encode(image.data).decode("ascii"),
            mime_type=image.mime_type,
        ),
    ]
