"""Typed executor image results: validation and conversion to content blocks.

``read_file`` on an image comes back as the ``ActionResponse.imageResponse`` oneof and
reaches here as an :class:`~duo_workflow_service.executor.image_result.ImageActionResult`.
An invalid payload converts to an error string rather than raising, so the tool keeps its
string-or-blocks contract either way. Format policy is shared with user attachments.
"""

import base64
from typing import Any, Union

from duo_workflow_service.entities.attachments import (
    ALLOWED_IMAGE_MIME_TYPES,
    payload_matches_mime_type,
)
from duo_workflow_service.entities.image_blocks import image_content_block
from duo_workflow_service.executor.image_result import ImageActionResult

# A test asserts these names cover ``ALLOWED_IMAGE_MIME_TYPES``: the model only
# attempts the image reads a description advertises, so an unnamed format would
# leave the descriptions stale.
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
        if mime_type in ALLOWED_IMAGE_MIME_TYPES
    )


# 2 MiB of image-file bytes is ~2.7 MiB in protojson, which leaves transport
# headroom, but validation happens after receipt: an executor has to enforce
# the transport limit before sending.
#
# Checkpoints are a budget this does not cover: an unstripped image costs
# ~1.37x its bytes in every later checkpoint (measured), so persistence only
# stays in budget while the encoder strips payloads. Revisit both together
# with downscaling.
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
    if image.mime_type not in ALLOWED_IMAGE_MIME_TYPES:
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

    # The declared mime type reaches the provider verbatim, so mismatched bytes
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
