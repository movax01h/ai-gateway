"""User-supplied file attachments (currently images) sent alongside a chat turn.

Transport
---------
Attachments arrive as ``AdditionalContext`` envelopes on ``StartWorkflowRequest``
with ``category == "attachments"`` — one envelope per file — whose ``content`` is
a JSON object::

    {"mime_type": "image/png", "data": "<base64>", "filename": "screenshot.png"}

``attachments`` is an engine-level built-in category: it is claimed by the engine
before the ordinary additional-context machinery runs, so it needs no
flow-config declaration and no unit primitive.

Scope
-----
This module owns the *transport*: the category name, the envelope schema, the
caps, and the validation. Everything after parsing lives elsewhere, because an
attachment stops being special once it is a content block. The block shape, the
token estimate and the checkpoint policy belong to
:mod:`duo_workflow_service.entities.image_blocks`, which a tool-produced image
shares; assembling the user turn belongs to
:func:`~duo_workflow_service.entities.message_ingestion.assemble_user_message`.

Attachments deliberately never reach a Jinja prompt template. They ride the
model-facing ``HumanMessage`` as separate content blocks, so providers receive
real image parts rather than base64 text.
"""

import base64
import binascii
import json
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel

from duo_workflow_service.entities.image_blocks import image_content_block
from duo_workflow_service.workflows.type_definitions import AdditionalContext

__all__ = [
    "ALLOWED_IMAGE_MIME_TYPES",
    "ATTACHMENTS_CATEGORY",
    "MAX_ATTACHMENTS",
    "MAX_ATTACHMENT_BYTES",
    "MAX_TOTAL_ATTACHMENT_BYTES",
    "Attachment",
    "attachment_content_blocks",
    "parse_attachments",
    "split_attachment_envelopes",
]

# Engine-level built-in category for the attachments envelope: claimed by the
# engine rather than declared by a flow config.
ATTACHMENTS_CATEGORY = "attachments"

# Formats accepted by every multimodal provider we route to (Anthropic, OpenAI,
# Gemini). Deliberately conservative: an unsupported type is far better rejected
# here with a clear message than by the provider mid-turn.
ALLOWED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)

# Caps on the decoded payload. The whole `ClientEvent` is bounded by the 4 MiB
# gRPC receive limit (`duo_workflow_service.server.MAX_MESSAGE_SIZE`) and base64
# inflates by 4/3, so the total must stay comfortably below 3 MiB decoded to
# leave room for the goal, flow config and the rest of the request.
MAX_ATTACHMENTS = 5
MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 2_500_000


class Attachment(BaseModel):
    """A single validated user-supplied file attachment."""

    mime_type: str
    data: str
    filename: Optional[str] = None
    byte_size: int = 0


def _decoded_size(data: str, filename: str) -> int:
    """Return the decoded byte length of *data*, validating that it is base64.

    Args:
        data: The base64-encoded payload.
        filename: Used only to build a useful error message.

    Returns:
        The size of the decoded payload in bytes.

    Raises:
        ValueError: If *data* is not valid base64.
    """
    try:
        return len(base64.b64decode(data, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            f"attachment '{filename}' is not valid base64 data: {exc}"
        ) from exc


def _parse_one(item: AdditionalContext, index: int) -> Attachment:
    """Parse and validate a single ``attachments`` envelope.

    Args:
        item: The ``AdditionalContext`` envelope to parse.
        index: Position of the envelope in the request, used in error messages
            when the payload carries no filename.

    Returns:
        The validated :class:`Attachment`.

    Raises:
        ValueError: If the envelope is malformed, of an unsupported media type,
            or exceeds the per-attachment size cap.
    """
    if not item.content:
        raise ValueError(f"attachment #{index} is missing 'content'.")

    try:
        payload = json.loads(item.content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"attachment #{index} has invalid JSON content: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"attachment #{index} content must be a JSON object.")

    filename = payload.get("filename") or f"#{index}"
    mime_type = payload.get("mime_type")
    data = payload.get("data")

    if not isinstance(mime_type, str) or not mime_type:
        raise ValueError(f"attachment '{filename}' is missing 'mime_type'.")
    if not isinstance(data, str) or not data:
        raise ValueError(f"attachment '{filename}' is missing 'data'.")

    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError(
            f"attachment '{filename}' has unsupported media type '{mime_type}'. "
            f"Supported types: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}."
        )

    byte_size = _decoded_size(data, filename)
    if byte_size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachment '{filename}' is {byte_size} bytes, which exceeds the "
            f"{MAX_ATTACHMENT_BYTES} byte per-attachment limit."
        )

    return Attachment(
        mime_type=mime_type,
        data=data,
        filename=payload.get("filename"),
        byte_size=byte_size,
    )


def parse_attachments(items: Sequence[AdditionalContext]) -> list[Attachment]:
    """Parse and validate the ``attachments`` envelopes of a single turn.

    Args:
        items: The ``AdditionalContext`` envelopes whose category is
            ``attachments``.

    Returns:
        The validated attachments, in request order. Empty when *items* is empty.

    Raises:
        ValueError: If any envelope is malformed or the count/size caps are
            exceeded. Rejecting the whole turn is deliberate: silently dropping
            an attachment the user explicitly added would make the model answer
            about something it cannot see.
    """
    if not items:
        return []

    if len(items) > MAX_ATTACHMENTS:
        raise ValueError(
            f"{len(items)} attachments were sent, which exceeds the limit of "
            f"{MAX_ATTACHMENTS}."
        )

    attachments = [_parse_one(item, index) for index, item in enumerate(items, start=1)]

    total = sum(attachment.byte_size for attachment in attachments)
    if total > MAX_TOTAL_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachments total {total} bytes, which exceeds the "
            f"{MAX_TOTAL_ATTACHMENT_BYTES} byte limit for a single message."
        )

    return attachments


def split_attachment_envelopes(
    items: Optional[Sequence[AdditionalContext]],
) -> tuple[list[AdditionalContext], list[Attachment]]:
    """Claim the ``attachments`` envelopes out of a turn's additional context.

    Attachments must not travel on with the rest of the additional context: that
    list is rendered verbatim into the prompt by the additional-context partial
    and is echoed back to the client on every ``UiChatLog``, either of which
    would turn a base64 payload into prose.

    Args:
        items: The turn's raw ``AdditionalContext`` list, possibly ``None``.

    Returns:
        The remaining (non-attachment) context in its original order, and the
        parsed attachments.

    Raises:
        ValueError: If an attachment envelope is malformed or the caps are
            exceeded. See :func:`parse_attachments`.
    """
    if not items:
        return [], []

    envelopes = [item for item in items if item.category == ATTACHMENTS_CATEGORY]
    if not envelopes:
        return list(items), []

    remaining = [item for item in items if item.category != ATTACHMENTS_CATEGORY]
    return remaining, parse_attachments(envelopes)


def attachment_content_blocks(
    attachments: Iterable[Attachment],
) -> list[dict[str, Any]]:
    """Build the image content blocks carrying *attachments*.

    One block per attachment, in order, using the shared constructor so that an attached image is indistinguishable from
    an image any other producer emits.
    """
    return [
        image_content_block(
            base64=attachment.data,
            mime_type=attachment.mime_type,
        )
        for attachment in attachments
    ]
