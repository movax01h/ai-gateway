"""User-supplied file attachments (currently images) sent alongside a chat turn.

Transport
---------
Attachments arrive as ``AdditionalContext`` envelopes on ``StartWorkflowRequest``
with ``category == "attachments"`` — one envelope per file — whose ``content`` is
a JSON object::

    {"mime_type": "image/png", "data": "<base64>", "filename": "screenshot.png"}

Legacy chat claims the ``attachments`` category before the ordinary
additional-context machinery runs, so there it needs no flow-config declaration
and no unit primitive.

That is *not* yet true engine-wide. Agent Platform v1 has no attachment support:
:meth:`~duo_workflow_service.agent_platform.v1.flows.base.Flow._process_additional_context`
skips any category a flow config does not declare, with a server-side log and no
user-visible notice, so files sent to a flow silently never reach the model. The
name is also not reserved -- a flow config that declares an input named
``attachments`` would take this envelope through the normal path and render the
raw base64 straight into its Jinja prompt. Reserving the category engine-wide is
tracked with the Agent Platform wiring, not this module.

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

The one thing that does travel back out is a payload-free *reference* envelope
(:func:`attachment_reference_envelopes`), which exists purely so the client can
name what the user attached. It is transport in the same sense as the inbound
envelope, which is why it lives here rather than with the block shape.
"""

import base64
import binascii
import json
from typing import Any, Callable, Iterable, Optional, Sequence

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
    "attachment_reference_envelopes",
    "attachment_rejection_block",
    "parse_attachments",
    "partition_attachment_envelopes",
    "split_attachment_envelopes",
    "with_attachment_references",
]

# Engine-level built-in category for the attachments envelope: claimed by the
# engine rather than declared by a flow config.
ATTACHMENTS_CATEGORY = "attachments"

# The intersection of what every multimodal provider we route to accepts, which
# `models.yml` makes Anthropic, Gemini, Vertex AI, OpenAI and Bedrock.
#
# It has to be the intersection, not the union. Validation runs here, before
# model selection has decided who serves the turn, so a format only some
# providers take would fail at the provider for whichever session happened to
# pick a different one. LiteLLM does not help: it reshapes the block into each
# provider's form but never transcodes the bytes.
#
# Two formats are deliberately excluded despite being widely supported:
#
# - GIF: taken by Anthropic, OpenAI and Bedrock, but absent from Gemini's and
#   Vertex's documented lists.
# - HEIC/HEIF: the mirror image, documented by Gemini and Vertex only. Adding
#   Apple's formats would break the majority of providers, and browsers other
#   than Safari cannot decode them for the upload preview either. Supporting
#   them means transcoding to PNG/JPEG, which belongs with a ref-based upload.
ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

# Caps on the decoded payload. The whole `ClientEvent` is bounded by the 4 MiB
# gRPC receive limit (`duo_workflow_service.server.MAX_MESSAGE_SIZE`) and base64
# inflates by 4/3, so the total must stay comfortably below 3 MiB decoded to
# leave room for the goal, flow config and the rest of the request.
MAX_ATTACHMENTS = 5
MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 2_500_000

# The caps above measure the decoded image, so the filename is not covered by any
# of them: a tiny image with a megabyte-long name passes all three. That name is
# also the part with the longest life, because the `[attached file: ...]` label
# survives payload stripping and stays in the conversation for the rest of the
# thread. 255 is the limit essentially every filesystem imposes, so a longer name
# did not come from a file picker.
MAX_FILENAME_CHARS = 255


class Attachment(BaseModel):
    """A single validated user-supplied file attachment."""

    mime_type: str
    data: str
    filename: Optional[str] = None
    byte_size: int = 0


def _decode(data: str, filename: str) -> bytes:
    """Decode *data*, validating that it is base64.

    Args:
        data: The base64-encoded payload.
        filename: Used only to build a useful error message.

    Returns:
        The decoded payload.

    Raises:
        ValueError: If *data* is not valid base64.
    """
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            f"attachment '{filename}' is not valid base64 data: {exc}"
        ) from exc


# How to recognise each supported format from the start of its bytes. Keyed by the
# same MIME types as `ALLOWED_IMAGE_MIME_TYPES`, and asserted to cover it, so adding
# a format without teaching this how to spot it fails loudly rather than silently
# waving the new type through.
_MAGIC_NUMBERS: dict[str, Callable[[bytes], bool]] = {
    "image/png": lambda payload: payload.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda payload: payload.startswith(b"\xff\xd8\xff"),
    # RIFF container: "RIFF" <4-byte length> "WEBP".
    "image/webp": lambda payload: payload[:4] == b"RIFF" and payload[8:12] == b"WEBP",
}


def _clean_filename(raw: Any, index: int) -> Optional[str]:
    """Return *raw* reduced to something safe to put in a prompt, a log and the transcript.

    The filename is client-supplied and, unlike the payload, is subject to no cap and
    no validation. It reaches three places that all outlive the request: the
    ``[attached file: ...]`` label, which survives payload stripping and so stays in
    the conversation for the rest of the thread; the rejection messages, which are read
    by both the user and the model; and the reference envelopes the client renders.

    Two things are worth removing before it gets there. Unprintable characters (NUL,
    newlines, terminal escapes) corrupt structured logs and let a name span lines,
    which is what turns a label into something that reads like a separate instruction.
    Unbounded length costs prompt tokens and checkpoint space on every later turn, for
    a string no file picker could have produced.

    Truncating rather than rejecting is deliberate: an awkward name is not a reason to
    refuse an otherwise valid image, and the cap is already far beyond any real one.

    This does not make the name trustworthy. It is user text quoted into a prompt, and
    a name like ``ignore previous instructions.png`` still says what it says -- no
    amount of escaping fixes that, and here the user is quoting it to themselves.
    Path traversal is not a concern: the filename is never used as a path.

    Args:
        raw: The ``filename`` field as it arrived, of any type.
        index: Position of the envelope, used in the error message.

    Returns:
        The cleaned filename, or ``None`` when absent or left empty by cleaning.

    Raises:
        ValueError: If *raw* is present but not a string.
    """
    if raw is None:
        return None

    if not isinstance(raw, str):
        # Pydantic would catch this when `Attachment` is built, but its message is a
        # multi-line dump, and this one is read by the user and relayed by the model.
        raise ValueError(
            f"attachment #{index} has a 'filename' that is not text "
            f"(got {type(raw).__name__})."
        )

    cleaned = "".join(char for char in raw if char.isprintable()).strip()

    if len(cleaned) > MAX_FILENAME_CHARS:
        cleaned = cleaned[: MAX_FILENAME_CHARS - 1] + "…"

    return cleaned or None


def _verify_declared_type(payload: bytes, mime_type: str, filename: str) -> None:
    """Check *payload* actually starts like the format it claims to be.

    ``mime_type`` comes from the client and is otherwise taken on trust, so a file
    mislabelled by a buggy picker (or renamed by hand) would pass every check here and
    fail at the provider instead, with the opaque error the vision gate work describes.
    The payload is already decoded for the size check, so inspecting its first bytes is
    free.

    This is a sanity check on the container, not image validation: it catches a PDF or a
    HEIC sent as ``image/png``, not a truncated or corrupt PNG.

    Raises:
        ValueError: If the payload does not match its declared media type.
    """
    recogniser = _MAGIC_NUMBERS.get(mime_type)
    if recogniser is None or recogniser(payload):
        return

    raise ValueError(
        f"attachment '{filename}' does not look like {mime_type}: its contents "
        "do not match that format. Check the file is not renamed or corrupt."
    )


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

    # Cleaned first: every message below quotes it, and those are read by the user
    # and relayed to the model.
    filename = _clean_filename(payload.get("filename"), index)
    label = filename or f"#{index}"
    mime_type = payload.get("mime_type")
    data = payload.get("data")

    if not isinstance(mime_type, str) or not mime_type:
        raise ValueError(f"attachment '{label}' is missing 'mime_type'.")
    if not isinstance(data, str) or not data:
        raise ValueError(f"attachment '{label}' is missing 'data'.")

    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError(
            f"attachment '{label}' has unsupported media type '{mime_type}'. "
            f"Supported types: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}."
        )

    decoded = _decode(data, label)

    # Size before signature: an oversized file is worth reporting as too big even
    # if it is also mislabelled, because that is the more actionable of the two.
    byte_size = len(decoded)
    if byte_size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"attachment '{label}' is {byte_size} bytes, which exceeds the "
            f"{MAX_ATTACHMENT_BYTES} byte per-attachment limit."
        )

    _verify_declared_type(decoded, mime_type, label)

    return Attachment(
        mime_type=mime_type,
        data=data,
        filename=filename,
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
    remaining, envelopes = partition_attachment_envelopes(items)
    return remaining, parse_attachments(envelopes) if envelopes else []


def partition_attachment_envelopes(
    items: Optional[Sequence[AdditionalContext]],
) -> tuple[list[AdditionalContext], list[AdditionalContext]]:
    """Separate the ``attachments`` envelopes from the rest of a turn's context.

    The half of :func:`split_attachment_envelopes` that cannot fail. Removing the
    envelopes has to happen on *every* turn -- leaving one in the list would render its
    base64 into the prompt -- but validating them is only meaningful on a turn that
    actually builds a message to carry them. Keeping the two apart lets a caller drop
    attachments it cannot use without a malformed one taking the turn down with it.

    Args:
        items: The turn's raw ``AdditionalContext`` list, possibly ``None``.

    Returns:
        The non-attachment context and the attachment envelopes, each in their original
        order. Neither is parsed.
    """
    if not items:
        return [], []

    remaining: list[AdditionalContext] = []
    envelopes: list[AdditionalContext] = []
    for item in items:
        (envelopes if item.category == ATTACHMENTS_CATEGORY else remaining).append(item)
    return remaining, envelopes


def attachment_content_blocks(
    attachments: Iterable[Attachment],
) -> list[dict[str, Any]]:
    """Build the content blocks carrying *attachments*.

    Each named attachment contributes two blocks: a short text block naming the file, then the image itself. The image
    block uses the shared constructor, so an attached image is indistinguishable from an image any other producer emits.

    The label exists for what happens *after* this turn.
    :func:`~duo_workflow_service.entities.image_blocks.strip_image_payloads` drops the payload before the message is
    checkpointed and leaves a deliberately provenance-neutral placeholder, so a resumed session would otherwise read as
    a bare ``[image/png omitted from history]`` and the model could not answer even "which file did I send you?". A text
    block is not an image block, so it survives stripping untouched and keeps the filename in the conversation. It costs
    a handful of tokens and is not a recovery hint -- a user attachment cannot be re-fetched the way a tool-read one
    can.
    """
    blocks: list[dict[str, Any]] = []
    for attachment in attachments:
        if attachment.filename:
            blocks.append(
                {"type": "text", "text": f"[attached file: {attachment.filename}]"}
            )
        blocks.append(
            image_content_block(
                base64=attachment.data,
                mime_type=attachment.mime_type,
            )
        )
    return blocks


def attachment_rejection_block(reason: str) -> dict[str, Any]:
    """Build the content block telling the *model* that this turn's files were rejected.

    Dropping the files without saying so would leave the model answering "what is in this
    screenshot?" as though the user had sent nothing, and it would guess rather than say
    it cannot see the image. The reason is included because it is already phrased for a
    human ("screenshot.png: unsupported media type ..."), and the model relaying it is
    more useful than a bare refusal.
    """
    return {
        "type": "text",
        "text": (
            f"[The user attached one or more files, but they could not be "
            f"included: {reason} The files are not available to you. Tell the "
            f"user what went wrong instead of guessing at their contents.]"
        ),
    }


def attachment_reference_envelopes(
    attachments: Sequence[Attachment],
) -> list[AdditionalContext]:
    """Build payload-free envelopes naming *attachments*, for the client's transcript.

    The web client renders a turn's ``UiChatLog.additional_context`` as "included
    reference" tokens. The parsed attachment is not in that list --
    :func:`split_attachment_envelopes` claims it out -- so without these the user sees
    no trace of what they attached, on this turn or on any reload of the thread.

    ``content`` is empty rather than unset: the payload must not re-enter the checkpoint
    by the back door, and these envelopes never reach a prompt template, so there is
    nothing for the model to read here anyway. It cannot be ``None``, because GitLab
    declares ``AiAdditionalContext.content`` non-nullable and fails the whole transcript
    query when it reads null there.

    ``id`` is positional rather than the filename because the client interpolates it
    into a DOM id, and a filename carries spaces and dots.
    """
    return [
        AdditionalContext(
            category=ATTACHMENTS_CATEGORY,
            id=f"attachment-{index}",
            content="",
            metadata={
                # Keys read by the client's token and popover renderers. `enabled` must
                # be a bool: its context-item validator rejects the item otherwise.
                "title": attachment.filename or f"image {index}",
                "enabled": True,
                "icon": "paperclip",
                "secondaryText": attachment.mime_type,
            },
        )
        for index, attachment in enumerate(attachments, start=1)
    ]


def with_attachment_references(
    context: Optional[Sequence[AdditionalContext]],
    attachments: Sequence[Attachment],
) -> Optional[list[AdditionalContext]]:
    """Append reference envelopes for *attachments* to a turn's client-facing context.

    Only the ``UiChatLog`` copy of the context gets these. The copy handed to
    :func:`~duo_workflow_service.entities.message_ingestion.assemble_user_message` must
    stay as it was, or the references would also be rendered into the prompt.
    """
    if not attachments:
        return None if context is None else list(context)

    return list(context or []) + attachment_reference_envelopes(attachments)
