"""User-supplied file attachments (currently images) sent alongside a chat turn.

Transport
---------
Attachments arrive as ``AdditionalContext`` envelopes on ``StartWorkflowRequest``
with ``category == "attachments"`` — one envelope per file — whose ``content`` is
a JSON object::

    {"mime_type": "image/png", "data": "<base64>", "filename": "screenshot.png"}

An envelope may instead reference a file GitLab is holding for the session, which
:func:`resolve_attachment_envelopes` swaps for the inline form above before anything
else reads it::

    {
        "url": "/api/v4/ai/duo_workflows/workflows/1/session_files/2",
        "filename": "screenshot.png"
    }

The URL is a GitLab API path fetched through the executor like any other request, so
it carries the user's own credentials and GitLab authorizes it against the session it
belongs to. ``mime_type`` comes from that response rather than from the client, which
is why the reference form does not carry one.

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
name what the user attached, and fetch it again when it is a session file. It is
transport in the same sense as the inbound envelope, which is why it lives here
rather than with the block shape.
"""

import asyncio
import base64
import binascii
import json
import re
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import structlog
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from duo_workflow_service.entities.image_blocks import (
    block_text,
    image_content_block,
    is_image_placeholder,
)
from duo_workflow_service.entities.message_ingestion import ATTACHMENTS_KWARG
from duo_workflow_service.errors.typing import NotifiableException
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.workflows.type_definitions import AdditionalContext

__all__ = [
    "ALLOWED_IMAGE_MIME_TYPES",
    "ATTACHMENTS_CATEGORY",
    "MAX_ATTACHMENTS",
    "MAX_ATTACHMENT_BYTES",
    "MAX_TOTAL_ATTACHMENT_BYTES",
    "Attachment",
    "AttachmentsNotAllowedError",
    "attachment_content_blocks",
    "attachment_reference_envelopes",
    "attachment_rejection_block",
    "parse_attachments",
    "partition_attachment_envelopes",
    "payload_matches_mime_type",
    "reinline_session_files",
    "resolve_attachment_envelopes",
    "session_file_records",
    "session_files_in",
    "split_attachment_envelopes",
    "with_attachment_references",
]

log = structlog.stdlib.get_logger("attachments")

# The only path a reference may point at. It is client-supplied, so it is matched
# rather than trusted: nothing else in the API can be fetched through this seam and
# rendered to the model as an image.
_SESSION_FILE_PATH = re.compile(
    r"/api/v4/ai/duo_workflows/workflows/(?P<workflow_id>[0-9]+)/session_files/[0-9]+"
)

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
# also the part with the longest life, because the `[attachment N: ...]` label
# survives payload stripping and stays in the conversation for the rest of the
# thread. 255 is the limit essentially every filesystem imposes, so a longer name
# did not come from a file picker.
MAX_FILENAME_CHARS = 255


class AttachmentsNotAllowedError(NotifiableException):
    """Raised when a turn carries attachments this workflow does not allow.

    A ``NotifiableException`` because the message tells the user why the turn was refused and is safe to display
    verbatim.
    """


class Attachment(BaseModel):
    """A single validated user-supplied file attachment."""

    mime_type: str
    data: str
    filename: Optional[str] = None
    byte_size: int = 0
    # The session file path the attachment was fetched from, if it was sent as a reference.
    url: Optional[str] = None


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


def payload_matches_mime_type(payload: bytes, mime_type: str) -> bool:
    """Whether *payload* starts like the format *mime_type* declares.

    A ``mime_type`` with no recogniser returns ``False``, so callers that accept
    only known formats gate on ``ALLOWED_IMAGE_MIME_TYPES`` first.
    """
    recogniser = _MAGIC_NUMBERS.get(mime_type)
    return recogniser is not None and recogniser(payload)


def _clean_filename(raw: Any, index: int) -> Optional[str]:
    """Return *raw* reduced to something safe to put in a prompt, a log and the transcript.

    The filename is client-supplied and, unlike the payload, is subject to no cap and
    no validation. It reaches three places that all outlive the request: the
    ``[attachment N: ...]`` label, which survives payload stripping and so stays in
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
    if mime_type not in _MAGIC_NUMBERS or payload_matches_mime_type(payload, mime_type):
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
    # Only a session file path is kept: it feeds the message record and the chat log reference, not the label.
    url = payload.get("url")
    if not isinstance(url, str) or not _SESSION_FILE_PATH.fullmatch(url):
        url = None

    # Set by resolve_attachment_envelopes when a reference could not be fetched. It
    # travels on the envelope so resolution needs no error channel of its own: the
    # turn-level rejection path already tells both the user and the model.
    error = payload.get("error")
    if error:
        raise ValueError(f"attachment '{label}' could not be read: {error}")

    return _validated_attachment(mime_type, data, label).model_copy(
        update={"filename": filename, "url": url}
    )


def _validated_attachment(mime_type: Any, data: Any, label: str) -> Attachment:
    """Validate a file's media type, size and signature, however it arrived.

    Args:
        mime_type: The declared media type.
        data: The base64-encoded payload.
        label: Names the file in error messages.

    Returns:
        The validated :class:`Attachment`, without a filename or url.

    Raises:
        ValueError: If the file is missing, of an unsupported media type, over the
            per-attachment size cap, or does not match its declared type.
    """
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

    return Attachment(mime_type=mime_type, data=data, byte_size=byte_size)


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


async def resolve_attachment_envelopes(
    items: Optional[Sequence[AdditionalContext]],
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> Optional[list[AdditionalContext]]:
    """Replace every attachment reference in *items* with the inline payload it names.

    Runs once per turn, before anything reads the envelopes, so the rest of this module
    only ever sees the inline form. A reference that cannot be fetched comes back as an
    ``error`` on the envelope rather than raising: a fetch failure is a rejected file,
    not a failed turn, and :func:`parse_attachments` already turns a bad envelope into a
    message for the user and the model.

    Args:
        items: The turn's raw ``AdditionalContext`` list, possibly ``None``.
        gitlab_client: Client for the GitLab API, proxied through the executor.
        workflow_id: The session being run. A reference to any other session's file is
            refused here rather than fetched and found missing.

    Returns:
        A new list with references resolved and everything else unchanged, or ``None``
        when *items* is ``None``.
    """
    if items is None:
        return None

    # parse_attachments rejects a turn over the count cap anyway, so fetch nothing for it.
    if sum(item.category == ATTACHMENTS_CATEGORY for item in items) > MAX_ATTACHMENTS:
        return list(items)

    return list(
        await asyncio.gather(
            *(_resolve_item(item, gitlab_client, workflow_id) for item in items)
        )
    )


async def _resolve_item(
    item: AdditionalContext,
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> AdditionalContext:
    if item.category != ATTACHMENTS_CATEGORY:
        return item

    return await _resolve_one_envelope(item, gitlab_client, workflow_id)


async def _resolve_one_envelope(
    item: AdditionalContext,
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> AdditionalContext:
    """Resolve a single attachment envelope, leaving inline ones untouched."""
    try:
        payload = json.loads(item.content or "")
    except json.JSONDecodeError:
        # Malformed content is parse_attachments' to report, with its own message.
        return item

    if not isinstance(payload, dict):
        return item

    if payload.get("data"):
        return _without_client_url(item, payload)

    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return item

    try:
        fetched = await _fetch_session_file(url, gitlab_client, workflow_id)
    except ValueError as exc:
        return _with_payload(item, {**payload, "error": str(exc)})

    return _with_payload(
        item,
        {
            "mime_type": fetched.get("content_type"),
            "filename": payload.get("filename") or fetched.get("filename"),
            "data": fetched.get("data"),
            "url": url,
        },
    )


async def _fetch_session_file(
    url: str,
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> dict[str, Any]:
    """Fetch the session file *url* names.

    Raises:
        ValueError: If *url* is not a session-file path of this session, GitLab could
            not be reached, or it returned no usable file. The message is quoted to the
            user.
    """
    match = _SESSION_FILE_PATH.fullmatch(url)
    if not match:
        # The URL is client-supplied and unmatched, so it is not logged.
        log.warning(
            "Refused a session file reference",
            workflow_id=workflow_id,
            reason="not_a_session_file_path",
        )
        raise ValueError("it is not a file uploaded to this session")

    # A matched URL holds only numeric IDs, so it is safe to log from here on.
    if match.group("workflow_id") != str(workflow_id):
        log.warning(
            "Refused a session file reference",
            workflow_id=workflow_id,
            url=url,
            reason="other_session",
        )
        raise ValueError("it is not a file uploaded to this session")

    try:
        response = await gitlab_client.aget(path=url, parse_json=True)
    # The executor can raise timeouts, network errors or ToolException once its retries
    # run out; any of them is one unreadable file, not a reason to fail the turn.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning(
            "Could not fetch a session file",
            workflow_id=workflow_id,
            url=url,
            exception_class=type(exc).__name__,
        )
        raise ValueError("GitLab could not be reached") from exc

    if not response.is_success():
        log.warning(
            "Could not fetch a session file",
            workflow_id=workflow_id,
            url=url,
            status_code=response.status_code,
        )
        raise ValueError(f"GitLab returned HTTP {response.status_code}")

    body = response.body
    if not isinstance(body, dict) or not body.get("data"):
        log.warning(
            "Could not fetch a session file",
            workflow_id=workflow_id,
            url=url,
            reason="empty_body",
        )
        raise ValueError("GitLab returned no content for it")

    return body


def _with_payload(
    item: AdditionalContext, payload: dict[str, Any]
) -> AdditionalContext:
    return item.model_copy(update={"content": json.dumps(payload)})


def _without_client_url(
    item: AdditionalContext, payload: dict[str, Any]
) -> AdditionalContext:
    """Drop the url an inline file arrived with: it was not fetched from this session, so it has no address.

    Only a fetch sets ``url``, so the reference and the record never point at a file the user did not send.
    """
    if "url" not in payload:
        return item
    return _with_payload(
        item, {key: value for key, value in payload.items() if key != "url"}
    )


def attachment_content_blocks(
    attachments: Iterable[Attachment],
) -> list[dict[str, Any]]:
    """Build the content blocks carrying *attachments*.

    Each named attachment contributes two blocks: a short text label, ``[attachment N: filename]``, then the image
    itself. N matches the ``attachment-N`` id of its reference and record, so files with the same name stay apart
    within a turn. N restarts every turn, so it does not name a file across the thread. The image block uses the shared
    constructor, so an attached image is indistinguishable from an image any other producer emits.

    The label exists for what happens *after* this turn.
    :func:`~duo_workflow_service.entities.image_blocks.strip_image_payloads` drops the payload before the message is
    checkpointed and leaves a deliberately provenance-neutral placeholder, so a resumed session would otherwise read as
    a bare ``[image/png omitted from history]`` and the model could not answer even "which file did I send you?". A text
    block is not an image block, so it survives stripping untouched and keeps the filename in the conversation. It costs
    a handful of tokens and is not a recovery hint: code finds a session file again through the message's record (see
    :func:`session_file_records`), never through the label.
    """
    blocks: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        if attachment.filename:
            blocks.append(
                {"type": "text", "text": _attachment_label(index, attachment.filename)}
            )
        blocks.append(
            image_content_block(
                base64=attachment.data,
                mime_type=attachment.mime_type,
            )
        )
    return blocks


def _attachment_label(number: int | str, filename: str) -> str:
    """The label written before attachment *number*'s image, and rebuilt to find that image again."""
    return f"[attachment {number}: {filename}]"


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

    ``content`` never holds the payload, which must not re-enter the checkpoint by the
    back door. For a session file it holds the stored url and filename, so the client
    can fetch the file again on reload; otherwise it is empty rather than unset, because
    GitLab declares ``AiAdditionalContext.content`` non-nullable and fails the whole
    transcript query when it reads null there. These envelopes never reach a prompt
    template, and the model is not given the url: code finds it in the message's
    record (see :func:`session_file_records`).

    ``id`` is positional rather than the filename because the client interpolates it
    into a DOM id, and a filename carries spaces and dots.
    """
    return [
        AdditionalContext(
            category=ATTACHMENTS_CATEGORY,
            id=f"attachment-{index}",
            content=(
                json.dumps({"url": attachment.url, "filename": attachment.filename})
                if attachment.url
                else ""
            ),
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


def session_file_records(attachments: Sequence[Attachment]) -> list[dict[str, Any]]:
    """Return a record of each session file in *attachments*, for the user message to keep.

    The record, not the label, is where code finds the file again: the label is prose for the model. Each ``id`` is the
    ``attachment-N`` of the file's label and reference, counted over all of the turn's attachments.
    """
    return [
        {
            "id": f"attachment-{index}",
            "url": attachment.url,
            "filename": attachment.filename,
            "mime_type": attachment.mime_type,
        }
        for index, attachment in enumerate(attachments, start=1)
        if attachment.url
    ]


def session_files_in(message: BaseMessage) -> list[dict[str, Any]]:
    """Return the session file records *message* keeps, dropping any that do not point at a session file."""
    records = message.additional_kwargs.get(ATTACHMENTS_KWARG) or []
    return [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("id"), str)
        and isinstance(record.get("url"), str)
        and _SESSION_FILE_PATH.fullmatch(record["url"])
    ]


async def reinline_session_files(
    message: HumanMessage,
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
    cache: dict[str, Optional[dict[str, Any]]],
) -> HumanMessage:
    """Return *message* with each stripped session file image fetched again from the URL in its record.

    A checkpoint write replaces every image with a placeholder, and a tool approval resumes the turn from that
    checkpoint, so without this the model loses the user's image partway through the turn. Inline attachments keep
    their placeholder: only a session file can be fetched again.

    *cache* maps each URL to its image block, or ``None`` if it could not be read, so later model calls in the same
    run do not fetch it again.
    """
    if not isinstance(message.content, list):
        return message

    content = list(message.content)
    stripped = list(_stripped_session_file_images(content, session_files_in(message)))
    images = await asyncio.gather(
        *(
            _cached_session_file_image(url, cache, gitlab_client, workflow_id)
            for _, url in stripped
        )
    )
    reinlined = False
    for (index, _), image in zip(stripped, images):
        if image is None:
            continue

        content[index] = image
        reinlined = True

    return message.model_copy(update={"content": content}) if reinlined else message


def _stripped_session_file_images(
    content: Sequence[Any], records: Sequence[dict[str, Any]]
) -> Iterator[tuple[int, str]]:
    """Yield the index of each stripped session file image in *content*, with the URL its record keeps.

    A record's image is the placeholder right after the label written for it. The label is rebuilt from the record
    rather than parsed, so its wording is no contract and the user's text is never read as an address.
    """
    urls_by_label = _session_file_urls_by_label(records)
    for index in range(1, len(content)):
        url = urls_by_label.get(block_text(content[index - 1]) or "")
        if url and is_image_placeholder(content[index]):
            yield index, url


def _session_file_urls_by_label(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Map the label written for each named record to the URL that record keeps."""
    return {
        _attachment_label(
            record["id"].removeprefix("attachment-"), record["filename"]
        ): record["url"]
        for record in records
        if record.get("filename")
    }


async def _cached_session_file_image(
    url: str,
    cache: dict[str, Optional[dict[str, Any]]],
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> Optional[dict[str, Any]]:
    """Return the image block for *url*, fetching it only the first time each URL is seen."""
    if url not in cache:
        cache[url] = await _fetch_session_file_image(url, gitlab_client, workflow_id)
    return cache[url]


async def _fetch_session_file_image(
    url: str,
    gitlab_client: GitlabHttpClient,
    workflow_id: str,
) -> Optional[dict[str, Any]]:
    """Fetch and validate the session file *url* names, the same way a turn's reference is."""
    try:
        body = await _fetch_session_file(url, gitlab_client, workflow_id)
        attachment = _validated_attachment(
            body.get("content_type"), body.get("data"), url
        )
    except ValueError as exc:
        log.warning(
            "Could not reinline a session file",
            workflow_id=workflow_id,
            url=url,
            reason=str(exc),
        )
        return None

    return image_content_block(base64=attachment.data, mime_type=attachment.mime_type)
