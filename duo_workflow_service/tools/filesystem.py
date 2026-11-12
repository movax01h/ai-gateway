import json
import re
import uuid
from enum import IntEnum
from textwrap import dedent
from typing import Any, ClassVar, List, NamedTuple, Optional, Type
from urllib.parse import quote, unquote, urlparse, urlsplit

import gitmatch
import structlog
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, model_validator

from ai_gateway.models.v2._model_compat import current_model_supports_vision
from contract import contract_pb2
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.entities.image_response import (
    image_response_to_blocks,
    supported_image_formats_display,
)
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_accepting_image,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.executor.image_result import ImageActionResult
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.policies.file_exclusion_policy import (
    CONTEXT_EXCLUSION_MESSAGE,
    FileExclusionPolicy,
)
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from lib.context import gitlab_user_id, language_server_version
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

DEFAULT_READ_FILE_OFFSET = 0
DEFAULT_READ_FILE_LIMIT = 2000

_security_log = structlog.stdlib.get_logger("security")

# Trusted path segments for globally-installed agent skills, Duo plugins, and Duo config.
# Read-only tools need to access absolute paths (e.g. ~/.agents/skills/<skill>/SKILL.md or
# ~/.gitlab/duo/plugins/<plugin>/...) advertised via absolute file:// URIs by the
# workspace_agent_skills mechanism (!3201) and the per-user Duo plugins store.
#
# Scope is intentionally limited: `.agents` exposes only `skills/`, while the Duo config
# roots expose `skills/` and `plugins/`, so the rest of `.gitlab/duo` — which is in the
# denylist below — stays protected.
#
# Every entry is dotfile-anchored (first segment starts with ".") to prevent a CI
# repository checkout from matching: a repo containing a `gitlab/duo/skills/` directory
# must NOT be trusted via this mechanism. The XDG variant is therefore anchored to the
# conventional `~/.config` location rather than an arbitrary `$XDG_CONFIG_HOME`.
#
# Intentional fail-closed: skills under a non-standard `$GLAB_CONFIG_DIR` or
# `$XDG_CONFIG_HOME` are rejected server-side; the client-side gate
# (`getTrustedReadableDirectories`) is the authoritative second check.
TRUSTED_ABSOLUTE_PATH_SEGMENTS = (
    ".agents/skills",
    ".gitlab/duo/skills",
    ".config/gitlab/duo/skills",
    ".gitlab/duo/plugins",
    ".config/gitlab/duo/plugins",
)

# Path-traversal patterns rejected for every path, regardless of tool or trust level.
PATH_TRAVERSAL_PATTERNS = ("../", "..\\", "%2e%2e", "%252e%252e", "\u002e\u002e")

# NOTE appended to file-access tool descriptions to explain gitignore and secrets-denylist
# restrictions.  Kept in one place so all five tools stay in sync.
GITIGNORED_FILE_NOTE = (
    "NOTE on file-access restrictions:\n"
    "- Gitignored files: Cannot access files listed in .gitignore. If the file is not sensitive,\n"
    "    you may fall back to `run_command` with a shell command (e.g. `cat <file>`).\n"
    "    Do NOT use `git rm --cached` as a workaround.\n"
    "- Secrets-denylisted files: Cannot access `.env`, `.env.*`, `.ssh/`, `.gnupg/`,\n"
    "    `Dockerfile.secrets`, or similar sensitive paths.\n"
    "- Do NOT attempt to access secrets-denylisted files via `run_command` or any other shell command workaround."
)

# Security denylist of sensitive directories and files that should not be accessed
DEFAULT_CONTEXT_EXCLUSIONS = gitmatch.compile(
    [
        ".config/nvim",
        ".docker",
        ".emacs.d",
        ".env.*",
        ".env",
        ".git",
        ".gitlab/duo",
        ".gitlab/rules",
        ".gnupg",
        ".idea",
        ".metadata",
        ".settings",
        ".ssh",
        ".sublime-project",
        ".sublime-workspace",
        ".vim",
        ".vimrc",
        ".vscode",
        "Dockerfile.secrets",
        "!.env.example",
    ]
)


def _contains_path_traversal(file_path: str) -> bool:
    """Return `True` if *file_path* contains any known path-traversal pattern.

    Shared by `_is_trusted_absolute_path` (which fails closed) and
    `validate_duo_context_exclusions` (which raises) so the traversal denylist lives in
    one place and the two security-critical checks cannot drift apart.
    """
    return any(pattern in file_path for pattern in PATH_TRAVERSAL_PATTERNS)


def _is_trusted_absolute_path(file_path: str) -> bool:
    """Return True if *file_path* is absolute, traversal-free, and contains a trusted segment sequence.

    Traversal patterns are checked here so this helper is safe to call standalone — a path
    like `/home/u/.agents/skills/../../etc/passwd` is never reported as trusted.

    Args:
        file_path: File path to check (backslashes are normalised to `/` internally).

    Returns:
        True when the path is absolute, traversal-free, and contains a trusted segment
        sequence from `TRUSTED_ABSOLUTE_PATH_SEGMENTS`.
    """
    file_path = file_path.replace("\\", "/")
    if not file_path.startswith("/"):
        return False

    if _contains_path_traversal(file_path):
        return False

    segments = file_path.split("/")
    for trusted in TRUSTED_ABSOLUTE_PATH_SEGMENTS:
        parts = trusted.split("/")
        if any(
            segments[i : i + len(parts)] == parts
            for i in range(len(segments) - len(parts) + 1)
        ):
            return True
    return False


def validate_duo_context_exclusions(
    file_path: str, allow_trusted_absolute: bool = False
) -> None:
    """Check if the given file path is in the managed Duo Context Exclusion denylist of sensitive paths or contains path
    traversal attempts.

    Args:
        file_path: The file path to check.
        allow_trusted_absolute: When `True`, absolute paths whose segments include one
            of the entries in `TRUSTED_ABSOLUTE_PATH_SEGMENTS` are allowed through
            without hitting the `gitmatch` denylist (which rejects all absolute paths).
            Should only be set to `True` for read-only tools (`ReadFile`,
            `ReadFileChunked`, `ReadFiles`).  Write/edit/list tools keep the default
            `False` so they continue to reject absolute paths.

    Raises:
        ToolException: If the path is in the denylist or an invalid path.
    """
    if not file_path:
        return

    file_path = file_path.replace("\\", "/")
    while file_path.startswith("./"):
        file_path = file_path.replace("./", "", 1)

    # Traversal guard must run first — this prevents ~/.agents/../../etc/passwd from
    # being allowed even when allow_trusted_absolute is True.
    if _contains_path_traversal(file_path):
        raise ToolException(
            f"Access denied: Cannot access '{file_path}' as it contains path traversal patterns"
        )

    # gitmatch raises InvalidPathError on any absolute path, so trusted skill paths must
    # short-circuit before reaching it.
    if allow_trusted_absolute and _is_trusted_absolute_path(file_path):
        return

    try:
        excluded = DEFAULT_CONTEXT_EXCLUSIONS.match(file_path)
        if excluded is not None and bool(excluded):
            raise ToolException(
                f"Access denied: Cannot access '{file_path}' as it matches Duo Context Exclusion"
                f" patterns. Path '{excluded.path}' matches excluded pattern: '{excluded.pattern}'."
            )
    except gitmatch.InvalidPathError as ex:
        raise ToolException(
            f"Access denied: Not accessing invalid path '{file_path}'. {ex!s}"
        )

    if file_path != file_path.lower():
        validate_duo_context_exclusions(
            file_path.lower(), allow_trusted_absolute=allow_trusted_absolute
        )
        return


# Descriptions and conversion flip together on two switches, both required
# (_image_support_enabled): a model refuses image reads unless the description
# advertises them, and advertising without conversion only earns refusals. The
# instance flag is evaluated per user rather than per client, so on its own it
# would advertise to an older client that still refuses binaries. Tools are
# built once per run and both contexts are set per request, so a flip lands on
# the next run.
IMAGE_READ_CAPABILITY = "read_file_image"

_READ_FILE_IMAGE_NOTE = f"""Image files ({supported_image_formats_display()}) are supported: reading one returns the
    actual image so you can see its contents.

    """

_READ_FILE_CHUNKED_IMAGE_NOTE = f"""\
- Image files ({supported_image_formats_display()}) are supported and return the actual image so you can see its contents.
    - Offset/limit do not apply to images.

    """

# Only a client whose allowlist takes other owners can download these URLs
# (_upload_owner_urls_supported); an older client is told the old scope instead.
_READ_FILE_UPLOAD_URL_SENTENCE = """For an upload on an epic, a group
    work item, or another project you can read, pass its full GitLab URL, as
    GitLab shows it or built from the owner's full path, e.g.
    `https://<host>/groups/<group path>/-/uploads/<secret>/shot.png` or
    `https://<host>/<project path>/uploads/<secret>/shot.png`."""
_READ_FILE_UPLOAD_NO_URL_SENTENCE = (
    "Uploads on epics or other groups' items are not reachable this way."
)

# Shown only when this run may download uploads: image support on, and a GitLab
# privilege granted, since the download is a GitLab API read.
_READ_FILE_UPLOAD_NOTE = f"""Images uploaded to issues or merge requests in the current project are
    also supported: pass the upload reference exactly as it appears in the
    markdown, e.g. `/uploads/<secret>/screenshot.png`. It is downloaded
    using the user's own GitLab credentials. {_READ_FILE_UPLOAD_URL_SENTENCE}

    """

_READ_FILES_IMAGE_NOTE = """Image files are not supported here: read them individually with read_file.

    """


def _image_support_enabled() -> bool:
    """Whether tool-read images are on for this run: the instance flag and the client capability, both."""
    return is_feature_enabled(FeatureFlag.DAP_TOOL_IMAGE_INPUT) and is_client_capable(
        IMAGE_READ_CAPABILITY
    )


# An upload download is a GitLab API read with the user's credential, so it also
# needs a GitLab read privilege. The registry fills each read privilege in from its
# write superset before it hands the set over, so one name is enough here.
_UPLOAD_READ_PRIVILEGE = "read_only_gitlab"


def _granted_privileges(metadata: Any) -> frozenset[str]:
    """The run's privileges from tool metadata; empty when no run built the tool (ListTools)."""
    return (metadata or {}).get("granted_privileges") or frozenset()


def _upload_reads_enabled(metadata: Any) -> bool:
    """Whether this run may download uploads: image support on, and a GitLab read privilege granted."""
    return _image_support_enabled() and _UPLOAD_READ_PRIVILEGE in _granted_privileges(
        metadata
    )


def _upload_owner_urls_supported() -> bool:
    """Whether the client's allowlist takes uploads of groups and other projects (Language Server 10.2.0)."""
    version = language_server_version.get()
    return bool(version and version.supports_upload_owner_urls())


def _strip_note_unless(tool: DuoBaseTool, note: str, enabled: bool) -> None:
    if not enabled:
        tool.description = tool.description.replace(note, "")


def _gate_read_file_notes(tool: DuoBaseTool, image_note: str) -> None:
    _strip_note_unless(tool, image_note, _image_support_enabled())
    _strip_note_unless(
        tool, _READ_FILE_UPLOAD_NOTE, _upload_reads_enabled(tool.metadata)
    )
    if not _upload_owner_urls_supported():
        tool.description = tool.description.replace(
            _READ_FILE_UPLOAD_URL_SENTENCE, _READ_FILE_UPLOAD_NO_URL_SENTENCE
        )


_NO_VISION_NOTE = (
    "This model cannot see images. The file was read, but the image will not be shown to it. "
    "Do not read it again; ask the user to describe it, or to switch to a model with image support."
)


def _image_response_to_blocks_if_enabled(
    image: ImageActionResult, file_path: str
) -> str | list[dict[str, Any]]:
    """Convert a typed image result when image support is on, refuse readably when off.

    A client sends image responses whatever the server-side switches say, so the off paths have to answer in words the
    model can act on. The two refusals differ so a mismatch is diagnosable from the transcript alone: the instance flag
    is off, or the client sent an image without having declared the capability.
    """
    if _image_support_enabled():
        blocks = image_response_to_blocks(image, file_path=file_path)
        if isinstance(blocks, list) and current_model_supports_vision() is False:
            blocks = [*blocks, {"type": "text", "text": _NO_VISION_NOTE}]
        return blocks
    if not is_feature_enabled(FeatureFlag.DAP_TOOL_IMAGE_INPUT):
        return (
            f'Cannot read file: "{file_path}" is an image file, and image '
            "support is not enabled on this instance."
        )
    return (
        f'Cannot read file: "{file_path}" is an image file, and this client '
        "did not declare image support."
    )


# The two shapes GitLab gives an upload reference. In Markdown it is
# `/uploads/<secret>/<filename>` with no owner: the owner is whatever the
# Markdown belongs to, which for this tool is the session project. Rendered, or
# copied from the browser, it is a full URL that names the owner, in one of the
# four forms GitLab itself produces:
#   https://<host>/-/project/<id>/uploads/...     Banzai::Filter::UploadLinkFilter
#   https://<host>/-/group/<id>/uploads/...       same filter, group-owned
#   https://<host>/<group>/<project>/uploads/...  project web route
#   https://<host>/groups/<group>/-/uploads/...   group web route
# Secrets are 32 hex chars today, 10 to 32 on old uploads (FileUploader
# VALID_SECRET_PATTERN), and a filename never contains a slash. The slashless
# `uploads/<secret>/<filename>` form is not a reference: it is also a valid
# repository path, and it is the only shape where the two collide.
_UPLOAD_SECRET = r"[0-9a-f]{10,32}"
_UPLOAD_SECRET_RE = re.compile(_UPLOAD_SECRET)
_BARE_UPLOAD_REF = re.compile(
    rf"\A/uploads/(?P<secret>{_UPLOAD_SECRET})/(?P<filename>[^/]+)\Z"
)
# The characters GitLab allows in a namespace path segment.
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
# ASCII digits only: `isdigit()` and `\d` also take other scripts' digits,
# which `int()` reads back as a different id for the log. At most 19 digits,
# the width of a bigint id; `int()` refuses strings past 4300 digits.
_ASCII_ID_RE = re.compile(r"[0-9]{1,19}")
# `.` and `..` address a directory, never a file or a namespace; `-` is the
# segment GitLab reserves for its own routes, never a namespace.
_RESERVED_SEGMENTS = frozenset({".", "..", "-"})

# Appended to a failed download of the Markdown form. That form always asks
# the session project, so a "not found" may mean another owner. Worded as a
# condition, so a "too large" or "not an image" failure does not send the
# model to retry with a URL.
_UPLOAD_HINT_BARE = (
    "If the error says not found, this upload may be on an epic, a group work "
    "item, or another project: pass its full GitLab URL instead."
)

# Without the leading slash an upload reference is read as a repository path
# (see above). When that read fails, the model is pointed at the form that
# downloads it.
_SLASHLESS_UPLOAD_REF = re.compile(rf"\Auploads/{_UPLOAD_SECRET}/[^/]+\Z")


def _add_slashless_upload_hint(
    error: ToolException, metadata: Any, file_path: str
) -> None:
    """Extend a failed read of ``uploads/<secret>/<file>`` with the slashed form, when this run may read uploads."""
    if _SLASHLESS_UPLOAD_REF.match(file_path) and _upload_reads_enabled(metadata):
        error.args = (
            f"{error}\nIf this is a GitLab upload, pass it with the leading slash: "
            f'"/{file_path}".',
            *error.args[1:],
        )


class UploadTarget(NamedTuple):
    """Where an upload reference resolves to, and how the security log names its owner."""

    request_path: str
    entity_type: str
    entity_id: Optional[int]
    entity_path: Optional[str]
    bare: bool


def _upload_api_path(kind: str, owner: str, secret: str, filename: str) -> str:
    return f"/api/v4/{kind}/{owner}/uploads/{secret}/{quote(filename, safe='')}"


def _resolve_bare_reference(
    match: re.Match[str], project: Optional[Project]
) -> UploadTarget | None:
    # The Markdown form carries no owner and the API lookup is parent-scoped,
    # so it means the session project.
    if not project or not project.get("id"):
        return None
    if match["filename"] in _RESERVED_SEGMENTS:
        return None
    web_url = project.get("web_url")
    return UploadTarget(
        request_path=_upload_api_path(
            "projects", str(project["id"]), match["secret"], match["filename"]
        ),
        entity_type="Project",
        entity_id=project["id"],
        entity_path=(urlparse(web_url).path.strip("/") or None) if web_url else None,
        bare=True,
    )


class _UploadOwner(NamedTuple):
    """An upload's owner as a URL names it: how the API path addresses it, and how the log names it."""

    is_group: bool
    # The numeric id as given, or the full path as one percent-encoded segment.
    api_ref: str
    entity_id: Optional[int]
    entity_path: Optional[str]


def _upload_url_segments(
    file_path: str, gitlab_host: Optional[str]
) -> list[str] | None:
    """The path segments of a URL on this GitLab instance, or ``None`` for anything else."""
    # `urlsplit` would quietly lower-case the scheme, strip leading spaces, drop
    # tabs and newlines, and split off a query or fragment, so a URL it would
    # change is not a reference.
    if (
        not gitlab_host
        or not file_path.startswith(("http://", "https://"))
        or any(char in file_path for char in "\t\n\r?#")
    ):
        return None
    try:
        url = urlsplit(file_path)
    except ValueError:  # for example an unbalanced IPv6 bracket in the host
        return None
    # A URL on another host is not a reference to this instance. It stays an
    # ordinary path, so no credential is spent on it; the client only talks to
    # its own GitLab anyway.
    segments = url.path[1:].split("/")
    if url.netloc != gitlab_host or "" in segments:
        return None
    return segments


def _is_namespace_path(segments: list[str]) -> bool:
    return bool(segments) and all(
        _PATH_SEGMENT_RE.fullmatch(segment) and segment not in _RESERVED_SEGMENTS
        for segment in segments
    )


def _owner_by_path(segments: list[str], *, is_group: bool) -> _UploadOwner:
    path = "/".join(segments)
    return _UploadOwner(is_group, quote(path, safe=""), None, path)


def _upload_url_owner(segments: list[str]) -> _UploadOwner | None:
    """The owner named by the part of the URL before ``/uploads/``, in the four forms GitLab produces."""
    if segments[0] == "-":
        # `/-/project/<id>` or `/-/group/<id>`, as Rails renders a link.
        if (
            len(segments) == 3
            and segments[1] in ("project", "group")
            and _ASCII_ID_RE.fullmatch(segments[2])
        ):
            return _UploadOwner(
                segments[1] == "group", segments[2], int(segments[2]), None
            )
        return None
    if segments[0] == "groups":
        # `/groups/<group path>/-`, the group web route.
        if segments[-1] == "-" and _is_namespace_path(segments[1:-1]):
            return _owner_by_path(segments[1:-1], is_group=True)
        return None
    # `/<project path>`, the project web route. `api` is a reserved top-level
    # path, so it never starts one.
    if segments[0] != "api" and _is_namespace_path(segments):
        return _owner_by_path(segments, is_group=False)
    return None


def _resolve_url_reference(
    file_path: str, gitlab_host: Optional[str]
) -> UploadTarget | None:
    segments = _upload_url_segments(file_path, gitlab_host)
    if segments is None or len(segments) < 4:
        return None
    *owner_segments, marker, secret, raw_filename = segments
    owner = _upload_url_owner(owner_segments)
    # GitLab escapes the filename when it renders the link, so decode it once
    # before the API path quotes it again. A decoded `/` would change the path
    # the API sees, and GitLab filenames never hold one.
    filename = unquote(raw_filename)
    if (
        owner is None
        or marker != "uploads"
        or not _UPLOAD_SECRET_RE.fullmatch(secret)
        or "/" in filename
        or filename in _RESERVED_SEGMENTS
    ):
        return None
    return UploadTarget(
        request_path=_upload_api_path(
            "groups" if owner.is_group else "projects", owner.api_ref, secret, filename
        ),
        entity_type="Group" if owner.is_group else "Project",
        entity_id=owner.entity_id,
        entity_path=owner.entity_path,
        bare=False,
    )


def _resolve_upload_reference(
    file_path: str, project: Optional[Project], gitlab_host: Optional[str]
) -> UploadTarget | None:
    """Map an upload reference onto its REST API download path and its owner.

    The executor matches the returned path strictly before attaching the user's credential:
    ``projects`` or ``groups``, then the owner as a numeric id or a percent-encoded full path
    (uppercase ``%2F`` between segments, as ``quote`` emits), so this shape and the
    client-side check have to stay in sync.
    """
    # Upload reading is part of gated image support, both switches: with the
    # flag off or a client that has not declared the capability, the reference
    # stays an ordinary path (today's pre-feature behavior) and no download is
    # ever asked for. An older client could not fetch the API path anyway.
    if not _image_support_enabled():
        return None
    if match := _BARE_UPLOAD_REF.match(file_path):
        return _resolve_bare_reference(match, project)
    # Only a client whose allowlist takes other owners can fetch these; for an
    # older one the URL stays an ordinary path.
    if _upload_owner_urls_supported():
        return _resolve_url_reference(file_path, gitlab_host)
    return None


def _author_id() -> Optional[int]:
    """The requesting user's instance-local id, as the standard's integer ``author_id``."""
    raw = gitlab_user_id.get()
    return int(raw) if raw and raw.isdigit() else None


def _log_upload_read(
    target: UploadTarget,
    *,
    outcome: str,
    response_type: Optional[str],
    size_bytes: Optional[int],
    reason: Optional[str] = None,
) -> None:
    """Record the service's decision to ask for an upload, in the security logging standard's shape.

    The SIEM-bound record is the Rails ``ai_tool_invoked`` audit event, which already carries the tool arguments and
    the author. This line is the service's own part, and it fires on every attempt because a failed download is still
    credential spend.

    ``outcome`` reports the download, not what the model received: ``success`` means the client answered the request,
    even when the payload then failed the image checks. That case reads as ``size_bytes`` set with ``response_type``
    ``text``, since the bytes were pulled into the service before the check refused them.

    ``reason`` names why the service itself refused before any request went out, so a blocked reference reads
    differently from a failed download in a SIEM search; it is ``None`` whenever the download was attempted. The
    standard allows only ``success``, ``failure`` and ``unknown`` for ``outcome``, hence a separate field.

    ``entity_type``, ``entity_id`` and ``entity_path`` name the owner the request was addressed to. For a URL
    reference that is what the model gave, by id or by path, not what Rails answered.

    Fields this layer cannot fill: ``author_name`` (only ids reach the service), ``target_id`` (uploads are addressed
    by secret and filename), ``ip_address`` (the request arrives over gRPC) and ``details.token_type`` (the client
    holds the credential). The upload secret stays out because it is part of the download URL.
    """
    _security_log.info(
        "Tool read resolved a GitLab upload reference",
        id=str(uuid.uuid4()),
        event_type="data.read.upload",
        author_id=_author_id(),
        entity_type=target.entity_type,
        entity_id=target.entity_id,
        entity_path=target.entity_path,
        target_type="upload",
        # Not `filename`: stdlib LogRecord reserves that name and raises.
        # Decoded, so the value matches the upload's own name in a SIEM search.
        target_details=unquote(target.request_path.rsplit("/", 1)[-1]),
        details={
            "outcome": outcome,
            "provider": "duo_workflow_service",
            "response_type": response_type,
            "reason": reason,
        },
        gitlab={"data_type": "upload", "size_bytes": size_bytes},
    )


async def _read_upload_reference(
    metadata: Any, project: Optional[Project], file_path: str
) -> str | list | None:
    """Download ``file_path`` as an upload, or return ``None`` if it is not one.

    Both read tools funnel through here because ``ReadFileChunked`` supersedes ``ReadFile``
    under the same tool name, so a difference between the two would be invisible to a test
    that exercises only one.

    Args:
        metadata: Tool metadata carrying the executor outbox.
        project: The workflow's project, which scopes the download.
        file_path: The path the model asked for.

    Returns:
        The tool response for an upload reference, or ``None`` when
        ``file_path`` is an ordinary path the caller should handle itself.
    """
    target = _resolve_upload_reference(
        file_path, project, (metadata or {}).get("gitlab_host")
    )
    if target is None:
        return None

    # Refuse readably rather than fall through to a file read: a planted
    # reference in a flow without GitLab access is what the trail is for.
    if _UPLOAD_READ_PRIVILEGE not in _granted_privileges(metadata):
        _log_upload_read(
            target,
            outcome="failure",
            response_type=None,
            size_bytes=None,
            reason="missing_privilege",
        )
        return (
            f'Cannot read upload: "{file_path}" is a GitLab upload, '
            "and this flow has no GitLab API access."
        )

    # An agent pulling an upload into model context is worth a trail;
    # see _log_upload_read for what the trail is and is not.
    # The hint suggests a URL, so only a client that can fetch one gets it.
    hint = _UPLOAD_HINT_BARE if target.bare and _upload_owner_urls_supported() else None
    outcome = "failure"
    response_type: Optional[str] = None
    size_bytes: Optional[int] = None
    try:
        # offset/limit are meaningless for a downloaded image and are not sent.
        response = await _execute_action_accepting_image(
            metadata,
            contract_pb2.Action(
                runReadFile=contract_pb2.ReadFile(filepath=target.request_path)
            ),
        )
        converted: str | list
        if isinstance(response, ImageActionResult):
            size_bytes = len(response.data)
            converted = _image_response_to_blocks_if_enabled(response, file_path)
        else:
            # The client answered with text, so the download itself failed
            # (not found, too large, not an image, or an older client that read
            # the API path as a file). For the Markdown form, tell the model
            # what to try next.
            converted = f"{response}\n{hint}" if hint else response
        response_type = "image" if isinstance(converted, list) else "text"
        outcome = "success"
        return converted
    except ToolException as error:
        # The executor reports a refused download as an action error. Keep the
        # exception as it is, with its response and server_error for the
        # error handler, and only extend the message the model reads.
        if hint:
            error.args = (f"{error}\n{hint}", *error.args[1:])
        raise
    finally:
        _log_upload_read(
            target,
            outcome=outcome,
            response_type=response_type,
            size_bytes=size_bytes,
        )


class ReadFileInput(BaseModel):
    file_path: str = Field(description="the file_path to read the file from")


class ReadFile(DuoBaseTool):
    name: str = "read_file"
    description: str = f"""Read the contents of a file.

    {_READ_FILE_IMAGE_NOTE}{_READ_FILE_UPLOAD_NOTE}Batching:
    - When multiple files need inspection, emit multiple read_file calls concurrently in a single turn.
    - Do not make separate turns for each file - group all related file reads together.
    - Avoid redundant re-reads of files that are unchanged since you last read them.

    {GITIGNORED_FILE_NOTE}
    """
    args_schema: Type[BaseModel] = ReadFileInput
    handle_tool_error: bool = True
    eval_prompts: List[str] = [
        "I need to read the content of the `readme.md`",
        "Let me check if class `DuoBaseTool` exists in `./tools/base.py`",
    ]

    @model_validator(mode="after")
    def _gate_image_support_description(self) -> "ReadFile":
        _gate_read_file_notes(self, _READ_FILE_IMAGE_NOTE)
        return self

    async def _execute(self, file_path: str) -> str | list[dict[str, Any]]:
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        # Upload references are remote GitLab content, not workspace paths, so
        # they skip the filesystem exclusion check (which rejects all absolute
        # paths) and go to the executor as the owner's API path.
        upload = await _read_upload_reference(self.metadata, self.project, file_path)
        if upload is not None:
            return upload

        validate_duo_context_exclusions(file_path, allow_trusted_absolute=True)

        try:
            response = await _execute_action_accepting_image(
                self.metadata,  # type: ignore
                contract_pb2.Action(
                    runReadFile=contract_pb2.ReadFile(filepath=file_path)
                ),
            )
        except ToolException as error:
            _add_slashless_upload_hint(error, self.metadata, file_path)
            raise
        if isinstance(response, ImageActionResult):
            return _image_response_to_blocks_if_enabled(response, file_path)
        return response

    def format_display_message(
        self, args: ReadFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Read file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class ReadFileChunkedInput(BaseModel):
    file_path: str = Field(description="the file_path to read the file from")
    offset: int = Field(
        default=DEFAULT_READ_FILE_OFFSET,
        description="Starting line number (0-indexed). Use with limit for reading large files in chunks.",
    )
    limit: int = Field(
        default=DEFAULT_READ_FILE_LIMIT,
        description="Number of lines to read from offset. Use for reading large files in chunks.",
    )


class ReadFileChunked(DuoBaseTool):
    """Enhanced read_file tool with offset/limit support for chunked reading of large files.

    Supersedes ReadFile when the client declares the 'read_file_chunked' capability, ensuring the LLM only receives
    offset/limit parameters when the executor can honour them.
    """

    name: str = "read_file"
    description: str = f"""Read a file from the local filesystem.

    Batching:
    - When multiple files need inspection, emit multiple read_file calls concurrently in a single turn.
    - Do not make separate turns for each file - group all related file reads together.
    - Avoid redundant re-reads of files that are unchanged since you last read them.

    Usage:
    - Only read files directly relevant to the current task. Do NOT speculatively read unrelated files or entire directories.
    - Returns up to 2000 lines from offset (0-indexed).
    - For large files (>100 lines), specify offset and limit to inspect only the relevant section.
    {_READ_FILE_CHUNKED_IMAGE_NOTE}{_READ_FILE_UPLOAD_NOTE}
    {GITIGNORED_FILE_NOTE}
    """
    args_schema: Type[BaseModel] = ReadFileChunkedInput
    handle_tool_error: bool = True
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = ReadFile
    required_capability: ClassVar[frozenset[str]] = frozenset({"read_file_chunked"})

    @model_validator(mode="after")
    def _gate_image_support_description(self) -> "ReadFileChunked":
        _gate_read_file_notes(self, _READ_FILE_CHUNKED_IMAGE_NOTE)
        return self

    async def _execute(
        self,
        file_path: str,
        offset: int = DEFAULT_READ_FILE_OFFSET,
        limit: int = DEFAULT_READ_FILE_LIMIT,
    ) -> str | list[dict[str, Any]]:
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        # Same upload branch as ReadFile: this class replaces it under the same
        # tool name whenever the client is chunked-capable, so upload
        # references land here on modern clients.
        upload = await _read_upload_reference(self.metadata, self.project, file_path)
        if upload is not None:
            return upload

        validate_duo_context_exclusions(file_path, allow_trusted_absolute=True)

        try:
            response = await _execute_action_accepting_image(
                self.metadata,  # type: ignore
                contract_pb2.Action(
                    runReadFile=contract_pb2.ReadFile(
                        filepath=file_path, offset=offset, limit=limit
                    )
                ),
            )
        except ToolException as error:
            _add_slashless_upload_hint(error, self.metadata, file_path)
            raise
        if isinstance(response, ImageActionResult):
            return _image_response_to_blocks_if_enabled(response, file_path)
        return response

    def format_display_message(
        self, args: ReadFileChunkedInput, _tool_response: Any = None
    ) -> str:
        msg = "Read file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class ReadFilesInput(BaseModel):
    file_paths: list[str] = Field(description="List of file paths to read")


class ReadFiles(DuoBaseTool):
    name: str = "read_files"
    description: str = f"""Read one or more files in a single operation.

    {_READ_FILES_IMAGE_NOTE}{GITIGNORED_FILE_NOTE}
    """
    args_schema: Type[BaseModel] = ReadFilesInput
    handle_tool_error: bool = True

    @model_validator(mode="after")
    def _gate_image_support_description(self) -> "ReadFiles":
        _strip_note_unless(self, _READ_FILES_IMAGE_NOTE, _image_support_enabled())
        return self

    async def _execute(self, file_paths: list[str]) -> str:
        policy = FileExclusionPolicy(self.project)
        file_paths, excluded_file_paths = policy.filter_allowed(file_paths)
        log = structlog.stdlib.get_logger("workflow")

        for file_path in file_paths:
            validate_duo_context_exclusions(file_path, allow_trusted_absolute=True)

        result_dict = {}

        if file_paths:
            file_contents_result_action_response = (
                await _execute_action_and_get_action_response(
                    self.metadata,  # type: ignore
                    contract_pb2.Action(
                        runReadFiles=contract_pb2.ReadFiles(filepaths=file_paths)
                    ),
                )
            )

            if not file_contents_result_action_response:
                log.error("Received empty grpc response")
                return "Could not read files"

            file_contents_result = (
                file_contents_result_action_response.plainTextResponse.response
            )
            try:
                result_dict = json.loads(file_contents_result)
            except json.JSONDecodeError as e:
                plain_text_response = (
                    file_contents_result_action_response.plainTextResponse
                )
                error_msg = f"Could not parse file contents as JSON: {e}"
                if (
                    plain_text_response
                    and hasattr(plain_text_response, "error")
                    and plain_text_response.error
                ):
                    error_msg = (
                        f"{error_msg}. Executor error: {plain_text_response.error}"
                    )
                raise ToolException(error_msg)

        for path in excluded_file_paths:
            result_dict[path] = {"error": CONTEXT_EXCLUSION_MESSAGE}

        return json.dumps(result_dict)

    def format_display_message(
        self, args: ReadFilesInput, tool_response: Any = None
    ) -> str:
        file_count = len(args.file_paths)
        excluded_files_msg = ""

        if tool_response:
            excluded_files = [
                path
                for path, data in json.loads(tool_response.content).items()
                if data.get("error") == CONTEXT_EXCLUSION_MESSAGE
            ]

            excluded_files_msg = FileExclusionPolicy.format_user_exclusion_message(
                excluded_files
            )

            file_count -= len(excluded_files)

        return f"Read {file_count} file{'s' if file_count != 1 else ''}{excluded_files_msg}"


class WriteFileInput(BaseModel):
    file_path: str = Field(
        description="The path of the new file to create. The file must not already exist."
    )
    contents: str = Field(
        description="The full contents to write into the newly created file. Must be non-empty."
    )


class WriteFile(DuoBaseTool):
    name: str = "create_file_with_contents"
    description: str = dedent(
        f"""\
        Use this tool to create a brand new file with the given contents.

        IMPORTANT:
        - This tool is only for creating files that do not yet exist. It writes the full contents of a new file.
        - Do not use this tool to modify, append to, or overwrite an existing file. If the file already exists, use other dedicated tools instead.

        {GITIGNORED_FILE_NOTE}"""
    )
    args_schema: Type[BaseModel] = WriteFileInput
    handle_tool_error: bool = True
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, file_path: str, contents: str) -> str:
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        validate_duo_context_exclusions(file_path)

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runWriteFile=contract_pb2.WriteFile(
                    filepath=file_path, contents=contents
                )
            ),
        )

    def format_display_message(
        self, args: WriteFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Create file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class FilesScopeEnum(IntEnum):
    ALL = 0
    TRACKED = 1
    UNTRACKED = 2
    MODIFIED = 3
    DELETED = 4


class FindFilesInput(BaseModel):
    name_pattern: str = Field(description="The pattern to search for files.")


class FindFiles(DuoBaseTool):
    name: str = "find_files"
    description: str = """Find files by name patterns (equivalent to 'find' command).

    **Primary use cases:**
    - Discover codebase structure and locate files before searching inside them
    - Find files by filename or extension patterns
    - Locate specific files across the codebase
    - Get list of files matching naming conventions

    **Best practice:**
    Use `find_files` first to locate candidate files and understand repository layout before running `grep`.

    **Replaces these commands:**
    - find . -name "*.py" → find_files(name_pattern="*.py")
    - find tests -name "test_*.js" → find_files(name_pattern="tests/**/test_*.js")
    - find src -name "*.json" → find_files(name_pattern="src/**/*.json")

    **Examples:**
    - All Python files: find_files(name_pattern="*.py")
    - Test files: find_files(name_pattern="test_*.py")
    - Config files: find_files(name_pattern="*.json")
    - Files directly in a directory: find_files(name_pattern="src/*.js")
    - Files anywhere under a directory: find_files(name_pattern="src/**/*.js")

    **Don't use this for:**
    - Searching text content within files (use grep instead)
    - Finding where functions/variables are used (use grep instead)

    Uses gitignore-style glob syntax and respects .gitignore rules:
    - A pattern without `/` matches file names at any depth.
    - A pattern with `/` is matched from the repository root, and `*` does not cross `/`.
    - `**` matches any number of directories.
    """
    args_schema: Type[BaseModel] = FindFilesInput

    async def _execute(
        self,
        name_pattern: str,
    ) -> str:
        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                findFiles=contract_pb2.FindFiles(
                    name_pattern=name_pattern,
                )
            ),
        )

        policy = FileExclusionPolicy(self.project)
        lines = result.strip().split("\n") if result.strip() else []
        allowed_files, _excluded_files = policy.filter_allowed(lines)

        response_parts = []
        if allowed_files:
            response_parts.append("\n".join(allowed_files))

        return (
            "\n\n".join(response_parts)
            if response_parts
            else _format_no_matches_message(name_pattern)
        )

    def format_display_message(
        self, args: FindFilesInput, _tool_response: Any = None
    ) -> str:
        return f"Search files with pattern `{args.name_pattern}`"


class MkdirInput(BaseModel):
    directory_path: str = Field(
        description="The directory path to create. Must be within the current working directory tree."
    )


class Mkdir(DuoBaseTool):
    name: str = "mkdir"
    description: str = """Create a new directory using the mkdir command.
    The directory creation is restricted to the current working directory tree."""

    args_schema: Type[BaseModel] = MkdirInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, directory_path: str) -> str:
        if ".." in directory_path:
            return "Creating directories above the current directory is not allowed"

        if not directory_path.startswith("./") and directory_path != ".":
            directory_path = f"./{directory_path}"

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                mkdir=contract_pb2.Mkdir(
                    directory_path=directory_path,
                )
            ),
        )

    def format_display_message(
        self, args: MkdirInput, _tool_response: Any = None
    ) -> str:
        return f"Create directory `{args.directory_path}`"


class EditFileInput(BaseModel):
    file_path: str = Field(
        description="The path of the existing file to edit. The file must already exist."
    )
    old_str: str = Field(
        "",
        description=(
            "The exact text to replace. Must match the file content character-for-character, "
            "including whitespace and indentation. Include enough surrounding context (typically "
            "at least one line above and below the change) so the text appears exactly once in the "
            "file; only the first occurrence is replaced."
        ),
    )
    new_str: str = Field(
        "",
        description=(
            "The text to insert in place of `old_str`. Provide the full replacement block, keeping "
            "any surrounding context lines from `old_str` unchanged. Use an empty string to delete "
            "the matched text."
        ),
    )


class EditFile(DuoBaseTool):
    name: str = "edit_file"
    description: str = dedent(
        f"""\
        Use this tool to edit an existing file by replacing `old_str` with `new_str`.

        Batching:
        - When a code change affects multiple files (or code and tests), dispatch independent edit_file calls concurrently in a single turn.
        - Avoid sequential single-file edit turns and intermediate git status checks.

        IMPORTANT:
        - You must read the file with read_file before editing it.
        - `old_str` must match the file exactly (including whitespace) and be unique; include enough surrounding context. Only the first match is replaced.
        - Secret-like values may appear as `[REDACTED]`. Anchor edits on surrounding non-secret text.

        {GITIGNORED_FILE_NOTE}"""
    )
    args_schema: Type[BaseModel] = EditFileInput
    handle_tool_error: bool = True
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, file_path: str, old_str: str, new_str: str) -> str:
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        validate_duo_context_exclusions(file_path)

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runEditFile=contract_pb2.EditFile(
                    filepath=file_path,
                    oldString=old_str,
                    newString=new_str,
                )
            ),
        )

    def format_display_message(
        self, args: EditFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Edit file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class ListDirInput(BaseModel):
    directory: str = Field(description="Directory path relative to the repository root")


class ListDir(DuoBaseTool):
    name: str = "list_dir"
    description: str = """List directory contents (equivalent to 'ls -la' command).

    **Primary use cases:**
    - See all files and subdirectories in a directory
    - Check if files or directories exist
    - Explore project structure and organization
    - Get directory contents before reading specific files

    **Replaces these commands:**
    - ls -la → list_dir(directory=".")
    - ls -l → list_dir(directory=".")
    - ls → list_dir(directory=".")
    - ls src/ → list_dir(directory="src/")
    - ls -la tests/ → list_dir(directory="tests/")
    - dir → list_dir(directory=".") (Windows equivalent)

    **Examples:**
    - List current directory: list_dir(directory=".")
    - List source code: list_dir(directory="src/")
    - Check if directory exists: list_dir(directory="tests/")
    - Explore subdirectory: list_dir(directory="config/")

    Shows files and subdirectories relative to the repository root.
    Use this instead of trying to run 'ls' commands.
    """
    args_schema: Type[BaseModel] = ListDirInput

    async def _execute(self, directory: str) -> str:
        if not FileExclusionPolicy.is_allowed_for_project(self.project, directory):
            return FileExclusionPolicy.format_llm_exclusion_message([directory])

        validate_duo_context_exclusions(directory)

        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                listDirectory=contract_pb2.ListDirectory(directory=directory)
            ),
        )

        policy = FileExclusionPolicy(self.project)
        lines = result.strip().split("\n") if result.strip() else []
        allowed_files, _excluded_files = policy.filter_allowed(lines)

        response_parts = []
        if allowed_files:
            response_parts.append("\n".join(allowed_files))

        return "\n\n".join(response_parts)


def _format_no_matches_message(pattern, search_directory=None):
    search_scope = f" in '{search_directory}'" if search_directory else ""
    return f"No matches found for pattern '{pattern}'{search_scope}."
