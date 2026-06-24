"""Secret redaction for tool outputs.

Uses the ``detect-secrets`` library (Yelp/detect-secrets) to locate
secret patterns and replace them with ``[REDACTED]``.

Two redaction functions are provided with different detector sets:

``redact_secrets``
    Structured/regex-based detectors only (GitLab tokens, JWTs, AWS keys,
    etc.).  Applied inside ``DuoBaseTool._arun`` so every tool result is
    scrubbed before it enters LLM context.  Entropy-based detectors are
    intentionally excluded here: they produce too many false positives on
    ordinary API response payloads (JSON field names, git SHAs, checksums)
    which would corrupt the data the LLM relies on.

``redact_secrets_for_ui``
    Structured detectors **plus** tuned entropy detectors.  Applied when
    tool results are written to ``UiChatLog`` (via ``build_tool_info`` in
    ``state.py``).  The UI path shows data to humans rather than feeding it
    to an LLM, so a conservative false-positive rate is acceptable in
    exchange for broader coverage.  Thresholds are raised above the defaults
    to avoid redacting git SHAs, MD5/SHA-256 checksums, and UUIDs.
    ``Base64HighEntropyString`` limit is raised from 4.5 to 4.0 to catch
    high-entropy API tokens (Azure storage keys, generic base64 blobs) while
    keeping all known GitLab API field names safe (max observed entropy: 3.72).
    ``HexHighEntropyString`` limit is raised from 3.0 to 3.7 to skip git SHAs
    (entropy <= 3.61) and SHA-256/MD5 digests (entropy <= 3.67) while still
    catching truly random hex tokens (entropy > 3.7).

Supported input types
---------------------
Both public functions accept any of the following via duck typing, and
recurse into nested structures automatically:

- ``str`` -- redacted in place.
- ``dict`` -- values are redacted recursively; keys are left unchanged.
- ``list`` -- each element is redacted recursively.
- Objects with an ``update`` field (e.g. ``langgraph.types.Command``) --
    ``update`` is redacted recursively; a shallow copy is returned when the field changed.
- Objects with a ``content`` attribute (e.g. ``langchain_core.messages.ToolMessage``) --
    ``content`` is redacted recursively; a shallow copy is returned when changed.
    Pydantic models are copied with ``model_copy``; plain objects via ``copy.copy``.
- Scalar values (``int``, ``float``, ``bool``, ``None``) -- returned unchanged.

Usage::

    from duo_workflow_service.security.secret_redaction import (
        redact_secrets,
        redact_secrets_for_ui,
    )

    # LLM context (structured patterns only)
    clean = redact_secrets("plain text", tool_name="my_tool")

    # UI display (structured + entropy), works directly with ToolMessage
    safe = redact_secrets_for_ui(tool_message, tool_name="my_tool")
    # -> ToolMessage with secrets replaced in .content
"""

import copy
from typing import Any, List

import structlog
from detect_secrets.plugins.artifactory import ArtifactoryDetector
from detect_secrets.plugins.aws import AWSKeyDetector
from detect_secrets.plugins.azure_storage_key import AzureStorageKeyDetector
from detect_secrets.plugins.base import BasePlugin, RegexBasedDetector
from detect_secrets.plugins.basic_auth import BasicAuthDetector
from detect_secrets.plugins.cloudant import CloudantDetector
from detect_secrets.plugins.discord import DiscordBotTokenDetector
from detect_secrets.plugins.github_token import GitHubTokenDetector
from detect_secrets.plugins.gitlab_token import GitLabTokenDetector
from detect_secrets.plugins.high_entropy_strings import (
    Base64HighEntropyString,
    HexHighEntropyString,
)
from detect_secrets.plugins.ibm_cloud_iam import IbmCloudIamDetector
from detect_secrets.plugins.ibm_cos_hmac import IbmCosHmacDetector
from detect_secrets.plugins.jwt import JwtTokenDetector
from detect_secrets.plugins.keyword import KeywordDetector
from detect_secrets.plugins.mailchimp import MailchimpDetector
from detect_secrets.plugins.npm import NpmDetector
from detect_secrets.plugins.openai import OpenAIDetector
from detect_secrets.plugins.private_key import PrivateKeyDetector
from detect_secrets.plugins.pypi_token import PypiTokenDetector
from detect_secrets.plugins.sendgrid import SendGridDetector
from detect_secrets.plugins.slack import SlackDetector
from detect_secrets.plugins.softlayer import SoftlayerDetector
from detect_secrets.plugins.square_oauth import SquareOAuthDetector
from detect_secrets.plugins.stripe import StripeDetector
from detect_secrets.plugins.telegram_token import TelegramBotTokenDetector
from detect_secrets.plugins.twilio import TwilioKeyDetector

log = structlog.stdlib.get_logger("security")

REDACTED_PLACEHOLDER = "[REDACTED]"

# ---------------------------------------------------------------------------
# Detector sets
# ---------------------------------------------------------------------------

# Structured (regex/pattern) detectors – no false positives on API payloads.
_STRUCTURED_DETECTORS: List[BasePlugin] = [
    ArtifactoryDetector(),
    AWSKeyDetector(),
    AzureStorageKeyDetector(),
    BasicAuthDetector(),
    CloudantDetector(),
    DiscordBotTokenDetector(),
    GitHubTokenDetector(),
    GitLabTokenDetector(),
    IbmCloudIamDetector(),
    IbmCosHmacDetector(),
    JwtTokenDetector(),
    KeywordDetector(),
    MailchimpDetector(),
    NpmDetector(),
    OpenAIDetector(),
    PrivateKeyDetector(),
    PypiTokenDetector(),
    SendGridDetector(),
    SlackDetector(),
    SoftlayerDetector(),
    SquareOAuthDetector(),
    StripeDetector(),
    TelegramBotTokenDetector(),
    TwilioKeyDetector(),
]

# Entropy detectors with raised thresholds – safe for human-facing UI logs.
# See module docstring for the rationale behind each limit value.
_ENTROPY_DETECTORS: List[BasePlugin] = [
    Base64HighEntropyString(limit=4.0),
    HexHighEntropyString(limit=3.7),
]

_UI_DETECTORS: List[BasePlugin] = _STRUCTURED_DETECTORS + _ENTROPY_DETECTORS


# ---------------------------------------------------------------------------
# Core redaction helpers
# ---------------------------------------------------------------------------


def _redact_string(text: str, detectors: List[BasePlugin]) -> str:
    """Redact secret-like patterns from a single string using the given detectors.

    For ``RegexBasedDetector`` subclasses the raw ``denylist`` regex patterns
    are applied directly so that the *full* matched token is replaced, not just
    the capture group stored in ``secret_value``.

    For ``HighEntropyStringsPlugin`` detectors ``analyze_string`` yields
    candidate values; each candidate is only replaced when its Shannon entropy
    exceeds the detector's configured limit.

    For any other ``BasePlugin`` detector ``analyze_string`` is used and every
    non-empty yielded value is replaced unconditionally.

    Args:
        text: The string to scan and redact.
        detectors: The list of detectors to apply.

    Returns:
        The string with any detected secrets replaced by ``[REDACTED]``.
    """
    for detector in detectors:
        if isinstance(detector, RegexBasedDetector):
            for pattern in detector.denylist:
                text = pattern.sub(REDACTED_PLACEHOLDER, text)
        elif isinstance(detector, (Base64HighEntropyString, HexHighEntropyString)):
            for secret_value in detector.analyze_string(text):
                if (
                    secret_value
                    and detector.calculate_shannon_entropy(secret_value)
                    > detector.entropy_limit
                ):
                    text = text.replace(secret_value, REDACTED_PLACEHOLDER)
        else:
            # Generic BasePlugin fallback (e.g. KeywordDetector)
            for secret_value in detector.analyze_string(text):
                if secret_value:
                    text = text.replace(secret_value, REDACTED_PLACEHOLDER)
    return text


def _shallow_copy_with(obj: Any, field: str, value: Any) -> Any:
    """Return a shallow copy of ``obj`` with one field replaced.

    ``copy.copy`` is used for all object types -- Pydantic models, frozen
    dataclasses, and plain objects -- because it correctly preserves the
    concrete type and shares all other fields by reference (same semantics as
    ``model_copy`` and ``dataclasses.replace`` for a shallow copy).

    ``object.__setattr__`` is then used to bypass write protection present on both
    Pydantic v2 models (whose ``__setattr__`` rejects writes after construction) and
    frozen dataclasses (which raise ``FrozenInstanceError``). Skipping Pydantic
    validation is intentional: we write a redacted string back into a field that already
    held a string, so there is nothing new to validate. For frozen dataclasses this is
    the same mechanism used internally by ``dataclasses.replace``.

    Args:
        obj: The object to copy.
        field: Name of the field to replace in the copy.
        value: New value for that field.

    Returns:
        A shallow copy of ``obj`` with ``field`` set to ``value``.
    """
    cloned = copy.copy(obj)
    object.__setattr__(cloned, field, value)
    return cloned


def _redact_recursive(  # noqa: PLR0911  # branchy recursive redaction over value types
    response: Any, detectors: List[BasePlugin], tool_name: str
) -> Any:
    """Recursively apply ``_redact_string`` over supported value types.

    Handles ``str``, ``dict``, ``list``, objects with an ``update`` field
    (e.g. ``langgraph.types.Command``), objects with a ``content`` attribute
    (e.g. ``langchain_core.messages.ToolMessage``), and scalar leaf values.
    All object copies are shallow; see ``_shallow_copy_with`` for details.

    Args:
        response: The value to scan and redact.
        detectors: The list of detectors to apply.
        tool_name: Name of the originating tool, used for log context.

    Returns:
        The value with any detected secrets replaced by ``[REDACTED]``.
    """
    if isinstance(response, str):
        redacted = _redact_string(response, detectors)
        if redacted != response:
            log.warning(
                "Secret-like content detected and redacted from tool response",
                tool_name=tool_name,
            )
        return redacted

    if isinstance(response, dict):
        return {
            k: _redact_recursive(v, detectors, tool_name) for k, v in response.items()
        }

    if isinstance(response, list):
        return [_redact_recursive(item, detectors, tool_name) for item in response]

    if hasattr(response, "update"):
        original_update = response.update
        redacted_update = _redact_recursive(original_update, detectors, tool_name)
        if redacted_update is not original_update:
            return _shallow_copy_with(response, "update", redacted_update)
        return response

    if hasattr(response, "content"):
        redacted_content = _redact_recursive(response.content, detectors, tool_name)
        if redacted_content is not response.content:
            return _shallow_copy_with(response, "content", redacted_content)
        return response

    return response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def redact_secrets(response: Any, tool_name: str) -> Any:
    """Redact structured secrets from a tool response before it enters LLM context.

    Uses only regex/pattern-based detectors.  Entropy detectors are excluded
    to avoid false positives on ordinary API payloads (JSON field names, git
    SHAs, file content, etc.) that would corrupt the data the LLM reasons over.

    Accepts ``str``, ``dict``, ``list``, objects with a ``content`` attribute
    (e.g. ``langchain_core.messages.ToolMessage``), or scalar values.

    Args:
        response: The tool response to scan.
        tool_name: Name of the originating tool, used for log context.

    Returns:
        The response with structured secrets replaced by ``[REDACTED]``.
        Objects with a ``content`` attribute are returned as shallow copies.
    """
    return _redact_recursive(response, _STRUCTURED_DETECTORS, tool_name)


def redact_secrets_for_ui(response: Any, tool_name: str) -> Any:
    """Redact secrets from tool output destined for UiChatLog display.

    Applies structured detectors **plus** entropy detectors with raised
    thresholds (Base64 >= 4.0, Hex >= 3.7) to catch high-entropy blobs such
    as Azure storage keys and generic API tokens without flagging git SHAs,
    checksums, or UUIDs.

    This function is intentionally separate from ``redact_secrets`` because
    the UI path shows data to humans, making a conservative false-positive
    rate acceptable.  It must **not** be used in the LLM context path where
    false positives corrupt reasoning data.

    Accepts ``str``, ``dict``, ``list``, objects with a ``content`` attribute
    (e.g. ``langchain_core.messages.ToolMessage``), or scalar values.

    Args:
        response: The tool response to scan.
        tool_name: Name of the originating tool, used for log context.

    Returns:
        The response with secrets replaced by ``[REDACTED]``.
        Objects with a ``content`` attribute are returned as shallow copies.
    """
    return _redact_recursive(response, _UI_DETECTORS, tool_name)
