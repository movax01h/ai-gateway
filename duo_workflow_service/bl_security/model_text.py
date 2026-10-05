"""Pure helpers that check and clean the title and impact the model writes for a BL finding.

No I/O: the report writer passes in the raw values from the finding. A value that fails a check is dropped, and the
writer keeps its own text.
"""

import re
from typing import Optional

TITLE_MAX = 120

# Bounds a model-written title or impact must meet to be used; outside them the writer's own text is used.
MODEL_TITLE_MIN = 10

MODEL_IMPACT_MAX = 400

_CWE_PREFIX = re.compile(r"^\s*CWE[-\s]?\d+\s*[:\-\u2013\u2014]?\s*", re.I)

_ESCAPED_QUOTE = re.compile(r"\\(['\"])")

# A line start markdown would read as a heading, quote, list item or numbered item.
_BLOCK_START = re.compile(r"^(\d*)([#>+*.)-])(?=\s)")

# A code span or a bare URL (kept as is), else what is unsafe outside them: an HTML tag start, which would hide a
# `<token>` placeholder; an `@name`, which GitLab would turn into a user or group mention; and a GitLab reference
# sigil (`#1`, `!1`, `%1`, `&1`, `~label`, `~"label"`, also after a `group/project` path), which would link an
# unrelated issue, MR, milestone, epic or label.
_INLINE_HAZARD = re.compile(
    r"(?P<code>(`+)(?:(?!\2).)+?\2)|(?P<url>https?://[^\s<`]+)|<|(?<![\w.@])@[\w.-]*\w|[#!%&](?=\d)|~(?=[\w\"])",
    re.S,
)


def _escape_block_start(sentence: str) -> str:
    """Backslash-escape a leading heading, quote or list marker, so it stays text."""
    return _BLOCK_START.sub(r"\1\\\2", sentence)


def _escape_inline(sentence: str) -> str:
    """Neutralise :data:`_INLINE_HAZARD` outside code spans and URLs: ``@name`` becomes code, ``<`` and a reference
    sigil are backslash-escaped."""

    def _fix(m: re.Match) -> str:
        text = m.group(0)
        if m.group("code") or m.group("url"):
            return text
        return f"`{text}`" if text.startswith("@") else "\\" + text

    return _INLINE_HAZARD.sub(_fix, sentence)


def escape_markdown(sentence: str) -> str:
    """``sentence`` made safe to show as markdown text.

    Outside code spans and bare URLs: no HTML tag, no ``@`` mention, no GitLab reference (``#1``, ``!1``, ``%1``,
    ``&1``, ``~label``), and no leading heading, quote or list marker.
    """
    return _escape_block_start(_escape_inline(sentence))


def plain_line(text: str) -> str:
    """``text`` as one plain line: escaped quotes unescaped, non-printables and whitespace runs collapsed."""
    text = _ESCAPED_QUOTE.sub(r"\1", str(text))
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(text.split())


def model_title(raw: object) -> Optional[str]:
    """The model's title as one plain line, or ``None`` when it is missing or not sane.

    Not sane: not a string, more than one line, or shorter than ``MODEL_TITLE_MIN`` or longer than ``TITLE_MAX``
    once cleaned. A leading CWE number, trailing ``.``, ``!`` or ``…``, and backticks are removed: the title is plain
    text in the vulnerability list.
    """
    if not isinstance(raw, str) or "\n" in raw.strip():
        return None
    title = _CWE_PREFIX.sub("", plain_line(raw).replace("`", "")).rstrip(".!\u2026 ")
    return title if MODEL_TITLE_MIN <= len(title) <= TITLE_MAX else None


def model_impact(raw: object) -> Optional[str]:
    """The model's impact sentence as one plain line, or ``None`` when it is missing, empty or too long."""
    if not isinstance(raw, str):
        return None
    impact = plain_line(raw)
    return impact if 0 < len(impact) <= MODEL_IMPACT_MAX else None
