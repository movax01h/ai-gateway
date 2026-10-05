"""Pure text helpers that turn a BL finding into the report's title and markdown description.

No I/O: the report writer passes in what it knows (CWE, location, code excerpt, model text).
"""

import re
from textwrap import dedent
from typing import Optional

from duo_workflow_service.bl_security.model_text import (
    TITLE_MAX,
    escape_markdown,
    plain_line,
)

#: A short name for the weakness, based on MITRE's CWE name, for the finding title.
SHORT_NAMES = {
    "200": "Exposure of sensitive information to an unauthorized actor",
    "284": "Improper access control",
    "285": "Improper authorization",
    "287": "Improper authentication",
    "362": "Race condition",
    "367": "Time-of-check time-of-use (TOCTOU) race condition",
    "459": "Incomplete cleanup",
    "639": "Authorization bypass through user-controlled key",
    "840": "Business logic error",
    "862": "Missing authorization",
    "863": "Incorrect authorization",
    "915": "Improperly controlled modification of dynamically-determined object attributes",
}

#: One plain sentence on what the weakness lets someone do, for the description's "What" part.
IMPACTS = {
    "200": "A user can read information they are not allowed to see.",
    "284": "A user can perform an action that should be restricted to someone else.",
    "285": "A user can perform an action without having permission for it.",
    "287": "A caller can act without proving who they are, or as someone else.",
    "362": "Two requests sent at the same time can both pass a check that only one of them should pass.",
    "367": "A value is checked and then used after it may have changed, so the check can be bypassed.",
    "459": "Access that was removed can still be used, because something that grants it was not cleaned up.",
    "639": "A caller can read or change records that belong to another user, team or tenant by changing an ID in the request.",
    "840": "A user can break a business rule, for example by skipping a step or repeating an action.",
    "862": "Any user who can reach this code can use it, because it checks no permission.",
    "863": "The permission check here can be passed by a user who should be refused.",
    "915": "A user can set fields they should not control, such as an owner, a role or a price.",
}

_FALLBACK_NAME = "Business-logic finding"

# A sentence ends at . or ! followed by whitespace, or at ? followed by whitespace and a capital letter. Abbreviations
# ("vs.", "etc.", "approx.", "cf.", and any single letter or digit before a dot, which covers "e.g.", "i.e.", an
# initial like "J." and a list number like "1.") and an ellipsis ("...", "…") do not end one. A dot inside a file name
# or code (`basket.ts`, `req.body.id`) is never followed by whitespace, so it does not either. The model often starts a
# sentence with a lowercase identifier (`security.appendUserId() computes ...`), so after . or ! the next sentence may
# start with any character; after ? it must start with a capital, because "?" also appears in code and query strings.
_SENTENCE_END = re.compile(
    r"(?<!\bvs\.)(?<!\betc\.)(?<!\bapprox\.)(?<!\bcf\.)(?<!\b[A-Za-z0-9]\.)(?<!\.\.\.)"
    r"(?:(?<=[.!])\s+(?=\S)|(?<=\?)\s+(?=[A-Z]))"
)

# Inline code spans, kept whole while splitting so a "." inside one never ends a sentence.
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1", re.S)


def clip_title(text: str, limit: int = TITLE_MAX) -> str:
    """``text`` cut to ``limit`` characters on a word boundary, ending in an ellipsis; never mid-word."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    whole_word = text[limit - 1] == " " or " " not in cut
    head = cut if whole_word else cut.rsplit(" ", 1)[0]
    return head.rstrip(" ,;:-") + "\u2026"


def inline_code(text: str) -> str:
    """``text`` as a markdown code span, fenced with more backticks than any run inside it."""
    text = plain_line(text)
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def code_block(text: str) -> str:
    """``text`` as a fenced markdown code block whose fence no line inside it can close."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text.rstrip()}\n{fence}"


def split_sentences(text: str) -> list[str]:
    """The sentences of ``text``, each as one plain line.

    Code spans are masked first, so nothing inside backticks can end a sentence.
    """
    text = plain_line(text)
    spans: list[str] = []

    def _mask(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = _CODE_SPAN.sub(_mask, text)
    parts = _SENTENCE_END.split(masked)
    unmask = re.compile(r"\x00(\d+)\x00")
    return [
        unmask.sub(lambda m: spans[int(m.group(1))], p).strip()
        for p in parts
        if p.strip()
    ]


def fallback_title(cwe: str, file: str) -> str:
    """A plain-text title built from fixed parts: the CWE short name in the file.

    A path too long for the title loses whole leading directories, or is cut on the left when its last part alone is
    too long. When the name leaves no room for even one character of the path, the title is the name alone. The title
    is never longer than ``TITLE_MAX``.
    """
    name = clip_title(SHORT_NAMES.get(cwe) or (f"CWE-{cwe}" if cwe else _FALLBACK_NAME))
    place = plain_line(file).replace("`", "")
    if not place:
        return name
    room = TITLE_MAX - len(name) - len(" in ")
    if len(place) > room:
        if room < 2:  # no room for "…" and one character: the name alone
            return name
        tail = place[-(room - 1) :]
        rest = tail.partition("/")[2]
        place = "\u2026" + (rest or tail)
    return f"{name} in {place}"


def markdown_description(
    *,
    cwe: str,
    file: str,
    line: int,
    excerpt: str,
    body: str,
    impact: Optional[str] = None,
) -> str:
    """The markdown description: What, Where and Details. A part with nothing to say is left out.

    What is ``impact`` (the model's sentence, already checked by ``model_text.model_impact``, escaped like a Details
    sentence) when given, else the fixed sentence for the CWE. The fix is not part of it: when the report has a
    ``solution``, that shows as its own section.
    """
    parts = []
    what = escape_markdown(impact) if impact else IMPACTS.get(cwe, "")
    if what:
        parts.append(f"**What**\n\n{what}")
    if file:
        where = f"{file}:{line}" if line else file
        block = f"**Where**\n\n{inline_code(where)}"
        if excerpt.strip():
            block += f"\n\n{code_block(dedent(excerpt).strip(chr(10)))}"
        parts.append(block)
    sentences = split_sentences(body)
    if sentences:
        bullets = "\n".join(f"- {escape_markdown(s)}" for s in sentences)
        parts.append(f"**Details**\n\n{bullets}")
    return "\n\n".join(parts)
