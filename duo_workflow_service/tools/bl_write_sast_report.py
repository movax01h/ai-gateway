"""Deterministic (non-LLM) BL-analyzer tool: write the SAST report.

A pure-code step meant to run inside DeterministicStepComponent, whose node calls tool.ainvoke(args) and publishes the
return at context:<name>.tool_responses.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, ClassVar, List, Optional, Type

import structlog
from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.bl_security.cwe_guidance import SOLUTIONS, owasp_identifier
from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    executor_truncated,
)
from duo_workflow_service.bl_security.finding_text import (
    fallback_title,
    markdown_description,
)
from duo_workflow_service.bl_security.findings import (
    TRIAGE_CLAUSE_MAX,
    WHITESPACE_RUN,
    audit_clause_of,
    coerce_to_list,
    cwe_digits,
    dedup_key,
    excerpt_of,
    findings_input,
    line_of,
    norm_excerpt,
    verdict_of,
)
from duo_workflow_service.bl_security.model_text import model_impact, model_title
from duo_workflow_service.bl_security.target_files import resolve_target_files
from duo_workflow_service.executor.action import _execute_action, _read_file_fully
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.filesystem import validate_duo_context_exclusions

__all__ = ["BlWriteSastReport"]

_log = structlog.stdlib.get_logger(__name__)

_ANALYZER_ID = "gitlab_bl_security_analyzer"

_ANALYZER_NAME = "GitLab Business-Logic Security Analyzer"

_VENDOR = "GitLab"

_ANALYZER_VERSION = "0.1.0"

# The first schema version that defines `scan.partial_scan`.
_REPORT_SCHEMA_VERSION = "15.2.2"

_SEVERITY_MAP = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "info": "Info",
    "informational": "Info",
    "unknown": "Unknown",
}


# Lines that identify no defect on their own: a block terminator or bare punctuation is
# byte-identical everywhere in a file. Anything else with no alphanumeric character counts too.
_STRUCTURAL_ONLY_LINES = frozenset({"end", "}", ")", "do", "else", "{", "];", "end)"})


def _is_structural_line(line: str) -> bool:
    """Return True when ``line`` carries no defect-identifying content by itself.

    Containment merging (:func:`_collapse_contained_excerpts`) treats one excerpt as a re-report of
    another when its LINES are a subset of the other's. A set made only of these tokens is a subset
    of almost every longer excerpt, so without this guard a finding whose excerpt is nothing but
    ``end`` would be silently swallowed by an unrelated finding in the same bucket.
    """
    stripped = line.strip().lower()
    if not stripped:
        return True
    if stripped in _STRUCTURAL_ONLY_LINES:
        return True
    return not any(ch.isalnum() for ch in stripped)


def _excerpt_line_set(finding: dict) -> frozenset:
    """The finding's excerpt as a set of non-empty, stripped, lowercased lines."""
    return frozenset(
        ln.strip().lower() for ln in excerpt_of(finding).splitlines() if ln.strip()
    )


def _has_substantive_line(lines) -> bool:
    """True when at least one line in ``lines`` is more than structural noise."""
    return any(not _is_structural_line(ln) for ln in lines)


def _collapse_contained_excerpts(findings: List[dict]) -> List[dict]:
    """Collapse re-reports that quote a LONGER OR SHORTER SPAN of one site, keeping the fuller quote.

    The exact key in :func:`dedup_key` demands a byte-identical normalized excerpt, so three
    re-reports of one defect that quote the statement, the statement plus its trailing ``end``, and
    just the inner line respectively keep three distinct keys and all survive.

    THE RULE IS LINE-SET SUBSET, NEVER RAW SUBSTRING. ``"activate(user)"`` IS a substring of
    ``"deactivate(user)"``, and those are two genuinely different handlers that re-anchor into the
    same bucket in the same file; a substring rule would merge them. As line SETS,
    ``{"activate(user)"}`` is not a subset of ``{"deactivate(user)"}``, so they stay apart.

    Scope is deliberately narrow: only findings already sharing ``(cwe, file, line bucket)`` are
    compared, and the smaller excerpt must contain at least one substantive (non-structural) line —
    see :func:`_is_structural_line` — so a bare ``end`` neither swallows nor is swallowed.
    """
    groups: dict = {}
    for idx, f in enumerate(findings):
        key = (cwe_digits(f), f.get("file") or f.get("path") or "", line_of(f) // 4)
        groups.setdefault(key, []).append(idx)

    dropped: set = set()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        line_sets = {i: _excerpt_line_set(findings[i]) for i in idxs}
        # Fullest quote first; input order breaks ties, so of two equal line-sets the earlier wins.
        kept: List[int] = []
        for i in sorted(idxs, key=lambda i: (-len(line_sets[i]), i)):
            contained = any(line_sets[i] <= line_sets[k] for k in kept)
            if contained and _has_substantive_line(line_sets[i]):
                dropped.add(i)
            else:
                kept.append(i)

    return [f for i, f in enumerate(findings) if i not in dropped]


def _merge_overlapping_spans(findings: List[dict], contents: dict) -> List[dict]:
    """Merge same-CWE findings in one file whose located quotes share a line unique in that file.

    Re-reports that quote different parts of one handler (its decorator, then its query) share no excerpt, so neither
    earlier pass sees them as one. A shared line counts only when :func:`_is_structural_line` is false for it. The kept
    finding is the verified one, then the one on the lowest line, then the first.
    """
    counts = {
        file: Counter(ln.strip() for ln in text.splitlines())
        for file, text in contents.items()
        if text
    }
    kept: List[dict] = []
    owner: dict = {}
    for f in findings:
        file = f.get("file") or f.get("path") or ""
        keys: set = set()
        if file in counts and f.get("anchor_status") in (
            ANCHOR_VERIFIED,
            ANCHOR_CORRECTED,
        ):
            keys = {
                (cwe_digits(f), file, ln.strip())
                for ln in excerpt_of(f).splitlines()
                if not _is_structural_line(ln) and counts[file][ln.strip()] == 1
            }
        hit = min((owner[k] for k in keys if k in owner), default=None)
        if hit is None:
            hit = len(kept)
            kept.append(f)
        elif _merge_rank(f) < _merge_rank(kept[hit]):
            kept[hit] = f
        owner.update(dict.fromkeys(keys, hit))
    return kept


def _merge_rank(finding: dict) -> tuple:
    return (finding.get("anchor_status") != ANCHOR_VERIFIED, line_of(finding))


def _collapse_after_reanchor(
    findings: List[dict], contents: Optional[dict] = None
) -> List[dict]:
    """Re-apply the exact-dedup key AFTER anchor correction, keeping the first of each group in input order.

    WHY A SECOND PASS. Dedup (`BlDedupFindings`) never reads the source files, so the only line it can key on is the one
    the MODEL claimed - and the model claims a different line each time it re-reports the SAME defect, so the re-reports
    land in different line buckets and all survive dedup.
    `BlWriteSastReport._verify_anchors` runs a stage later, reads the file, and corrects them all to the one real line,
    where their keys become identical. Collapsing again here, once the lines are real, is the only place that can see
    them as one.

    WHY IT IS SAFE FOR REAL DUPLICATION. N genuinely copy-pasted sites quote identical code but sit at N different
    lines, and `_find_excerpt_line` resolves each finding to the occurrence NEAREST its own hint - so real sites keep N
    distinct lines and survive, while N re-reports of a single site all converge on that site's one line and collapse.
    Findings whose excerpt was never located stay `unverified` on their claimed lines and are likewise left alone.
    """
    kept: List[dict] = []
    seen: set = set()
    for f in findings:
        key = dedup_key(f)
        if key in seen:
            continue
        seen.add(key)
        kept.append(f)
    # Second pass over the exact-key survivors: re-reports that quote a longer or shorter span of
    # the same site keep distinct exact keys, so only line-set containment can see them as one.
    # Running it after (never instead of) the exact pass makes the result strictly more collapsed.
    # Last, quotes of different parts of one handler; ``contents`` holds the file texts read by
    # `_verify_anchors`.
    return _merge_overlapping_spans(_collapse_contained_excerpts(kept), contents or {})


def _fingerprint_of(finding: dict) -> str:
    """The location-independent identity of a finding: its normalized code excerpt, or a body prefix when the pass that
    produced it emits no excerpt."""
    body = finding.get("body") or finding.get("description") or ""
    return norm_excerpt(finding) or str(body)[:200]


# A function or method declaration: Python/Ruby/Elixir, Go, JS/TS/PHP functions, JS/TS function or arrow assignments,
# Java/C#/Kotlin methods (a return type, modifier or `fun` before the name), and JS/TS class methods.
_DECL_RES = tuple(
    re.compile(p)
    for p in (
        r"^\s*(?:async\s+)?defp?\s+(?:self\.)?([A-Za-z_]\w*[?!]?)",
        r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)",
        r"^\s*(?:(?:export|default|public|private|protected|static|abstract|final|async)\s+)*function\s*\*?\s*([\w$]+)",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*(?::[^=]+)?=\s*(?:async\s+)?"
        r"(?:function\b|\([^)]*(?:$|\)\s*(?::[^=]+)?=>)|[\w$]+\s*=>)",
        r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|override|async|suspend|open|virtual"
        r"|synchronized)\s+)*(?:fun\s+|(?!(?:return|new|else|throw|await|yield|if|elif|while|for|assert|raise|not"
        r"|puts|print|echo|when|with|except|go|defer|case|lock|using)\b)"
        r"[\w<>\[\],.?]+\s+)([A-Za-z_]\w*)\s*\([^;]*$",
        r"^\s*(?:(?:async|static|get|set)\s+)*([\w$]+)\s*\([^)]*\)\s*\{\s*$",
    )
)

_NOT_A_NAME = frozenset(
    {"if", "for", "while", "switch", "catch", "function", "return", "foreach", "elseif"}
    | {"synchronized", "lock", "using", "with", "except"}
)

_DECORATOR_RE = re.compile(r"^\s*(?:@|\[\w[^\]]*\]\s*$)")

_MODULE_SCOPE = "<module>"


def _declared_name(line: str) -> str:
    for rx in _DECL_RES:
        m = rx.match(line)
        if m and m.group(1) not in _NOT_A_NAME:
            return m.group(1)
    return ""


def _enclosing_function(text: str, line: int) -> str:
    """The function declared at or around ``line``: below a decorator, else the nearest less-indented one above."""
    lines = text.splitlines()
    if not 1 <= line <= len(lines):
        return _MODULE_SCOPE
    at = lines[line - 1]
    if _DECORATOR_RE.match(at):
        return next(
            filter(None, map(_declared_name, lines[line : line + 10])), _MODULE_SCOPE
        )
    depth = len(at) - len(at.lstrip())
    for n in range(line - 1, -1, -1):
        name = _declared_name(lines[n])
        if name and (n == line - 1 or len(lines[n]) - len(lines[n].lstrip()) < depth):
            return name
    return _MODULE_SCOPE


def _tracking_signatures(findings: List[dict], sources: dict) -> List[Optional[str]]:
    """Each finding's ``scope_offset`` value, ``<file>|<function>[<k>]:<cwe>``; ``None`` when its file was not read.

    ``k`` numbers the same-CWE findings in one function by anchored line, never by model output order.
    """
    located = []
    for i, f in enumerate(findings):
        file = f.get("file") or f.get("path") or ""
        text = sources.get(file)
        # An unlocated quote has no reliable line, so no stable scope either.
        if text and _anchor_status_of(f) in (ANCHOR_VERIFIED, ANCHOR_CORRECTED):
            fn = _enclosing_function(text, line_of(f))
            key = (file, fn, cwe_digits(f))
            located.append((key, line_of(f), _fingerprint_of(f), i))
    out: List[Optional[str]] = [None] * len(findings)
    seen: Counter = Counter()
    for key, _, _, i in sorted(located):
        out[i] = f"{key[0]}|{key[1]}[{seen[key]}]:{key[2]}"
        seen[key] += 1
    return out


def _shared_fingerprints(findings: List[dict]) -> set:
    """The (file, fingerprint) pairs carried by MORE THAN ONE finding.

    N separate sites that miss the same control quote near-identical (often byte-identical) code, so they collide on
    this pair. The caller disambiguates exactly those with the line number, leaving every other finding's ID line-free.

    THE CWE IS DELIBERATELY ABSENT FROM THIS KEY, AND MUST STAY ABSENT. This key has to be the SAME key the ID is
    built from in :meth:`BlWriteSastReport._build_report`, which excludes the CWE (the reason is documented there).
    Keying the collision detector on the CWE while the ID ignores it is a silent-collision bug: two findings differing
    ONLY by CWE would hash to the SAME id, yet count as two separate keys here, so neither would get a disambiguating
    line - two vulnerabilities arriving as one.
    """
    counts: dict = {}
    for f in findings:
        key = (
            f.get("file") or f.get("path") or "",
            _fingerprint_of(f),
        )
        counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n > 1}


def _find_excerpt_line(content: str, excerpt: str, hint: int = 0) -> int:
    """1-based line where `excerpt` first appears in `content`, matched on whitespace-collapsed text (LLM excerpts
    rarely preserve exact indentation).

    LLM-emitted line numbers are often wrong, but the COPIED code is reliable —
    so we locate the code and trust that line. When the excerpt matches several
    lines, pick the one closest to the LLM's `hint` line (its region guess is
    usually roughly right even when the exact number isn't). Returns 0 if not
    found (caller then keeps the LLM line).

    KEY CHOICE IS THE WHOLE GAME. A multi-line excerpt almost never matches ONE
    source line once collapsed, so the fallback key decides the anchor. A quote
    very often opens on a bare decorator, annotation or attribute (of the
    `@post_only` / `[MustBeSignedIn]` shape) that recurs at many unrelated sites
    in the same file, so keying on the first line and resolving by the unreliable
    hint lands the finding on an arbitrary same-annotation neighbour — possibly
    the correctly guarded sibling. Once two findings are re-anchored onto one
    line, :func:`_collapse_after_reanchor` merges them and one is deleted.

    So try EVERY substantive line of the excerpt as a key, not just the first, and
    prefer a key that pins exactly ONE location — a signature or call line
    identifies a site, a bare annotation does not. Only if no key is unique do we
    fall back to the fewest-match key and the hint.
    """
    if not excerpt or not excerpt.strip() or not content:
        return 0
    norm_lines = [
        WHITESPACE_RUN.sub(" ", ln.strip().lower()) for ln in content.splitlines()
    ]
    # Match keys in priority order: the whole normalized excerpt first (an exact
    # single-line quote should always win), then each substantive excerpt line.
    keys = [WHITESPACE_RUN.sub(" ", excerpt.strip().lower())]
    for raw in excerpt.splitlines():
        stripped = raw.strip()
        if stripped and not _is_structural_line(stripped):
            keys.append(WHITESPACE_RUN.sub(" ", stripped.lower()))
    fewest: List[int] = []
    for key in keys:
        matches = [i + 1 for i, nl in enumerate(norm_lines) if key in nl]
        if not matches:
            continue
        if len(matches) == 1:
            return matches[0]
        if not fewest or len(matches) < len(fewest):
            fewest = matches
    if fewest:
        if hint:
            return min(fewest, key=lambda m: abs(m - hint))
        return fewest[0]
    return 0


# --------------------------------------------------------------------------- #
# ANCHOR VERIFICATION
#
# A finding's LINE is the coordinate everything downstream binds on: a scorer
# matches a finding to a known vulnerability by file + class + line window, and
# a reviewer reads the line to find the code. It is also the field the
# reviewing model is worst at, and a mislocated finding counts as a miss and a
# false positive at once.
#
# Checking the finding's quoted code against the file yields one of three
# materially different outcomes:
#
#   1. the quote was found AT the claimed line          -> the anchor is located
#   2. the quote was found ELSEWHERE in the same file   -> the anchor was WRONG
#                                                          and has been corrected
#   3. there is no usable quote, or the quote is not in the file at all
#                                                       -> the anchor is a CLAIM
#                                                          and nothing checked it
#
# Case 3 is structural, not accidental: a detection pass whose output contract
# carries no excerpt produces anchors that cannot be checked. Collapsing case 3
# into case 1 hides a real defect behind a bad coordinate; dropping case 3
# would discard a correct security finding. So the state is recorded per
# finding and disclosed in the report.
#
# WHAT THIS IS NOT. This makes mislocation DETECTABLE and, where the quote
# locates it, correctable. It says nothing about whether a finding is right,
# and a corrected line is not an accuracy figure. A line that matches the
# quote may still differ from the anchoring convention a scorer uses (defect
# statement versus enclosing declaration); that is a binding-side question and
# is not addressed here.
# --------------------------------------------------------------------------- #

ANCHOR_VERIFIED = "verified"

ANCHOR_CORRECTED = "corrected"

ANCHOR_UNVERIFIED = "unverified"

#: Keys the annotation is carried on, in the finding dict and in the report's
#: ``vulnerability.details`` named-list.
ANCHOR_DETAIL_KEY = "bl_anchor_status"

ANCHOR_REASON_DETAIL_KEY = "bl_anchor_unverified_reason"

ANCHOR_CLAIMED_DETAIL_KEY = "bl_anchor_claimed_line"

ANCHOR_CLAIMED_FILE_DETAIL_KEY = "bl_anchor_claimed_file"

#: Every anchor disclosure in ``scan.messages`` starts with this, so a reader
#: (and the coverage tests) can tell it apart from the coverage sentences.
ANCHOR_MESSAGE_PREFIX = "Anchor verification:"

_ANCHOR_NO_FILE = "the finding carries no file path, so nothing could be checked"

_ANCHOR_NO_EXCERPT = (
    "the pass that produced this finding emitted no code excerpt, so its line "
    "could not be checked against the file"
)

_ANCHOR_UNREADABLE = "the file could not be read back, so the line was not checked"

_ANCHOR_EXCLUDED = "Duo is not allowed to read this file, so the line was not checked"

_ANCHOR_NOT_PRESENT = (
    "the quoted code does not appear anywhere in this file, so the reported line "
    "is the model's claim and nothing confirms it"
)

_ANCHOR_RAISED = "anchor verification raised while checking this finding"

_ANCHOR_NO_CLAIM = (
    "the finding carried no line number; the anchor was located from its quote"
)

_ANCHOR_NOT_RUN = "anchor verification did not run for this report"


def _mark_anchor(
    finding: dict,
    status: str,
    reason: str = "",
    claimed_line: Optional[int] = None,
) -> None:
    """Record one finding's anchor outcome ON the finding."""
    finding["anchor_status"] = status
    if reason:
        finding["anchor_reason"] = reason
    # Only when the model actually stated a line. `line_of` returns 0 for a
    # finding that carried none, and writing `anchor_claimed_line: 0` would
    # assert a claim of line 0 that was never made.
    if claimed_line:
        finding["anchor_claimed_line"] = claimed_line


def _anchor_status_of(finding: dict) -> str:
    """The recorded outcome, defaulting to UNVERIFIED.

    An unannotated finding is one verification never saw - ``_verify_anchors``
    is best-effort and ``_execute`` swallows its failure - and "not checked"
    must not read as "checked and fine".
    """
    status = str(finding.get("anchor_status") or "").strip().lower()
    return status if status in _ANCHOR_STATES else ANCHOR_UNVERIFIED


_ANCHOR_STATES = (ANCHOR_VERIFIED, ANCHOR_CORRECTED, ANCHOR_UNVERIFIED)


def _text_detail(name: str, value: Any) -> dict:
    """One ``vulnerability.details`` entry.

    Satisfies the SAST schema's ``named_field`` + ``text`` detail type: a
    non-empty ``name``, ``type: "text"``, and a string ``value``.
    """
    return {"name": name, "type": "text", "value": str(value)}


def _anchor_details(finding: dict) -> dict:
    """The anchor state as schema-valid ``vulnerability.details`` entries.

    Emitted for EVERY vulnerability, including verified ones. Emitting only the bad cases would make absence mean
    "fine".
    """
    status = _anchor_status_of(finding)
    details = {
        ANCHOR_DETAIL_KEY: _text_detail("Anchor verification", status),
    }
    if status == ANCHOR_UNVERIFIED:
        details[ANCHOR_REASON_DETAIL_KEY] = _text_detail(
            "Why the anchor is unverified",
            finding.get("anchor_reason") or _ANCHOR_NOT_RUN,
        )
    claimed = finding.get("anchor_claimed_line")
    if status == ANCHOR_CORRECTED and claimed:
        details[ANCHOR_CLAIMED_DETAIL_KEY] = _text_detail(
            "Line originally claimed by the reviewer", claimed
        )
    if finding.get("claimed_file"):
        details[ANCHOR_CLAIMED_FILE_DETAIL_KEY] = _text_detail(
            "Path originally claimed by the reviewer", finding["claimed_file"]
        )
    return details


def _anchor_message(findings: List[dict]) -> dict:
    """The scan-level ``scan.messages`` entry for the run's anchor outcomes.

    Per-vulnerability detail answers "is THIS line trustworthy"; this answers "how much of this report is located and
    how much is claimed", so a wholly mis-anchored run cannot read as a clean one.
    """
    counts = dict.fromkeys(_ANCHOR_STATES, 0)
    for f in findings:
        counts[_anchor_status_of(f)] += 1
    total = len(findings)
    body = (
        f"{ANCHOR_MESSAGE_PREFIX} of {total} findings, "
        f"{counts[ANCHOR_VERIFIED]} verified (the quoted code is at the reported "
        f"line), {counts[ANCHOR_CORRECTED]} corrected (moved to where the quoted "
        f"code actually is), {counts[ANCHOR_UNVERIFIED]} could not be verified."
    )
    if counts[ANCHOR_UNVERIFIED] or counts[ANCHOR_CORRECTED]:
        return {
            "level": "warn",
            "value": (
                f"{body} An unverified line is the reviewer's CLAIM, not a "
                f"located position: the finding may still be real while "
                f"pointing at the wrong function. Do not read it as confirmed."
            ),
        }
    return {"level": "info", "value": body}


#: Keys the triage decision is carried on in the report's
#: ``vulnerability.details`` named-list. Prefixed like the anchor keys so the
#: analyzer's own annotations are distinguishable from any ingesting side's.
TRIAGE_CLAUSE_DETAIL_KEY = "bl_triage_clause"

TRIAGE_VERDICT_DETAIL_KEY = "bl_triage_verdict"

#: Emitted when a reported finding carries no triage annotation at all. Absence
#: of the key would read as "adjudicated and fine".
_TRIAGE_NOT_RECORDED = (
    "no triage annotation on this finding: either it never passed through the "
    "triage stage, or the adjudicator returned it without a verdict"
)

#: Emitted when the adjudicator gave a verdict but named no clause. The verdict
#: is real and the reason is missing -- a different fact from either of the
#: cases above, and it must not borrow their wording.
_TRIAGE_CLAUSE_UNSPECIFIED = (
    "the adjudicator returned a verdict but named no clause, so which criterion "
    "it applied is unrecorded"
)


def _triage_details(finding: dict) -> dict:
    """The triage decision as schema-valid ``vulnerability.details`` entries.

    Emitted for EVERY vulnerability, annotated or not, for the same reason
    :func:`_anchor_details` is: a key that appears only in the interesting cases
    makes its absence mean "fine".

    Every finding that reaches here is a survivor -- ``_record_triage_verdicts``
    has already removed the DROPs -- so the verdict is KEEP or nothing. It is
    still published, because "adjudicated KEEP under KEEP-incorrect-guard" and
    "never adjudicated" are different claims about the same reported line.
    """
    verdict = verdict_of(finding)
    clause = audit_clause_of(finding)[:TRIAGE_CLAUSE_MAX]
    if not verdict and not clause:
        return {
            TRIAGE_CLAUSE_DETAIL_KEY: _text_detail(
                "Triage clause", _TRIAGE_NOT_RECORDED
            )
        }
    details = {
        TRIAGE_CLAUSE_DETAIL_KEY: _text_detail(
            "Triage clause", clause or _TRIAGE_CLAUSE_UNSPECIFIED
        ),
    }
    if verdict:
        details[TRIAGE_VERDICT_DETAIL_KEY] = _text_detail("Triage verdict", verdict)
    return details


# --------------------------------------------------------------------------- #
# SCAN COVERAGE DISCLOSURE
#
# Caps truncate a scan: discovery may dispatch only some of the review units,
# and triage only some of the candidate findings. Those figures must travel
# with `gl-sast-report.json`, the artifact a reader is actually handed, or a
# capped scan reads as a complete one.
#
# WHERE IT GOES. The GitLab SAST report schema this tool emits (pinned in
# `_REPORT_SCHEMA_VERSION`) provides a purpose-built channel for exactly
# this: `scan.messages`, an array of `{level, value}` objects where `level` is
# one of `info` / `warn` / `fatal` and `value` is a non-empty string. The
# schema describes the property as "Communication intended for the initiator of
# a scan." It is the least invasive VALID location — it needs no new top-level
# property, no field the schema does not define, and no version bump beyond the
# 15.2.2 that `partial_scan` already requires.
#
# WHAT IT SAYS. The numbers are not re-derived here: each fan-out stage builds
# its coverage sentences once and publishes them to `context:<stage>.coverage`,
# which is passed to this tool. This layer only decides the level: what a stage
# covered is `info`, a cap that actually bound is `warn`. When no stage
# reported at all, the report says THAT rather than stay silent and read as
# complete.
# --------------------------------------------------------------------------- #

_COVERAGE_UNAVAILABLE = (
    "Scan coverage was not reported for this run: how many files were reviewed, "
    "and how many were discovered but never opened, is unknown. This report must "
    "not be read as covering the whole repository."
)


def _coverage_record(raw: Any) -> Optional[dict]:
    """One stage's coverage record, tolerant of a JSON-string context value."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def _coverage_messages(stages: List[Any]) -> List[dict]:
    """``scan.messages`` entries disclosing what each stage actually covered.

    One ``info`` entry per stage saying how much of what it was given it
    reviewed, a ``warn`` entry whenever that stage's cap actually bound, and a
    ``warn`` entry whenever that stage DISCARDED units whose findings it could
    not read.
    """
    messages: List[dict] = []
    for raw in stages:
        record = _coverage_record(raw)
        if record is None:
            continue
        summary = str(record.get("summary") or "").strip()
        if summary:
            messages.append({"level": "info", "value": summary})
        truncation = str(record.get("truncation") or "").strip()
        if truncation:
            messages.append({"level": "warn", "value": truncation})
        # A cap that bound is a CHOICE; findings the pipeline produced and then
        # threw away because it could not read them is a DEFECT, and a report
        # must not let the two read alike by omitting the second.
        loss = str(record.get("loss") or "").strip()
        if loss:
            messages.append({"level": "warn", "value": loss})
    return messages or [{"level": "warn", "value": _COVERAGE_UNAVAILABLE}]


class BlWriteSastReportInput(BaseModel):
    findings: Any = Field(
        description="Deduped findings (context:<dedup>.tool_responses): a flat JSON "
        "array (or JSON string) of finding dicts."
    )
    target_files: Optional[Any] = Field(
        default=None,
        description="The scoped-scan file list the flow gave discovery (on MR pipelines, "
        "the MR's changed files). When it names any file, only those files were "
        "reviewed and the report sets `scan.partial_scan` (mode `differential`), "
        "like diff-based GitLab Advanced SAST, so GitLab does not resolve "
        "vulnerabilities in files the scan never looked at.",
    )
    review_coverage: Optional[Any] = Field(
        default=None,
        description="Coverage record of the primary review fan-out "
        "(context:<map>.coverage), disclosed in the report's scan.messages.",
    )
    sibling_coverage: Optional[Any] = Field(
        default=None,
        description="Coverage record of the sibling-asymmetry fan-out "
        "(context:<map>.coverage), disclosed in the report's scan.messages.",
    )
    triage_coverage: Optional[Any] = Field(
        default=None,
        description="Coverage record of the triage fan-out over findings "
        "(context:<map>.coverage), disclosed in the report's scan.messages.",
    )


# Path-repair lookups per report; each is one findFiles call.
_MAX_PATH_LOOKUPS = 20


class BlWriteSastReport(DuoBaseTool):
    name: str = "bl_write_sast_report"
    description: str = dedent(
        """Deterministically build a schema-valid GitLab SAST report from deduped
        findings and write it to gl-sast-report.json in the workspace. No LLM."""
    )
    args_schema: Type[BaseModel] = BlWriteSastReportInput
    # Below 1.0.0 so ListTools does not publish it: it only runs inside
    # the BL security flow.
    tool_version: ClassVar[Version] = Version("0.1.0")
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    _PINNED = "gl-sast-report.json"

    def format_display_message(
        self, args: Any, _tool_response: Any = None
    ) -> Optional[str]:
        # See BlDedupFindings.format_display_message: the default dumps
        # `findings` (the full deduped array, potentially large) into the
        # chat-log `content` field uncapped.
        findings = getattr(args, "findings", None)
        count = len(findings) if isinstance(findings, list) else "?"
        return f"Writing SAST report for {count} findings"

    async def _verify_anchors(
        self, findings: List[dict], cache: Optional[dict] = None
    ) -> dict:
        """Check every finding's line against the file, and RECORD WHICH of the three outcomes it was.

        The finding's ``code_excerpt`` is expected to be a verbatim quote of the vulnerable statement. This locates that
        quote in the file and compares it with the line the finding claims:

        * quote at the claimed line -> ``verified``, emitted as-is.
        * quote elsewhere in the same file -> ``corrected``: the line is MOVED to where the quote really is, and the
            line originally claimed is kept alongside it, because a correction that erases what it corrected cannot
            be audited.
        * anything else (no quote, unreadable file, quote absent from the file) -> ``unverified``. The finding is
            KEPT with its claimed line - a sound security claim must survive a bad coordinate - but the report says
            the coordinate is a claim.

        Reads each unique file ONCE and IN FULL, via ``_read_file_fully`` -- a bare ``runReadFile`` would hand this
        check the first page only and it would relocate quotes onto lookalikes in that prefix. Best-effort: it MUST
        NEVER fail report generation, so every step is defensively guarded - but a guard that fires yields
        ``unverified`` rather than silence. Mutates findings in place; returns the per-outcome counts. The file contents
        it read are left in ``cache``.
        """
        cache = {} if cache is None else cache
        counts = dict.fromkeys(_ANCHOR_STATES, 0)

        def _record(
            finding: dict, status: str, reason: str = "", claimed: Optional[int] = None
        ) -> None:
            _mark_anchor(finding, status, reason, claimed)
            counts[status] += 1

        listed: dict = {}
        policy = FileExclusionPolicy(self.project)

        def _allowed(path: str) -> bool:
            # The file comes from model output: apply the project's rules AND the
            # always-on denylist and traversal guard, as ReadFile does.
            try:
                validate_duo_context_exclusions(path)
            except ToolException:
                return False
            return policy.is_allowed(path)

        async def _read(path: str) -> Optional[str]:
            if path not in cache:
                try:
                    # MUST follow read pagination: against a prefix, every
                    # quote below the cut is either a false "not present" or
                    # is matched to a lookalike inside the prefix and MOVED.
                    resp = await _read_file_fully(
                        self.metadata or {}, path, execute=_execute_action
                    )
                    cache[path] = resp if isinstance(resp, str) else None
                except Exception as exc_err:
                    _log.info("bl_anchor read failed", file=path, err=str(exc_err))
                    cache[path] = None
            return cache.get(path)

        for f in findings:
            try:
                exc = excerpt_of(f)
                file = f.get("file") or f.get("path") or ""
                if not file:
                    _record(f, ANCHOR_UNVERIFIED, _ANCHOR_NO_FILE)
                    continue
                if not exc.strip():
                    # No quote was ever emitted, so this anchor is uncheckable
                    # by construction, not merely unchecked.
                    _record(f, ANCHOR_UNVERIFIED, _ANCHOR_NO_EXCERPT)
                    continue
                if not _allowed(file):
                    _record(f, ANCHOR_UNVERIFIED, _ANCHOR_EXCLUDED)
                    continue
                content = await _read(file)
                repaired = "" if content else await self._repair_path(file, listed)
                if repaired and _allowed(repaired) and await _read(repaired):
                    f["claimed_file"], f["file"], file = file, repaired, repaired
                    content = cache[file]
                if not content:
                    _record(f, ANCHOR_UNVERIFIED, _ANCHOR_UNREADABLE)
                    continue
                claimed = line_of(f)
                ln = _find_excerpt_line(content, exc, hint=claimed)
                if not ln:
                    _record(f, ANCHOR_UNVERIFIED, _ANCHOR_NOT_PRESENT)
                elif ln == claimed:
                    _record(f, ANCHOR_VERIFIED)
                else:
                    f["new_line"] = ln
                    _record(
                        f,
                        ANCHOR_CORRECTED,
                        "" if claimed else _ANCHOR_NO_CLAIM,
                        claimed,
                    )
            except Exception as loop_err:
                _log.info("bl_anchor finding skipped", err=str(loop_err))
                _record(f, ANCHOR_UNVERIFIED, _ANCHOR_RAISED)
                continue
        return counts

    async def _repair_path(self, file: str, listed: dict) -> str:
        """The one repository path ending in ``/<file>``, or ``""``: a model sometimes drops a leading directory.

        Lists the basename once per report, for at most ``_MAX_PATH_LOOKUPS`` basenames, through ``findFiles``. No
        match, several matches, ``file`` itself listed or a truncated listing leaves the path alone.
        """
        base = file.rsplit("/", 1)[-1]
        if base not in listed and len(listed) < _MAX_PATH_LOOKUPS:
            try:
                resp = await _execute_action(
                    self.metadata or {},
                    contract_pb2.Action(
                        findFiles=contract_pb2.FindFiles(name_pattern=base)
                    ),
                )
            except Exception as err:
                _log.info("bl_anchor path lookup failed", file=file, err=str(err))
                resp = ""
            text = EXIT_CODE_HEADER.sub("", resp if isinstance(resp, str) else "", 1)
            # A byte cut can leave a fragment that still ends in "/<file>".
            lines = [] if executor_truncated(text) else text.splitlines()
            listed[base] = FileExclusionPolicy(self.project).filter_allowed(lines)[0]
        matches = [
            c for c in listed.get(base, ()) if c == file or c.endswith("/" + file)
        ]
        return matches[0] if len(matches) == 1 and matches[0] != file else ""

    def _build_report(
        self,
        findings: List[dict],
        coverage: Optional[List[Any]] = None,
        partial: bool = False,
        sources: Optional[dict] = None,
    ) -> dict:
        """Build the report; `partial` sets `scan.partial_scan` to `{"mode": "differential"}`.

        Only a `target_files` scan is partial, as with diff-based GitLab Advanced SAST. A capped run, or one that lost
        some units or findings, still ran over the whole repository and stays a full scan.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        identity = {
            "id": _ANALYZER_ID,
            "name": _ANALYZER_NAME,
            "vendor": {"name": _VENDOR},
            "version": _ANALYZER_VERSION,
        }
        # Findings that share (file, fingerprint) are N separate sites quoting
        # the same code — their IDs must differ or the ingesting side sees one vuln.
        # This key must stay identical to the id key below; see _shared_fingerprints.
        shared_fp = _shared_fingerprints(findings)
        signatures = _tracking_signatures(findings, sources or {})
        vulns = []
        for f, signature in zip(findings, signatures):
            cwe = cwe_digits(f)
            file = f.get("file") or f.get("path") or ""
            line = line_of(f)
            body = f.get("body") or f.get("description") or ""
            sev = _SEVERITY_MAP.get(str(f.get("severity", "")).lower(), "Unknown")
            # ID keyed on the code excerpt rather than the LLM-emitted line number,
            # which is often wrong. It still changes when the quote does, so GitLab
            # tracks the vulnerability by the scope_offset signature below.
            #
            # The CWE is left out of the id. GitLab tracks a vulnerability by its
            # location and first identifier (the CWE), not by this id, so a
            # finding re-labelled under a different CWE still reads there as a
            # new vulnerability. The CWE is reported in `identifiers` below.
            fingerprint = _fingerprint_of(f)
            vid_source = f"{file}|{fingerprint}"
            # ...but when several findings DO share a fingerprint they are distinct
            # sites, so fold the (already re-anchored) line in to tell them apart.
            # Singletons — the overwhelming majority — keep their line-free, churn-
            # free ID.
            if (file, fingerprint) in shared_fp:
                vid_source = f"{vid_source}|{line}"
            vid = hashlib.sha256(vid_source.encode("utf-8")).hexdigest()
            identifiers = []
            if cwe:
                identifiers.append(
                    {
                        "type": "cwe",
                        "name": f"CWE-{cwe}",
                        "value": cwe,
                        "url": f"https://cwe.mitre.org/data/definitions/{cwe}.html",
                    }
                )
                # Never before the CWE: identifiers[0] is the primary identifier,
                # and GitLab's vulnerability identity and tracking depend on it.
                owasp = owasp_identifier(cwe)
                if owasp:
                    identifiers.append(owasp)
            vuln = {
                "id": vid,
                "name": fallback_title(cwe, file),
                "description": markdown_description(
                    cwe=cwe,
                    file=file,
                    line=line,
                    excerpt=excerpt_of(f),
                    body=body,
                    impact=model_impact(f.get("impact")),
                ),
                "severity": sev,
                "location": {"file": file, "start_line": line},
                "identifiers": identifiers
                or [{"type": "bl_finding", "name": "BL finding", "value": vid[:16]}],
                # Whether `location.start_line` was LOCATED or merely CLAIMED,
                # and WHICH triage clause let this finding survive. `details` is
                # the schema's named-list of typed fields; no new property is
                # invented and no schema version is bumped. Neither is folded
                # into `vid` above: disclosure must not change identity.
                "details": {**_anchor_details(f), **_triage_details(f)},
            }
            solution = SOLUTIONS.get(cwe)
            if solution:
                vuln["solution"] = solution
            # The schema's own home for "an unsanitized excerpt of the affected
            # source code" - the quote the check above was made against. Without
            # it the verdict is unauditable downstream: a reader cannot re-derive
            # `verified` from the report alone.
            excerpt = excerpt_of(f)
            if excerpt:
                vuln["raw_source_code_extract"] = excerpt
            # The model's own title, when it gave a usable one: it is written
            # for the customer. Otherwise the fixed title above stays.
            title = model_title(f.get("title"))
            if title:
                vuln["name"] = title
            # Only a scope_offset signature: GitLab matches on the highest-priority
            # algorithm present, and that ranks above hash and location.
            if signature:
                sig = {"algorithm": "scope_offset", "value": signature}
                item = {"file": file, "start_line": line, "end_line": line}
                item["signatures"] = [sig]
                vuln["tracking"] = {"type": "source", "items": [item]}
            vulns.append(vuln)
        scan = {
            "start_time": now,
            "end_time": now,
            "status": "success",
            "type": "sast",
            "analyzer": identity,
            "scanner": identity,
            # How much of the repository this run actually looked at, and
            # whether a cap discarded findings (see SCAN COVERAGE DISCLOSURE),
            # then how much of what it DID report is anchored at a located line
            # rather than a claimed one (see ANCHOR VERIFICATION).
            "messages": _coverage_messages(coverage or [])
            + [_anchor_message(findings)],
        }
        if partial:
            scan["partial_scan"] = {"mode": "differential"}
        return {
            "version": _REPORT_SCHEMA_VERSION,
            "scan": scan,
            "vulnerabilities": vulns,
        }

    async def _execute(
        self,
        findings: Any,
        review_coverage: Optional[Any] = None,
        sibling_coverage: Optional[Any] = None,
        triage_coverage: Optional[Any] = None,
        target_files: Optional[Any] = None,
    ) -> str:
        parsed = [
            f
            for f in coerce_to_list(findings_input(findings, "findings"))
            if isinstance(f, dict)
        ]
        sources: dict = {}
        try:
            anchors = await self._verify_anchors(parsed, sources)
        except Exception as e:  # pylint: disable=exception-swallowing-in-tool
            # The pass must never fail report generation - but a report built
            # from findings it never annotated is not a clean one, and
            # `_anchor_status_of` defaults them all to `unverified` rather than
            # letting the failure read as a pass.
            _log.info("bl_anchor pass failed; every anchor stays a claim", err=str(e))
            anchors = dict.fromkeys(_ANCHOR_STATES, 0)
        _log.info(
            "bl_write_sast_report input",
            findings_type=type(findings).__name__,
            parsed=len(parsed),
            anchors_verified=anchors[ANCHOR_VERIFIED],
            anchors_corrected=anchors[ANCHOR_CORRECTED],
            anchors_unverified=anchors[ANCHOR_UNVERIFIED],
        )
        # Dedup keyed on the lines the MODEL claimed; `_verify_anchors` has just
        # replaced them with the real ones. Re-reports of a single site only look
        # identical now, so this is the first (and last) point they can collapse.
        before_collapse = len(parsed)
        parsed = _collapse_after_reanchor(parsed, sources)
        _log.info(
            "bl_write_sast_report post_reanchor_collapse",
            before=before_collapse,
            after=len(parsed),
            collapsed=before_collapse - len(parsed),
        )
        # Pipeline order: primary review, sibling pass, then triage. Each
        # stage's coverage record is disclosed in scan.messages. Only a
        # target_files scan is partial; caps and losses leave it a full scan.
        coverage = [review_coverage, sibling_coverage, triage_coverage]
        partial = bool(resolve_target_files(target_files))
        report = self._build_report(parsed, coverage, partial=partial, sources=sources)
        contents = json.dumps(report, ensure_ascii=False, indent=2)

        # One file action, the same runWriteFile `create_file_with_contents`
        # issues: no command execution. It creates or truncates, so it is NOT
        # atomic; the contract has no rename action to make it so. An executor
        # error raises ToolException from _execute_action; an older executor
        # that reports it only in the response text is caught here.
        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runWriteFile=contract_pb2.WriteFile(
                    filepath=self._PINNED, contents=contents
                )
            ),
        )
        if str(result).lstrip().lower().startswith("error"):
            raise ToolException(f"Writing {self._PINNED} failed: {result}")
        return (
            f"Wrote {len(report['vulnerabilities'])} vulnerabilities to "
            f"{self._PINNED} (analyzer={_ANALYZER_ID})."
        )
