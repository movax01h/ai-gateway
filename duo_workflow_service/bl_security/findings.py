"""Pure helpers that read BL findings, shared by the dedup and report-writer tools.

No I/O and no tool classes: they read fields off finding dicts and parse step outputs. Tools import from here, and
nothing here imports from ``duo_workflow_service.tools``.
"""

import json
import re
from typing import Any

from langchain_core.tools import ToolException

from duo_workflow_service.security.secret_redaction import redact_secrets

# A model-written finding, validated at the boundary.
Finding = dict[str, Any]


def cwe_digits(finding: Finding) -> str:
    """The digits of the finding's CWE (``"CWE-639"`` -> ``"639"``); ``""`` when it has none."""
    raw = str(finding.get("cwe") or finding.get("CWE") or "")
    m = re.search(r"(\d+)", raw)
    return m.group(1) if m else ""


def line_of(finding: Finding) -> int:
    """The finding's line, from the first line key holding an integer; ``0`` when none does."""
    for k in ("new_line", "line", "start_line", "lineNumber"):
        v = finding.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return 0


#: A run of whitespace. Collapsing it to one space is the normalization every
#: excerpt comparison uses, so a caller matching quotes against file lines uses it too.
WHITESPACE_RUN = re.compile(r"\s+")


def excerpt_of(finding: Finding) -> str:
    """The finding's quoted code; ``""`` when it carries none."""
    return str(finding.get("code_excerpt") or finding.get("excerpt") or "")


def norm_excerpt(finding: Finding) -> str:
    """Whitespace/case-normalized code excerpt — the strongest same-defect key:

    two findings that cite the SAME source statement are the same defect no matter what CWE/line the LLM tagged (this is
    what facet-explosion looks like).
    """
    return WHITESPACE_RUN.sub(" ", excerpt_of(finding).strip().lower())


def dedup_key(f: Finding, cwe: str | None = None) -> tuple[str, str, int, str]:
    """The exact-dedup identity of a finding: ``(cwe, file, line bucket, normalized excerpt)``.

    The line bucket absorbs the LLM's +/-few-line anchor jitter on ONE site; the excerpt keeps two DIFFERENT statements
    that land in the same bucket apart. Adjacent sites on a short stride (N near-identical resolvers 2-3 lines apart)
    share a bucket but quote different code, so without the excerpt in the key all but one would be silently merged.

    ``cwe`` exists only so a caller that has already computed ``cwe_digits(f)`` need not recompute it; omitting it is
    always correct.
    """
    return (
        cwe if cwe is not None else cwe_digits(f),
        f.get("file") or f.get("path") or "",
        line_of(f) // 4,
        norm_excerpt(f),
    )


def json_or_none(text: str) -> Any:
    """``text`` decoded as strict JSON; ``None`` when it is not JSON.

    Every input here was written by an earlier step of this flow with ``json.dumps``, so text that does not decode
    is a defect to report, not prose to salvage a JSON value from.
    """
    try:
        return json.loads(text)
    except ValueError:
        return None


def coerce_to_list(raw: Any) -> list[Any]:
    """``raw`` as a list: JSON text is decoded first, a truthy non-list is wrapped, and anything else is ``[]``."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parsed = json_or_none(raw)
        raw = parsed if parsed is not None else []
    return raw if isinstance(raw, list) else ([raw] if raw else [])


def findings_input(raw: Any, arg: str) -> list[Any]:
    """``raw`` parsed to a list; raises when it is anything else.

    A failed flow step publishes ``None`` (or an error string) and the next step still runs. Read as
    zero findings, that would become a "success" report with no vulnerabilities.
    """
    parsed = raw
    if isinstance(raw, str):
        # Deterministic step output, not model prose: anything that does not
        # open as a JSON array/object (e.g. "Error: failed on item [0]") is not
        # a findings input.
        text = raw.strip()
        parsed = json_or_none(text) if text[:1] in ("[", "{") else None
    if not isinstance(parsed, list):
        # Exceptions skip DuoBaseTool's output redaction, so redact the echoed input here.
        snippet = str(redact_secrets(str(raw)[:2000], "bl_security"))[:200]
        raise ToolException(
            f"`{arg}` is not a findings list ({snippet!r}); an earlier "
            "step most likely failed. Refusing to read it as zero findings."
        )
    return parsed


def verdict_of(finding: Finding) -> str:
    """The adjudicator's verdict, normalized.

    ``""`` when the finding carries none.
    """
    return str(finding.get("verdict") or "").strip().upper()


def audit_clause_of(finding: Finding) -> str:
    """The criterion the adjudicator says it applied (a ``DROP-N`` or ``KEEP-*`` arm)."""
    return str(finding.get("clause") or "").strip()


#: Length bound on the model-authored clause string, in the report and in the
#: ``bl_triage_verdict`` log record. The clause is LLM output and a large report
#: must not be able to grow without limit through it.
TRIAGE_CLAUSE_MAX = 160
