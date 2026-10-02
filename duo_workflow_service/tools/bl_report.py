"""Deterministic (non-LLM) BL-analyzer tool: dedup findings.

A pure-code step meant to run inside DeterministicStepComponent, whose node calls tool.ainvoke(args) and publishes the
return at context:<name>.tool_responses.
"""

import re
from textwrap import dedent
from typing import Any, ClassVar, List, Optional, Type

import structlog
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.bl_security.findings import (
    TRIAGE_CLAUSE_MAX,
    Finding,
    audit_clause_of,
    coerce_to_list,
    cwe_digits,
    dedup_key,
    findings_input,
    line_of,
    norm_excerpt,
    verdict_of,
)
from duo_workflow_service.security.secret_redaction import redact_secrets
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig

__all__ = ["BlDedupFindings"]

_log = structlog.stdlib.get_logger(__name__)

# Never truncate the findings blob passing between dedup -> write (default is
# 200 KiB; a stringified findings list can approach that).
_NO_TRUNCATION = TruncationConfig(
    max_bytes=64 * 1024 * 1024, truncated_size=64 * 1024 * 1024
)

# --------------------------------------------------------------------------- #
# SINGLE SOURCE OF TRUTH for the BL analyzer's vulnerability scope.
#
# The boundary is the detection MECHANISM, not a curated CWE list: this analyzer
# owns the classes that can only be found by reasoning about what the code is
# SUPPOSED to do (authorization, state/interleaving, business workflow). Classes
# expressible as a syntactic pattern or a source->sink taint flow - SSRF
# (CWE-918), insecure deserialization (CWE-502), SQLi, XSS, path traversal, weak
# crypto (CWE-327) - belong to SAST/GLAS and are NOT in scope here.
#
# CWE-502 is out for that mechanism reason: untrusted bytes (the source) reach a
# decoder (the sink), which a taint trace finds without understanding what the
# code intends.
#
# CWE-269 is absent because privilege escalation is not a DETECTOR class: it is
# the IMPACT of an authorization defect. The reviewer hunts the defect - the
# missing, incorrect, or ownership-less check - and reports it as 862, 863, or
# 639; "this leaves the caller with more privilege than intended" is the
# severity argument carried ON that finding, never a finding of its own.
IN_SCOPE_CWES = frozenset(
    {
        "200",  # sensitive data exposure (qualified: authz-at-data-layer variant)
        "284",  # improper access control
        "285",  # improper authorization
        "287",  # improper authentication
        "362",  # race condition
        "367",  # TOCTOU
        "459",  # incomplete cleanup (qualified: leaves an authz/access residue)
        "639",  # IDOR / BOLA
        "840",  # business logic errors / incorrect state validation
        "862",  # missing authorization
        "863",  # incorrect authorization
        "915",  # mass assignment (qualified: permitted-but-sensitive-field residual)
    }
)

# A SECOND, ORTHOGONAL axis - deliberately NOT folded into IN_SCOPE_CWES.
# The allowlist answers "does this CLASS belong to another analyzer?"; this
# regex answers "is this noise that arrived wearing an in-scope CWE?" - the
# reviewer routinely tags a rate-limiting, DoS, CORS or panic report as
# CWE-862 or CWE-284, so no CWE list can express it. Collapsing the two would
# mean inventing CWE codes for noise shapes, not removing a duplicate scope
# definition. It names no vulnerability class at all, so it cannot disagree
# with the allowlist.
_DROP_KEYWORDS = re.compile(
    r"rate.?limit|brute.?force|denial.of.service|\bdos\b|"
    r"panic|unwrap|cleartext|cors|secure.?flag|header.?trust",
    re.IGNORECASE,
)


def _is_out_of_scope_cwe(cwe: str) -> bool:
    """Return True when ``cwe`` names a class outside :data:`IN_SCOPE_CWES`.

    Deliberately conservative: a finding whose CWE is missing or unparsable is
    never reported as out of scope, so a finding is dropped only when its class
    is positively known to belong to another analyzer.
    """
    return bool(cwe) and cwe not in IN_SCOPE_CWES


def _dominant_cwe_per_statement(
    findings: List[Finding],
) -> dict[tuple[str, str], str]:
    """Map each (file, normalized excerpt) group to the CWE the most findings in it carry.

    ``max`` returns the first maximal key and dicts preserve insertion order, so a tie resolves to the first-seen
    (highest-ranked) CWE.
    """
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for f in findings:
        exc = norm_excerpt(f)
        cwe = cwe_digits(f)
        if not exc or not cwe:
            continue
        group = (f.get("file") or f.get("path") or "", exc)
        per_cwe = counts.setdefault(group, {})
        per_cwe[cwe] = per_cwe.get(cwe, 0) + 1
    return {
        group: max(per_cwe, key=per_cwe.__getitem__)
        for group, per_cwe in counts.items()
    }


def _second_pass_collapse(findings: List[Finding]) -> List[Finding]:
    """After the (cwe,file,line//4,excerpt) keep-first pass, collapse residual facet explosion where the reviewer re-
    reported the SAME source statement under DIFFERENT CWEs.

    Key = (file, normalized code excerpt) restricted to a CROSS-CWE match — usually the same defect wearing two labels.
    NOT risk-free: the key has no line, so two genuinely distinct sites that quote byte-identical code AND were tagged
    with different CWEs merge, and the minority-CWE site is dropped. Body-similarity is NOT used: facets of one logical
    vuln spanning handler+finder+schema sit at different real lines with dissimilar bodies, so any similarity threshold
    low enough to catch them would also merge distinct findings. Semantic facet consolidation is the reviewer's job.

    SAME-CWE repeats of one statement are NOT collapsed here. A file can hold N separate sites — N resolvers/endpoints
    on a short stride — that miss the SAME control and so carry the SAME CWE and near-identical (sometimes byte-
    identical) excerpts. Each is independently exploitable, so each is its own finding: fixing N-1 of them leaves the
    app vulnerable. Same-CWE true duplicates of ONE site are already collapsed upstream by the exact pass, which anchors
    on the line bucket. Findings without an excerpt or a CWE are never collapsed here, and never vote.

    Which CWE survives a group is decided by FREQUENCY, not arrival order. Keep-first would let a single stray re-label
    that happened to sort first evict every one of the N real sites behind it. Ties fall back to first-seen, which is
    the ranking order.
    """
    dominant = _dominant_cwe_per_statement(findings)
    kept: List[Finding] = []
    for f in findings:
        exc = norm_excerpt(f)
        if not exc:
            kept.append(f)
            continue
        group = (f.get("file") or f.get("path") or "", exc)
        cwe = cwe_digits(f)
        if cwe and cwe != dominant[group]:
            continue
        kept.append(f)
    return kept


def _flatten_findings(batches: Any) -> List[Finding]:
    """Flatten a JSON array (or string) whose elements are per-batch arrays of finding dicts (each possibly a JSON
    string)."""
    out: List[Finding] = []
    for element in coerce_to_list(batches):
        for item in coerce_to_list(element):
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                for sub in coerce_to_list(item):
                    if isinstance(sub, dict):
                        out.append(sub)
    return out


# --------------------------------------------------------------------------- #
# TRIAGE AUDIT
#
# A triage stage fans every deduped candidate out over a separate adjudicator
# run. The adjudicator annotates each finding with ``verdict`` / ``clause`` /
# ``evidence`` and returns it for BOTH verdicts, so a dropped finding leaves a
# trace: every decision is logged structurally here, then the DROPs are removed.
# Without that, "did the gate delete a real detection?" cannot be answered from
# the artifacts.
#
# WHERE THE VERDICT LIVES. A DROPped finding has no report entry to carry one
# on, so the service log is the only sink for the DROPs. The SURVIVORS carry
# their clause and verdict in ``vulnerability.details`` (see
# ``_triage_details`` in ``bl_write_sast_report``), the schema's own
# named-list of typed fields and the same channel the anchor state uses: no
# invented property and no schema-version bump. That makes "did this KEEP arm
# ever fire?" a grep over the report. Scan-LEVEL disclosure has its own home;
# see the SCAN COVERAGE DISCLOSURE block in ``bl_write_sast_report``.
#
# Two properties keep this instrumentation-only:
#   * a findings list where NOTHING carries a verdict is returned unchanged, so
#     a pre-triage call of this same tool is untouched;
#   * only an explicit verdict of ``DROP`` removes a finding. A missing, empty,
#     or unrecognised verdict KEEPS it.
# --------------------------------------------------------------------------- #


def _audit_evidence_of(finding: Finding) -> str:
    """The code fact justifying the verdict.

    Prefers ``evidence`` and falls back to ``triage_evidence``, which the
    adjudicator may set on KEEP only.
    """
    return str(finding.get("evidence") or finding.get("triage_evidence") or "").strip()


def _log_safe(text: str, limit: int) -> str:
    """Redact secrets from model-written text, then bound it for the audit log."""
    return str(redact_secrets(text, "bl_dedup_findings"))[:limit]


def _record_triage_verdicts(findings: List[Finding]) -> List[Finding]:
    """Log every adjudication decision, then drop the ones the adjudicator DROPped.

    Returns ``findings`` unchanged (same list object) when no finding carries a
    verdict, so this is a no-op on every non-triage caller.
    """
    adjudicated = sum(1 for f in findings if verdict_of(f))
    if not adjudicated:
        return findings

    kept: List[Finding] = []
    dropped = 0
    for f in findings:
        verdict = verdict_of(f)
        if verdict:
            # One structured record per decision: this IS the audit trail. Bodies
            # and evidence are bounded so a 500-finding scan cannot flood the log.
            _log.info(
                "bl_triage_verdict",
                verdict=verdict,
                clause=_log_safe(audit_clause_of(f), TRIAGE_CLAUSE_MAX)
                or "UNSPECIFIED",
                evidence=_log_safe(_audit_evidence_of(f), 600) or "UNSPECIFIED",
                file=f.get("file") or f.get("path") or "",
                line=line_of(f),
                cwe=cwe_digits(f),
                severity=str(f.get("severity", "")),
                body=_log_safe(str(f.get("body") or f.get("description") or ""), 240),
            )
        if verdict == "DROP":
            dropped += 1
            continue
        kept.append(f)

    _log.info(
        "bl_triage_verdicts summary",
        candidates=len(findings),
        adjudicated=adjudicated,
        # Findings the adjudicator returned with no verdict at all. Kept — a
        # non-zero count means the model is ignoring the output contract.
        unlabelled=len(findings) - adjudicated,
        dropped=dropped,
        kept=len(kept),
    )
    return kept


class BlDedupFindingsInput(BaseModel):
    batches: Any = Field(
        description="Findings from a detection or triage stage: a JSON array (or JSON "
        "string) of finding dicts, or of per-batch arrays of them."
    )
    batches2: Any = Field(
        default=None,
        description="Optional SECOND findings source MERGED with `batches` before "
        "dedup, e.g. a second detection pass. Flattened independently, "
        "then concatenated. None for single-source callers.",
    )
    drop_out_of_scope: bool = Field(
        default=False,
        description="If true, drop findings whose CWE is outside IN_SCOPE_CWES "
        "(taint/pattern classes SAST owns, e.g. SSRF/crypto) plus DoS/config "
        "keyword matches. Off by default; dedup-collapse applies either way.",
    )


class BlDedupFindings(DuoBaseTool):
    name: str = "bl_dedup_findings"
    description: str = dedent(
        """Deterministically flatten, deduplicate, and scope-filter business-logic
        security findings. No LLM and no I/O. Dedup key =
        (normalized CWE digits, file, line//4, normalized code excerpt),
        keep-first, then a cross-CWE collapse of findings quoting the same code
        in one file. Distinct sites are usually kept, but two sites quoting
        identical code merge if they share a 4-line bucket and CWE, or carry
        different CWEs anywhere in the file.
        When findings carry a triage `verdict`, every verdict + clause + evidence
        is logged structurally first and the DROPs are then removed."""
    )
    args_schema: Type[BaseModel] = BlDedupFindingsInput
    # Below 1.0.0 so ListTools does not publish it: it only runs inside
    # the BL security flow.
    tool_version: ClassVar[Version] = Version("0.1.0")
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    truncation_config: TruncationConfig = _NO_TRUNCATION

    def format_display_message(
        self, _args: Any, tool_response: Any = None
    ) -> Optional[str]:
        # DuoBaseTool's default dumps every arg's str() into the chat-log
        # `content` field with NO size cap. `batches`/`batches2` here can be
        # the full inline findings blob, and stringifying it verbatim can push
        # a single checkpoint delta over the gRPC transport cap.
        count = len(tool_response) if isinstance(tool_response, list) else "?"
        return f"Deduplicated to {count} findings"

    async def _execute(
        self, batches: Any, batches2: Any = None, drop_out_of_scope: bool = False
    ) -> list[Finding]:
        # Both sources are flattened independently, then concatenated, before dedup.
        findings = _flatten_findings(findings_input(batches, "batches"))
        if batches2 is not None:
            findings = findings + _flatten_findings(
                findings_input(batches2, "batches2")
            )
        _log.info(
            "bl_dedup_findings input",
            flattened=len(findings),
            drop_out_of_scope=drop_out_of_scope,
        )
        # Audit + apply the triage verdicts BEFORE dedup, so dedup never sees a
        # DROPped finding. No-op unless a finding carries a verdict.
        findings = _record_triage_verdicts(findings)
        kept: List[Finding] = []
        seen: set[tuple[str, str, int, str]] = set()
        for f in findings:
            cwe = cwe_digits(f)
            if drop_out_of_scope:
                if _is_out_of_scope_cwe(cwe):
                    continue
                if _DROP_KEYWORDS.search(str(f.get("body", ""))):
                    continue
            key = dedup_key(f, cwe=cwe)
            if key in seen:
                continue
            seen.add(key)
            kept.append(f)
        after_exact = len(kept)
        # Second pass: collapse residual facet explosion (one statement re-reported
        # under a DIFFERENT CWE) within each file. Same-CWE repeats are distinct
        # sites and are deliberately kept.
        kept = _second_pass_collapse(kept)
        _log.info(
            "bl_dedup_findings output",
            kept=len(kept),
            after_exact_dedup=after_exact,
            collapsed_facets=after_exact - len(kept),
        )
        return kept
