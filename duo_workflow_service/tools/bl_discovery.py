"""Deterministic authorization-surface discovery for the BL analyzer: the ``bl_discover_and_cluster`` tool.

Runs inside Duo Workflow Service against a remote repo. Candidate paths come from the ``find_files`` executor action;
the default path reads no file bodies. Which files are reviewed, and how they are grouped, is decided by the pure
path logic in ``duo_workflow_service.bl_security.units``.
"""

import re
from textwrap import dedent
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import structlog
from packaging.version import Version
from pydantic import BaseModel, Field, field_validator

from contract import contract_pb2
from duo_workflow_service.bl_security import units as _units
from duo_workflow_service.bl_security.discovery_patterns import AUTHZ_GLOBS
from duo_workflow_service.bl_security.executor_output import (
    EXIT_CODE_HEADER,
    complete_lines,
    executor_truncated,
)
from duo_workflow_service.bl_security.target_files import resolve_target_files

# Some names are only re-exported: the flow-config tests read them through this
# module.
from duo_workflow_service.bl_security.units import (  # noqa: F401  # pylint: disable=unused-import
    MAX_FILES_PER_ANCHOR_PATTERN,
    SCAN_EFFORT_TIERS,
    cap_extra_anchor_files,
    cap_scoped_units,
    content_filter_candidates,
    enumerate_candidates,
    is_anchor,
    resolve_cap_aware_backfill,
    resolve_content_filter,
    resolve_coverage_first,
    resolve_extra_anchor_patterns,
    resolve_extra_globs,
    resolve_saturation_stop,
    resolve_scan_effort,
    select_units,
    target_file_units,
)
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig

__all__ = ["BlDiscoverAndCluster"]

_log = structlog.stdlib.get_logger("bl_discovery")


def _wire_pattern(pattern: str) -> str:
    """The ``find_files`` spelling of a glob.

    ``find_files`` matches a ``/``-free pattern against each basename and a pattern with ``/`` against the
    repo-relative path, so a directory glob is prefixed with ``**/``: it then matches at any depth (a project nested in
    a monorepo is found like a top-level one), while ``*`` stays inside one segment and ``**`` spans zero or more.
    """
    if "/" not in pattern or pattern.startswith("**/"):
        return pattern
    return "**/" + pattern


# ---------------------------------------------------------------------------
# CONTENT NET -- the only body read in this stage (off by default; the BL flow turns it on)
# ---------------------------------------------------------------------------
# No body is handed to the model: the executor runs one `rg -l` over the working
# tree and returns paths. The net is two term groups, unioned, matched
# whole-word and case-insensitively:
#   AUTHZ  -- how frameworks spell "may this caller do this".
#   LOOKUP -- how frameworks spell "fetch the object this request names", which
#             is where a missing ownership check sits.
# Both are framework vocabulary; no term names a file or symbol of any
# particular repository. A file matching neither makes no authorization
# decision and looks no object up.
_AUTHZ_TERMS: List[str] = (
    "authorize authorized authorization can allowed permission permissions policy ability "
    "current_user admin is_admin privilege forbidden unauthorized before_action preauthorize "
    "authenticate authenticated user owner role session token login account actor"
).split()

_LOOKUP_TERMS: List[str] = (
    "objects getobject get_object findbyid find_by_id findone findbypk getbyid get_by_id "
    "queryset filter pk primary_key lookup_field retrieve fetchone byid params request"
).split()

#: The union, in a stable order (AUTHZ first, then whatever LOOKUP adds).
CONTENT_NET_TERMS: List[str] = _AUTHZ_TERMS + [
    term for term in _LOOKUP_TERMS if term not in _AUTHZ_TERMS
]

#: The pattern handed to the executor's search program. Import it rather than
#: restating it.
CONTENT_NET_PATTERN: str = r"\b(" + "|".join(CONTENT_NET_TERMS) + r")\b"

#: The search program the executor runs. It honours the repository's ignore
#: rules. If it is missing, the content filter degrades to paths only.
_SEARCH_PROGRAM = "rg"

_LEADING_DOT_SLASH = re.compile(r"^\./")

# Later flow steps parse the unit list and hand each unit to an agent prompt.
# Display truncation (200 KiB) would turn it into a string on a large repository.
_NO_TRUNCATION = TruncationConfig(
    max_bytes=64 * 1024 * 1024, truncated_size=64 * 1024 * 1024
)


class BlDiscoverAndClusterInput(BaseModel):
    files_per_unit: int = Field(
        ge=1, description="Max files per review unit (an anchor + its context files)."
    )
    max_units: int = Field(
        ge=1,
        description="Unit budget for the context tier; the entry-point and business-logic tiers are always emitted in "
        "full. Caps a scoped scan.",
    )
    project_id: Optional[Any] = Field(
        default=None, description="GitLab project id. Logged only."
    )
    branch: Optional[Any] = Field(
        default=None,
        description="Branch ref. Logged only; find_files reads the checked-out working tree.",
    )
    scan_effort: Optional[str] = Field(
        default=None,
        description=(
            "Effort tier (low|standard|high); a known tier replaces files_per_unit/max_units and sets "
            "the context pool multiplier and merge weight (see SCAN_EFFORT_TIERS; 1/1 with no tier)."
        ),
    )
    coverage_first: Optional[Any] = Field(
        default=None,
        description="When true, a slot that would re-review a selected file goes to an unselected one.",
    )
    cap_aware_backfill: Optional[Any] = Field(
        default=None,
        description="When true (with coverage_first), backfilling skips anchors whose solo unit is already reviewed.",
    )
    saturation_stop: Optional[Any] = Field(
        default=None,
        description="When true, the list is cut after the last unit that adds a new file.",
    )
    content_filter: Optional[Any] = Field(
        default=None,
        description="When true, keep only candidates whose body matches the authorization or lookup vocabulary "
        "(anchors are always kept).",
    )
    extra_globs: Optional[Any] = Field(
        default=None,
        description="Additional path globs for this repo: a list, or an object whose `globs` key holds one. Added "
        "to the built-in globs, never substituted.",
    )
    extra_anchor_patterns: Optional[Any] = Field(
        default=None,
        description="Additional entry-point path regexes: a list, or an object whose `anchor_patterns` key holds "
        "one. Added to the built-in patterns.",
    )
    target_files: Optional[Any] = Field(
        default=None,
        description="Scoped-scan file list: repo-relative paths (a list, or one separated string). When non-empty, "
        "discovery is skipped and exactly these files are reviewed.",
    )

    @field_validator("files_per_unit", "max_units", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        return int(str(v).strip())


class BlDiscoverAndCluster(DuoBaseTool):
    name: str = "bl_discover_and_cluster"
    description: str = dedent(
        """Deterministically enumerate a repo's authorization surface by path
        (no LLM) and group the candidate paths into review units: a guaranteed
        unit per entry point, business-logic units, and entry-point clusters with
        their related files, interleaved so any prefix of the list samples all
        three. The same repo yields the same units every run, and a larger
        max_units includes more files. A larger files_per_unit makes units wider
        rather than nesting them, so compare it at equal file-slot cost.
        Supplying target_files skips discovery entirely and reviews exactly those
        paths (scoped scan)."""
    )
    args_schema: Type[BaseModel] = BlDiscoverAndClusterInput
    # Below 1.0.0 so ListTools does not publish it: it only runs inside
    # the BL security flow.
    tool_version: ClassVar[Version] = Version("0.1.0")
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    truncation_config: TruncationConfig = _NO_TRUNCATION
    handle_tool_error: bool = True

    async def _find(self, pattern: str, truncated: Optional[set] = None) -> List[str]:
        """List the files matching ``pattern``; a truncated listing is added to ``truncated``."""
        try:
            result = await _execute_action(
                self.metadata,  # type: ignore
                contract_pb2.Action(
                    findFiles=contract_pb2.FindFiles(name_pattern=pattern)
                ),
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.warning(
                "bl_discover find_files failed; candidates are missing",
                pattern=pattern,
                err=str(e),
            )
            return []
        if not isinstance(result, str) or not result.strip():
            return []
        lines = result.strip().splitlines()
        if executor_truncated(result.strip()):
            _log.info("bl_discover find_files output truncated", pattern=pattern)
            lines = complete_lines(result.strip())
            if truncated is not None:
                truncated.add(pattern)
        # Same exclusion policy as FindFiles, so an excluded path can never
        # reach a unit or the logs.
        allowed, _ = FileExclusionPolicy(self.project).filter_allowed(lines)
        return allowed

    async def _search(self, arguments: List[str]) -> Optional[str]:
        """Run the executor's search program once; ``None`` means no content signal.

        Never raises: the caller falls back to paths-only behaviour.
        """
        try:
            result = await _execute_action(
                self.metadata,  # type: ignore
                contract_pb2.Action(
                    runCommand=contract_pb2.RunCommandAction(
                        program=_SEARCH_PROGRAM, arguments=list(arguments), flags=[]
                    )
                ),
            )
        except Exception as e:  # pylint: disable=broad-except
            _log.info("bl_discover content search failed", err=str(e))
            return None
        if not isinstance(result, str) or not result.strip():
            return None
        header = EXIT_CODE_HEADER.match(result)
        if header:
            result = result[header.end() :]
            # The search program exits 1 when nothing matched, 2 on an error.
            if header.group(1) not in ("0", "1"):
                _log.info(
                    "bl_discover content search failed", exit_code=header.group(1)
                )
                return None
        if executor_truncated(result):
            # The executor dropped the middle of the output, so it is no
            # longer a complete list: treat it as no content signal.
            _log.info("bl_discover content search output truncated")
            return None
        return result if result.strip() else None

    async def _content_hits(self) -> Optional[set]:
        """The set of tree paths whose body is in the content net (one pass)."""
        out = await self._search(
            [
                "-l",
                "-i",
                "--text",
                "--no-messages",
                "-e",
                CONTENT_NET_PATTERN,
                ".",
            ]
        )
        if out is None:
            return None
        hits = set()
        for raw in out.splitlines():
            path = raw.strip()
            if not path:
                continue
            hits.add(_LEADING_DOT_SLASH.sub("", path))
        return hits or None

    async def _execute(
        self,
        files_per_unit: int,
        max_units: int,
        project_id: Optional[Any] = None,
        branch: Optional[Any] = None,
        scan_effort: Optional[str] = None,
        coverage_first: Optional[Any] = None,
        cap_aware_backfill: Optional[Any] = None,
        saturation_stop: Optional[Any] = None,
        content_filter: Optional[Any] = None,
        target_files: Optional[Any] = None,
        extra_globs: Optional[Any] = None,
        extra_anchor_patterns: Optional[Any] = None,
    ) -> List[dict]:
        files_per_unit, max_units, ctx_mult, ctx_weight = resolve_scan_effort(
            scan_effort, files_per_unit, max_units
        )

        # SCOPED SCAN: an explicit scope replaces discovery outright. `max_units`
        # still caps the result.
        targets = resolve_target_files(target_files)
        if targets:
            # A scope of only excluded paths yields no units, never a full scan.
            targets, _ = FileExclusionPolicy(self.project).filter_allowed(targets)
            selected = cap_scoped_units(
                target_file_units(targets, files_per_unit), max_units
            )
            _log.info(
                "bl_discover_and_cluster scoped_scan",
                project_id=project_id,
                branch=branch,
                target_files=len(targets),
                units_selected=len(selected),
                files_per_unit=int(files_per_unit),
                max_units=int(max_units),
                authz_sample_count=_units.AUTHZ_SAMPLE_COUNT,
                scan_effort=scan_effort,
            )
            # Same telemetry line as the full path: only files in selected units.
            reviewed = sorted({f for u in selected for f in u["files"]})
            _log.info(
                "bl_discover_reviewed_pool", n=len(reviewed), files="|".join(reviewed)
            )
            return selected

        # Additions are unioned onto the built-in net and can never remove from
        # it.
        add_globs = resolve_extra_globs(extra_globs)
        extra_anchors: Tuple[Any, ...] = tuple(
            resolve_extra_anchor_patterns(extra_anchor_patterns)
        )
        n_anchor_patterns = len(extra_anchors)

        per_glob: List[List[str]] = []
        found: Dict[str, List[str]] = {}
        truncated: set = set()
        incomplete: List[str] = []
        for pattern in AUTHZ_GLOBS + add_globs:
            key = _wire_pattern(pattern)
            if key not in found:
                found[key] = await self._find(key, truncated)
            if key in truncated:
                incomplete.append(pattern)
            per_glob.append(found[key])
        if extra_anchors:
            # Anchor patterns add the files they match, not only label glob
            # hits, so they need the whole tree.
            listed = await self._find("*", truncated)
            if "*" in truncated:
                incomplete.append("anchor_patterns")
            hits = {rx: [p for p in listed if rx.search(p)] for rx in extra_anchors}
            for rx, matches in hits.items():
                if len(matches) > MAX_FILES_PER_ANCHOR_PATTERN:
                    _log.warning(
                        "bl_discover anchor pattern matches too many files -- none added",
                        pattern=rx.pattern,
                        matched=len(matches),
                        cap=MAX_FILES_PER_ANCHOR_PATTERN,
                    )
            # A rejected pattern must not label glob hits as entry points either.
            accepted = [
                rx for rx, m in hits.items() if len(m) <= MAX_FILES_PER_ANCHOR_PATTERN
            ]
            extra_anchors = cap_extra_anchor_files(
                accepted, [p for rx in accepted for p in hits[rx]]
            )
            per_glob.extend(
                [p for p in hits[rx] if is_anchor(p, extra_anchors)] for rx in accepted
            )
            n_anchor_patterns = len(accepted)
        if incomplete:
            _log.warning(
                "bl_discover file listing truncated by the executor; candidates are missing",
                globs=incomplete,
            )

        candidates = enumerate_candidates(per_glob)

        # Content filter (off by default; the BL flow turns it on). If the search
        # program is unavailable the candidate set is left unchanged.
        want_filter = resolve_content_filter(content_filter)
        content_hits = await self._content_hits() if want_filter else None
        candidates_before = len(candidates)
        if content_hits:
            candidates = content_filter_candidates(
                candidates, content_hits, extra_anchors=extra_anchors
            )
        elif want_filter:
            _log.info(
                "bl_discover content_filter unavailable -- keeping every candidate",
                candidates=candidates_before,
            )

        cover_first = resolve_coverage_first(coverage_first)
        cap_aware = resolve_cap_aware_backfill(cap_aware_backfill)
        sat_stop = resolve_saturation_stop(saturation_stop)
        selected = select_units(
            candidates,
            files_per_unit,
            max_units,
            coverage_first=cover_first,
            cap_aware_backfill=cap_aware,
            saturation_stop=sat_stop,
            context_pool_multiplier=ctx_mult,
            context_merge_weight=ctx_weight,
            extra_anchor_patterns=extra_anchors,
        )

        reviewed = sorted({f for u in selected for f in u["files"]})
        anchors_total = sum(1 for c in candidates if is_anchor(c, extra_anchors))
        anchors_reviewed = sum(1 for f in reviewed if is_anchor(f, extra_anchors))
        _log.info(
            "bl_discover_and_cluster output",
            project_id=project_id,
            branch=branch,
            globs=len(AUTHZ_GLOBS) + len(add_globs),
            extra_globs=len(add_globs),
            extra_anchor_patterns=n_anchor_patterns,
            candidates=len(candidates),
            anchors_total=anchors_total,
            anchors_reviewed=anchors_reviewed,
            units_selected=len(selected),
            files_per_unit=int(files_per_unit),
            max_units=int(max_units),
            scan_effort=scan_effort,
            coverage_first=cover_first,
            cap_aware_backfill=cap_aware,
            saturation_stop=sat_stop,
            context_pool_multiplier=ctx_mult,
            context_merge_weight=ctx_weight,
            content_filter=want_filter,
            candidates_before_content_filter=candidates_before,
            distinct_reviewed_files=len(reviewed),
        )
        _log.info(
            "bl_discover_reviewed_pool", n=len(reviewed), files="|".join(reviewed)
        )
        return selected
