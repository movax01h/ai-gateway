"""Deterministic review-unit selection for the BL analyzer's discovery stage.

Pure path logic, no executor calls and no file reads: noise filtering, entry-point (anchor) and role classification,
clustering, and the tiered selection the ``bl_discover_and_cluster`` tool returns. The same candidates yield the same
units every run, and a higher ``max_units`` includes more files. Context files join entry points by path affinity alone
(same directory, same resource name, same module).

Only a prefix of the emitted unit list is reviewed (the downstream fan-out cap truncates it), so order is part of the
output: the ordering choices below exist to make any prefix representative, and a new candidate can displace a
relevant file from that prefix.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import structlog

from duo_workflow_service.bl_security.discovery_patterns import (
    ANCHOR_RES,
    AUTHZ_GLOBS,
    NOISE_SUFFIX,
    ROLE_SEGMENTS,
)

# Same logger name as the tool, so the log stream is unchanged.
_log = structlog.stdlib.get_logger("bl_discovery")


def _normalize_glob(pattern: str) -> str:
    """Drop empty and ``.`` segments; a trailing ``/`` means the files directly under that directory."""
    segs = [seg for seg in pattern.split("/") if seg not in ("", ".")]
    if pattern.endswith("/") and segs:
        segs.append("*")
    return "/".join(segs)


# ---------------------------------------------------------------------------
# Noise exclusion. Migrations are deliberately kept.
# ---------------------------------------------------------------------------
_NOISE_DIR = re.compile(
    r"(^|/)(spec|test|tests|Test|Tests|__tests__|fixtures|factories|coverage|"
    r"docs|doc|node_modules|vendor|target|obj|\.github|\.gitlab)(/)"
)
# Anchored with (^|/): these are noise at the repo root as much as nested.
_NOISE_PATH = re.compile(r"(^|/)(static|assets|public|dist|build)/")
_NOISE_LOCKFILE = frozenset(
    "package-lock.json yarn.lock pnpm-lock.yaml Cargo.lock poetry.lock Gemfile.lock composer.lock".split()
)

# Security-relevance keywords, matched over a unit's whole file-path text. The
# score is additive and order-only: it ranks units and never admits or excludes
# a file, so a keyword that does not fire costs priority, never coverage.
_SCORE_KEYWORDS = tuple(
    "admin auth user session account payment order token api permission access role "
    "transfer password login webhook upload org team member group repo".split()
)


def _is_noise(path: str) -> bool:
    p = path.strip()
    return (
        not p
        or p.rsplit("/", 1)[-1] in _NOISE_LOCKFILE
        or any(rx.search(p) for rx in (_NOISE_DIR, _NOISE_PATH, NOISE_SUFFIX))
    )


def enumerate_candidates(find_files_results: Any) -> List[str]:
    """Union per-glob ``find_files`` results into a noise-filtered, de-duplicated, sorted candidate list.

    Args:
        find_files_results: A flat iterable of paths, or an iterable of per-glob path lists / newline-joined strings.

    Returns:
        The candidate paths, sorted so the output is independent of glob-firing order.
    """
    seen = set()
    for group in find_files_results:
        if group is None:
            continue
        if isinstance(group, str):
            items = group.splitlines()
        elif isinstance(group, (list, tuple, set)):
            items = list(group)
        else:
            items = [group]
        for raw in items:
            path = str(raw).strip()
            if not path or path in seen:
                continue
            if _is_noise(path):
                continue
            seen.add(path)
    return sorted(seen)


_TRUE_TOKENS = frozenset({"true", "1", "yes", "on"})


def _resolve_bool_dial(value: Optional[Any]) -> bool:
    """Interpret a boolean flow-config dial; anything unrecognized means OFF.

    Flow-config literals arrive as text, so a token spelling is accepted alongside a real bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TOKENS
    return False


# Public names for the boolean dials (the flow-config tests resolve the shipped
# literals through them).
resolve_content_filter = _resolve_bool_dial
resolve_coverage_first = _resolve_bool_dial
resolve_cap_aware_backfill = _resolve_bool_dial
resolve_saturation_stop = _resolve_bool_dial


def content_filter_candidates(
    candidates: List[str], matched: Any = (), extra_anchors: Sequence[Any] = ()
) -> List[str]:
    """Keep the candidates whose body is in the content net, plus every anchor.

    Anchors are exempt so the filter cannot undo ``handler_coverage_units``' guarantee that every entry point gets a
    unit.

    Args:
        candidates: The candidate paths.
        matched: The paths whose body matched the content net.
        extra_anchors: Extra compiled entry-point patterns.

    Returns:
        An order-preserving subset of ``candidates``.
    """
    keep = matched if isinstance(matched, (set, frozenset)) else set(matched or ())
    return [p for p in candidates if p in keep or _is_anchor(p, extra_anchors)]


# ---------------------------------------------------------------------------
# Path helpers (no body reads)
# ---------------------------------------------------------------------------
# Context (non-anchor) file smells -- the files a reviewer must read alongside
# an entrypoint to judge one feature's authorization.
_CONTEXT_RE = re.compile(
    r"(model|policy|serializer|service|finder|permission|ability|"
    r"middleware|guard|auth|access|role|"
    r"filter|servlet|voter|security|authentic|authoriz)",
    re.IGNORECASE,
)


def _is_anchor(path: str, extra: Sequence[Any] = ()) -> bool:
    """Whether a path is a request entry point.

    ``extra`` holds additional compiled per-repository patterns, unioned with ``ANCHOR_RES``; nothing can suppress a
    built-in pattern. A tuple, so callers can memoise.
    """
    if any(rx.search(path) for rx in ANCHOR_RES):
        return True
    return bool(extra) and any(rx.search(path) for rx in extra)


is_anchor = _is_anchor  # public name for the tool module


def _unit_name(path: str) -> str:
    """A unit name from a path: extension dropped, ``/`` as ``.``."""
    return re.sub(r"\.[^.]+$", "", path).replace("/", ".")


def _chunks(paths: List[str], size: int) -> List[List[str]]:
    """``paths`` in consecutive groups of ``size``, the last possibly shorter."""
    return [paths[i : i + size] for i in range(0, len(paths), size)]


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _module_dir(path: str) -> str:
    """Top two path segments -- a coarse feature/module bucket."""
    segs = path.split("/")
    return "/".join(segs[:2]) if len(segs) >= 2 else (segs[0] if segs else "")


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _resource_tokens(path: str) -> set:
    """Lowercased alnum tokens of the basename (ext stripped), plural/singular
    folded -- the resource-name signal for affinity."""
    base = path.rsplit("/", 1)[-1]
    base = re.sub(r"\.[^.]+$", "", base)
    base = re.sub(
        r"_(controller|resolver|serializer|policy|finder|service|"
        r"viewset|viewsets|view|views|model|models|permission|"
        r"permissions|handler|middleware|guard)$",
        "",
        base,
    )
    # CamelCase authz suffixes (Java/.NET/PHP: FooController, FooResource, ...)
    base = re.sub(
        r"(Controller|Resource|Action|Servlet|Service|Filter|"
        r"Middleware|Voter|Policy|Handler)$",
        "",
        base,
    )
    toks = set()
    for t in _TOKEN_RE.findall(base.lower()):
        if len(t) < 3:
            continue
        toks.add(t)
        toks.add(t.rstrip("s"))
    return toks


def _affinity(anchor: str, ctx: str) -> int:
    """Deterministic affinity rank of a context file to an anchor (higher = stronger).

    Independent of files_per_unit, so each anchor's context order -- and the per-anchor superset property of
    ``cluster`` -- is stable across unit widths.
    """
    rank = 0
    if _dirname(anchor) and _dirname(anchor) == _dirname(ctx):
        rank += 100
    if _module_dir(anchor) and _module_dir(anchor) == _module_dir(ctx):
        rank += 40
    at, ct = _resource_tokens(anchor), _resource_tokens(ctx)
    shared = at & ct
    rank += 10 * len(shared)
    if _CONTEXT_RE.search(ctx):
        rank += 5
    return rank


def score(unit: dict) -> int:
    """Security-relevance soft-priority score over a unit's file paths.

    Args:
        unit: A unit dict with a ``files`` list.

    Returns:
        How many ``_SCORE_KEYWORDS`` occur in the paths.
    """
    text = " ".join(unit.get("files", [])).lower()
    return sum(1 for k in _SCORE_KEYWORDS if k in text)


def cluster(
    candidates: List[str], files_per_unit: int, extra_anchors: Sequence[Any] = ()
) -> List[dict]:
    """Group candidates into review units: one per anchor, joined with its highest-affinity context files.

    Each anchor's context is ordered once (affinity DESC, path ASC) and sliced to ``files_per_unit - 1``, so a given
    anchor's unit at a larger ``files_per_unit`` is a superset of its unit at a smaller one. That does NOT hold for the
    reviewed set: under a fixed unit budget a larger ``files_per_unit`` spends the budget on fewer anchors, so
    ``files_per_unit`` values may only be compared at equal file slots.

    Args:
        candidates: The sorted candidate paths.
        files_per_unit: Max files per unit (the anchor plus its context).
        extra_anchors: Extra compiled entry-point patterns.

    Returns:
        One ``{"name", "files"}`` unit per anchor, anchors in lexicographic order.
    """
    fpu = max(1, int(files_per_unit))
    cand = list(candidates)  # already lexicographically sorted upstream
    anchors = [p for p in cand if _is_anchor(p, extra_anchors)]
    contexts = [p for p in cand if not _is_anchor(p, extra_anchors)]

    units: List[dict] = []
    for anchor in anchors:  # anchors iterate in lexicographic order (stable)
        ranked = sorted(
            ((-_affinity(anchor, c), c) for c in contexts if c != anchor),
        )
        ctx_ordered = [c for neg_rank, c in ranked if -neg_rank > 0]
        files = [anchor] + ctx_ordered[: fpu - 1]
        units.append({"name": _unit_name(anchor), "files": files})
    return units


def _authz_unit(files: List[str]) -> dict:
    """Build a guaranteed-coverage unit dict named after its anchor (files[0])."""
    return {"name": f"authz_handlers.{_unit_name(files[0])}", "files": files}


# ---------------------------------------------------------------------------
# Language-agnostic SEMANTIC surface (the business-logic layer)
# ---------------------------------------------------------------------------
# The anchor tier covers entry points, but ownership/tenancy invariants (IDOR),
# mass-assignment allowlists and check-then-use ordering (TOCTOU) live in the
# model / service / policy layer the entry point delegates to. Roles are matched
# over path SEGMENTS, not framework suffixes: a `models/` or `services/`
# directory means the same thing in Go, Rails, Django, Express/Nest, Spring,
# ASP.NET and Laravel.

# Role precedence when a path matches more than one bucket. `access` first: a
# file under services/auth/ is access-control machinery, not generic logic.
_ROLE_ORDER = ("access", "state", "logic")


def _role_of(path: str) -> Optional[str]:
    """Classify a path into a semantic ROLE from its directory + basename tokens.

    Segment-based (not suffix-based) so the classification survives a change of language or framework. Returns None for
    paths outside the semantic surface.
    """
    segs = path.lower().split("/")
    tokens = set(segs[:-1])
    stem = re.sub(r"\.[^.]+$", "", segs[-1]) if segs else ""
    tokens.update(_TOKEN_RE.findall(stem))
    for role in _ROLE_ORDER:
        if tokens & ROLE_SEGMENTS[role]:
            return role
    return None


# Units reserved for the semantic surface, split evenly across the three roles so
# one large layer cannot crowd out the others.
SEMANTIC_UNIT_BUDGET: int = 400


def semantic_surface_units(
    candidates: List[str],
    files_per_unit: int,
    budget: int = SEMANTIC_UNIT_BUDGET,
    extra_anchors: Sequence[Any] = (),
) -> List[dict]:
    """Guaranteed, language-agnostic coverage of the STATE / LOGIC / ACCESS layer the entry points delegate to.

    Anchors are skipped (``handler_coverage_units`` guarantees them), so the two tiers never review the same file
    twice. The budget is split evenly across roles, and within a role files are ordered by score DESC then path ASC.

    Args:
        candidates: The candidate paths.
        files_per_unit: Max files per unit.
        budget: Total units across the three roles; 0 or less disables the tier.
        extra_anchors: Extra compiled entry-point patterns.

    Returns:
        The ``semantic_<role>.*`` units, grouped by role in ``_ROLE_ORDER``.
    """
    fpu = max(1, int(files_per_unit))
    if int(budget) <= 0:
        return []
    by_role: Dict[str, List[str]] = {r: [] for r in _ROLE_ORDER}
    for path in candidates:
        if _is_anchor(path, extra_anchors):
            continue  # entrypoints are already guaranteed by the anchor tier
        role = _role_of(path)
        if role:
            by_role[role].append(path)

    per_role_units = max(1, int(budget) // len(_ROLE_ORDER))
    units: List[dict] = []
    for role in _ROLE_ORDER:
        ranked = sorted(by_role[role], key=lambda p: (-score({"files": [p]}), p))
        for chunk in _chunks(ranked[: per_role_units * fpu], fpu):
            units.append(
                {"name": f"semantic_{role}.{_unit_name(chunk[0])}", "files": chunk}
            )
    return units


# Context units the guaranteed tiers can never starve (see select_units).
CONTEXT_UNIT_FLOOR: int = 40

# The default discovery budget. Used only to scale the context floor; not a cap.
DEFAULT_MAX_UNITS: int = 150

# Smallest context pool worth admitting at any tier.
MIN_CONTEXT_UNIT_FLOOR: int = 10


def context_unit_floor(max_units: int) -> int:
    """Scale the context-tier floor with the requested unit budget.

    A fixed floor would make a smaller ``max_units`` meaningless whenever the anchor surface already exceeds it.

    Args:
        max_units: The requested unit budget.

    Returns:
        ``CONTEXT_UNIT_FLOOR`` for ``max_units >= DEFAULT_MAX_UNITS``; otherwise that floor scaled down proportionally,
        never below ``MIN_CONTEXT_UNIT_FLOOR``.
    """
    budget = max(0, int(max_units))
    if budget >= DEFAULT_MAX_UNITS:
        return CONTEXT_UNIT_FLOOR
    return max(
        MIN_CONTEXT_UNIT_FLOOR, (CONTEXT_UNIT_FLOOR * budget) // DEFAULT_MAX_UNITS
    )


# ---------------------------------------------------------------------------
# Scan-effort tiers (discovery side)
# ---------------------------------------------------------------------------
# A tier is (files_per_unit, max_units, context_pool_multiplier,
# context_merge_weight) for the unit POOL.
#
# max_units is not a ceiling on emitted units: the anchor and semantic tiers are
# emitted in full and only the context tier is budgeted. Each tier's max_units
# must be >= its fan-out cap, or the slots the cap fills go to anchor samples and
# semantic units instead of cross-file context.
#
# context_pool_multiplier scales how many context units exist to be interleaved;
# context_merge_weight is how many prefix slots per pass the context tier wins.
# A weight without pool depth has nothing to promote, so they are paired per tier.
SCAN_EFFORT_TIERS: dict[str, tuple[int, int, int, int]] = {
    "low": (8, 62, 4, 4),
    "standard": (8, 150, 2, 1),
    "high": (8, 400, 4, 1),
}


def resolve_scan_effort(
    scan_effort: Optional[Any], files_per_unit: int, max_units: int
) -> tuple[int, int, int, int]:
    """Map an effort tier name to its unit-pool settings.

    Args:
        scan_effort: The tier name; ``None``, a non-string or an undeclared name is a miss.
        files_per_unit: The configured files per unit, kept on a miss.
        max_units: The configured unit budget, kept on a miss.

    Returns:
        ``(files_per_unit, max_units, context_pool_multiplier, context_merge_weight)`` from ``SCAN_EFFORT_TIERS``; on
        a miss, the configured values with both context dials at 1.
    """
    if isinstance(scan_effort, str):
        tier = SCAN_EFFORT_TIERS.get(scan_effort.strip().lower())
        if tier is not None:
            return tier
    return int(files_per_unit), int(max_units), 1, 1


# ---------------------------------------------------------------------------
# Prefix-representative ordering
# ---------------------------------------------------------------------------
def _weighted_round_robin(
    groups: List[List[dict]], weights: Sequence[int]
) -> List[dict]:
    """Round-robin merge that takes ``weights[i]`` units from group ``i`` per pass.

    Groups are consumed in the order given and exhausted groups are skipped; a weight below 1 counts as 1. The result
    is a permutation: a weight changes a group's share of the reviewed prefix, never its presence in the output.
    """
    out: List[dict] = []
    pos = [0] * len(groups)
    while True:
        moved = False
        for gi, g in enumerate(groups):
            for _ in range(max(1, int(weights[gi]))):
                if pos[gi] < len(g):
                    out.append(g[pos[gi]])
                    pos[gi] += 1
                    moved = True
        if not moved:
            break
    return out


def _spread_by_module(units: List[dict]) -> List[dict]:
    """Reorder units so no single directory dominates any prefix of the list.

    Units are bucketed by the module dir of their first file and merged round-robin, buckets in lexicographic order;
    otherwise the reviewed prefix covers only the alphabetically-first slice of the repo.
    """
    buckets: Dict[str, List[dict]] = {}
    for u in units:
        key = _module_dir(u["files"][0]) if u.get("files") else u["name"]
        buckets.setdefault(key, []).append(u)
    return _weighted_round_robin(
        [buckets[k] for k in sorted(buckets)], [1] * len(buckets)
    )


def handler_coverage_units(
    candidates: List[str],
    extra_anchors: Sequence[Any] = (),
) -> List[dict]:
    """Guaranteed entry-point coverage: every anchor path gets a focused single-file unit of its own.

    No handler can then be dropped by the score sort or the ``max_units`` cap. Subtle per-handler authorization bugs
    are found reliably only when the handler is its own focused unit.

    Args:
        candidates: The candidate paths.
        extra_anchors: Extra compiled entry-point patterns.

    Returns:
        One ``authz_handlers.*`` unit per anchor (never a non-anchor file), so the count scales with the anchor surface.
    """
    # Within one directory, built-in anchors come first, then the ones only a
    # per-repository pattern marks (select_units later spreads units by directory).
    anchors = sorted(
        (p for p in candidates if _is_anchor(p, extra_anchors)),
        key=lambda p: (not _is_anchor(p), p),
    )
    return [_authz_unit([anchor]) for anchor in anchors]


# How many times each authz-anchor unit is reviewed. Subtle per-handler
# authorization bugs are detected intermittently, so reviewing the same unit
# several times and unioning the findings raises recall. Only the anchor tier is
# multiplied, so cost grows linearly in anchor-unit reviews. 1 disables sampling.
AUTHZ_SAMPLE_COUNT: int = 3


def _multi_sample_authz_units(units: List[dict]) -> List[dict]:
    """Repeat each authz-anchor unit ``AUTHZ_SAMPLE_COUNT`` times (block repeat).

    Block order ([all units], then [all units] again, ...) degrades gracefully
    under a downstream fan-out cap: every unit keeps at least ``floor(cap/len)``
    samples and none is dropped entirely, versus interleaving which would starve
    later units. Each sample is a distinct dict (defensive copy) so no downstream
    in-place mutation can alias across samples.
    """
    if AUTHZ_SAMPLE_COUNT <= 1:
        return list(units)
    return [dict(u) for _ in range(AUTHZ_SAMPLE_COUNT) for u in units]


# ---------------------------------------------------------------------------
# SCOPED SCAN: caller-supplied file scope (`target_files`)
# ---------------------------------------------------------------------------
# Naming the paths to review skips discovery entirely -- no glob firing, no
# enumeration, no selection -- so testing one hypothesis does not cost a full
# repo scan. A scoped scan differs from a full scan in scope only:
#   * units are multi-sampled like the anchor tier, so a miss is comparable with
#     a full-scan miss;
#   * unit names are neutral (no `authz_handlers.` prefix): the reviewer sees the
#     name, and a handler label would prime it.
# An absent, empty or unusable value (see ``resolve_target_files``) runs full
# discovery.
def target_file_units(files: List[str], files_per_unit: int) -> List[dict]:
    """Build scoped-scan review units from an explicit path list (no discovery).

    Args:
        files: The paths to review, in order.
        files_per_unit: Paths per unit, so 1 makes each file a focused unit.

    Returns:
        Neutrally named units (see the SCOPED SCAN note), multi-sampled like the anchor tier.
    """
    units = [
        {"name": _unit_name(chunk[0]), "files": chunk}
        for chunk in _chunks(files, max(1, int(files_per_unit)))
    ]
    return _multi_sample_authz_units(units)


def cap_scoped_units(units: List[dict], max_units: int) -> List[dict]:
    """Cap scoped-scan units at ``max_units``, warning when whole file groups are dropped.

    Args:
        units: The scoped-scan units.
        max_units: The unit cap.

    Returns:
        The first ``max_units`` units.
    """
    selected = units[: int(max_units)]
    dropped = {tuple(u["files"]) for u in units} - {tuple(u["files"]) for u in selected}
    if dropped:
        _log.warning(
            "bl_discover scoped_scan over max_units; files are not reviewed",
            groups_dropped=len(dropped),
            files_dropped=sum(len(g) for g in dropped),
            max_units=int(max_units),
        )
    return selected


# ---------------------------------------------------------------------------
# Coverage-aware selection dials (off by default; the BL flow turns them on)
# ---------------------------------------------------------------------------
# Every file slot in the reviewed prefix is paid budget, and neighbouring anchors
# in the context tier share the same top affinity files. ``coverage_first`` re-spends those repeated slots on unclaimed
# files at identical cost (see ``_cover_before_repeat``); ``cap_aware_backfill``
# refines it and is inert without it; ``saturation_stop`` cuts the tail that
# surfaces no new file (see ``_stop_at_saturation``).


# ---------------------------------------------------------------------------
# Per-repository additions to the static net (empty by default; the BL flow feeds
# them from its tuner stage)
# ---------------------------------------------------------------------------
# AUTHZ_GLOBS and ANCHOR_RES are framework vocabulary, which is also their
# ceiling: a project that keeps its entry points off-convention is under-served,
# and selection cannot recover a file the net never enumerated. An upstream
# stage may therefore propose extra globs and entry-point patterns for one
# repository. They are UNIONED onto the built-ins and can never replace one.
# Additive protects the candidate pool, not the reviewed set: under a fixed
# budget an added candidate can evict a relevant file, so evaluate changes per
# repository.
#
# Model-written input is bounded:
#   * more than MAX_EXTRA_GLOBS / MAX_EXTRA_ANCHOR_PATTERNS entries means no
#     additions at all, never a truncated prefix;
#   * a pattern matching the whole tree is dropped;
#   * a glob with `[`, `]`, `?` or a backslash is dropped: a malformed character
#     class can fail the whole listing call, and no path convention needs one;
#   * an anchor pattern that does not compile is dropped individually.
# Every miss yields an empty list.
MAX_EXTRA_GLOBS: int = 40
MAX_EXTRA_ANCHOR_PATTERNS: int = 40
# Entry points get guaranteed, multi-sampled units that max_units does not
# bound, so an anchor pattern matching more files than this adds none.
MAX_FILES_PER_ANCHOR_PATTERN: int = 50
# And the accepted patterns together may add no more than this many entry
# points beyond the built-in ones (40 patterns x 50 files would otherwise be
# 2,000, each multi-sampled).
MAX_EXTRA_ANCHOR_FILES: int = 200

# Long enough for any real convention; rejects a runaway generation or pasted
# body.
MAX_EXTRA_PATTERN_CHARS: int = 200

# Globs that keep every path without being segment-wise wildcards (those are
# caught by `_glob_is_too_broad`).
_TOO_BROAD_GLOBS = frozenset({"", "*.*", "**.*"})

# Regexes that match every path (`search`, not `fullmatch`, is what runs).
_ANCHOR_RE_LITERAL = re.compile(r"[A-Za-z0-9_]")

_UNSAFE_GLOB_CHARS = "[]?" + chr(92)

_RE_TYPE = type(re.compile(""))


def _extra_items(value: Optional[Any], json_key: str) -> List[str]:
    """Normalise a caller-supplied value to a list of non-empty strings.

    Accepts a list of strings, or an object whose ``json_key`` holds one (so one structured answer can feed both
    inputs). Anything else, including a list holding a non-string, yields ``[]``.
    """
    if isinstance(value, dict):
        value = value.get(json_key)
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            # A non-string entry means this is not a pattern list at all.
            return []
        item = item.strip()
        if item:
            out.append(item)
    return out


def _glob_is_not_a_glob(pattern: str) -> bool:
    """Whether a value is a bare word (``off``, ``none``) rather than a path pattern.

    A glob has to name an extension, a directory or a wildcard.
    """
    return not any(c in pattern for c in "*/.")


def _glob_is_too_broad(pattern: str) -> bool:
    """Whether a glob keeps every path: every segment is ``*``, ``**`` or empty."""
    if pattern in _TOO_BROAD_GLOBS:
        return True
    return all(seg in ("*", "**", "") for seg in pattern.split("/"))


def resolve_extra_globs(value: Optional[Any]) -> List[str]:
    """The additional path globs, normalised; anything unrecognized means no additions.

    Args:
        value: A list of globs, or an object whose ``globs`` key holds one.

    Returns:
        The accepted globs for the caller to union onto ``AUTHZ_GLOBS`` (duplicates of a built-in are dropped), or
        ``[]`` when the input is unrecognized or over ``MAX_EXTRA_GLOBS``.
    """
    items = _extra_items(value, "globs")
    if not items:
        return []
    if len(items) > MAX_EXTRA_GLOBS:
        _log.info(
            "bl_discover extra_globs over the cap -- no additions",
            supplied=len(items),
            cap=MAX_EXTRA_GLOBS,
        )
        return []
    seen = set(AUTHZ_GLOBS)
    out: List[str] = []
    for pattern in items:
        if len(pattern) > MAX_EXTRA_PATTERN_CHARS:
            continue
        if any(c in pattern for c in _UNSAFE_GLOB_CHARS):
            continue
        pattern = _normalize_glob(pattern)
        if pattern in seen:
            continue
        if _glob_is_not_a_glob(pattern):
            continue
        if _glob_is_too_broad(pattern):
            continue
        seen.add(pattern)
        out.append(pattern)
    return out


def resolve_extra_anchor_patterns(value: Optional[Any]) -> List[Any]:
    """The additional entry-point path patterns, compiled; unioned onto ``ANCHOR_RES`` by the caller.

    Args:
        value: A list of regex strings, an object whose ``anchor_patterns`` key holds one, or already-compiled
            patterns (returned as-is, so the call is idempotent).

    Returns:
        The compiled patterns; one that does not compile is dropped on its own. ``[]`` when the input is unrecognized
        or over ``MAX_EXTRA_ANCHOR_PATTERNS``.
    """
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, _RE_TYPE) for v in value)
    ):
        return list(value)
    items = _extra_items(value, "anchor_patterns")
    if not items:
        return []
    if len(items) > MAX_EXTRA_ANCHOR_PATTERNS:
        _log.info(
            "bl_discover extra_anchor_patterns over the cap -- no additions",
            supplied=len(items),
            cap=MAX_EXTRA_ANCHOR_PATTERNS,
        )
        return []
    out: List[Any] = []
    seen = set()
    for pattern in items:
        if pattern in seen:
            continue
        seen.add(pattern)
        if len(pattern) > MAX_EXTRA_PATTERN_CHARS:
            continue
        if not _ANCHOR_RE_LITERAL.search(pattern):
            # Only wildcards/anchors/separators: matches every path.
            continue
        try:
            out.append(re.compile(pattern))
        except re.error as e:
            _log.info(
                "bl_discover extra anchor pattern does not compile -- dropped",
                pattern=pattern,
                err=str(e),
            )
    return out


def cap_extra_anchor_files(
    extra_anchors: Sequence[Any], matched: Sequence[str]
) -> Tuple[Any, ...]:
    """The extra entry-point patterns, bounded to ``MAX_EXTRA_ANCHOR_FILES`` added entry points in total.

    Args:
        extra_anchors: The compiled extra patterns.
        matched: Every path the patterns match; paths a built-in pattern already marks do not count and are never
            affected.

    Returns:
        The patterns unchanged within the cap. Over it, one exact-path pattern for the first ``MAX_EXTRA_ANCHOR_FILES``
        added paths by path order, so a dropped path is neither added nor labelled an entry point.
    """
    added = sorted({p for p in matched if not _is_anchor(p)})
    if len(added) <= MAX_EXTRA_ANCHOR_FILES:
        return tuple(extra_anchors)
    kept = added[:MAX_EXTRA_ANCHOR_FILES]
    _log.warning(
        "bl_discover extra anchor patterns add too many files -- capped",
        kept=len(kept),
        dropped=len(added) - len(kept),
        cap=MAX_EXTRA_ANCHOR_FILES,
    )
    return (re.compile(r"\A(?:" + "|".join(map(re.escape, kept)) + r")\Z"),)


def _backfill_queue(
    candidates: List[str], extra_anchors: Sequence[Any] = ()
) -> List[str]:
    """The whole candidate surface, best-first: anchors, then context files, each by ``score`` DESC then path.

    Anchors first: under any production cap most solo anchor units are truncated, so an unclaimed anchor is the most
    valuable thing a recovered slot can buy. Score order rather than the anchor tier's emission order, because draining
    in emission order hands early units exactly the anchors whose solo units come next.
    """
    ranked = sorted(candidates, key=lambda p: (-score({"files": [p]}), p))
    return [p for p in ranked if _is_anchor(p, extra_anchors)] + [
        p for p in ranked if not _is_anchor(p, extra_anchors)
    ]


def _cover_before_repeat(
    units: List[dict],
    candidates: List[str],
    max_units: Optional[int] = None,
    extra_anchors: Sequence[Any] = (),
) -> List[dict]:
    """Spend each file slot on an unclaimed file before any file gets a second one.

    Walks units in emitted order, so the units the fan-out actually reviews get first claim. Each multi-file unit drops
    files an earlier unit already claimed and backfills from the unclaimed queue up to its original size. Unit count,
    sizes, names, positions and the tier interleave are preserved, so review cost is unchanged. Solo units are left
    alone: each is a focused review of one anchor. Once nothing is unclaimed, a unit keeps its own files.

    An anchor backfilled into an early unit can be reviewed again by its own solo unit. Given ``max_units``, anchors
    whose solo unit falls inside the reviewed prefix move to the back of the queue. They are deprioritised, not
    excluded: the actual fan-out cap can be smaller than ``max_units``, and excluding them collapses anchor reach.
    """
    queue = _backfill_queue(candidates, extra_anchors)
    # Stable partition: anchors whose own solo unit is inside the reviewed prefix
    # go last.
    if max_units is not None:
        solo_in_prefix = {
            files[0]
            for unit in units[: int(max_units)]
            for files in (list(unit.get("files") or []),)
            if len(files) == 1
        }
        if solo_in_prefix:
            queue = [p for p in queue if p not in solo_in_prefix] + [
                p for p in queue if p in solo_in_prefix
            ]
    cursor = 0
    covered: set = set()
    out: List[dict] = []
    for unit in units:
        files = list(unit.get("files") or [])
        if len(files) < 2:  # solo anchor units are left alone
            covered.update(files)
            out.append(unit)
            continue
        kept = [f for f in files if f not in covered]
        while len(kept) < len(files) and cursor < len(queue):
            nxt = queue[cursor]
            cursor += 1
            if nxt not in covered and nxt not in kept:
                kept.append(nxt)
        if len(kept) < len(files):
            # The unclaimed queue ran dry: refill with the unit's own files so it
            # never shrinks and the run's cost never changes.
            kept += [f for f in files if f not in kept][: len(files) - len(kept)]
        covered.update(kept)
        out.append({**unit, "files": kept})
    return out


def _stop_at_saturation(units: List[dict]) -> List[dict]:
    """Cut the tail of the emitted list that cannot surface a single new file.

    Keeps the prefix up to and including the last unit that contributes a file no earlier unit carries. Every later
    unit's files are a subset of that prefix's union, so the reviewed file set is identical -- at any cap, since cutting
    and then capping gives the same prefix as capping alone. It is a per-repo stop: a global cap would have to suit the
    repo that saturates last.

    What is dropped is repetition, not nothing: mostly solo anchor units whose anchor an earlier unit already carries,
    and repeat samples. Review depth per anchor can fall, which is why this dial is off by default (the BL flow turns it
    on).
    """
    seen: set = set()
    last_new = -1
    for i, unit in enumerate(units):
        files = unit.get("files") or []
        if any(f not in seen for f in files):
            last_new = i
        seen.update(files)
    return units[: last_new + 1]


def select_units(
    candidates: List[str],
    files_per_unit: int,
    max_units: int,
    coverage_first: bool = False,
    cap_aware_backfill: bool = False,
    saturation_stop: bool = False,
    extra_anchor_patterns: Optional[Any] = None,
    context_pool_multiplier: int = 1,
    context_merge_weight: int = 1,
) -> List[dict]:
    """Deterministic review-unit selection over three tiers, interleaved so any prefix samples all three.

    1. ``handler_coverage_units`` -- every anchor, multi-sampled; never truncated.
    2. ``semantic_surface_units`` -- the state / logic / access layer; never truncated.
    3. ``cluster`` -- anchor-plus-context units, score ordered, budgeted to ``max_units`` minus the anchor units but
        never below ``context_unit_floor(max_units) * context_pool_multiplier``.

    Args:
        candidates: The sorted candidate paths.
        files_per_unit: Max files per unit.
        max_units: The context-tier budget (not a cap on emitted units).
        coverage_first: Spend repeated file slots on unclaimed files (see ``_cover_before_repeat``).
        cap_aware_backfill: With ``coverage_first``, deprioritise anchors whose solo unit is in the reviewed prefix.
        saturation_stop: Cut the tail that adds no new file (see ``_stop_at_saturation``).
        extra_anchor_patterns: Extra entry-point patterns (see ``resolve_extra_anchor_patterns``).
        context_pool_multiplier: Scales the context-tier floor (see ``SCAN_EFFORT_TIERS``).
        context_merge_weight: Context units taken per interleave pass.

    Returns:
        The ordered unit list. The dials are off (or 1) by default.
    """
    extra_anchors: Tuple[Any, ...] = tuple(
        resolve_extra_anchor_patterns(extra_anchor_patterns)
    )
    guaranteed = handler_coverage_units(candidates, extra_anchors)
    # Spread by module before sampling, so any prefix covers the whole entrypoint
    # surface; block-repeat sampling keeps sample 1 of every unit ahead of sample
    # 2 of any, so a downstream cap costs samples, never coverage.
    guaranteed = _spread_by_module(guaranteed)
    sampled = _multi_sample_authz_units(guaranteed)
    semantic = _spread_by_module(
        semantic_surface_units(candidates, files_per_unit, extra_anchors=extra_anchors)
    )
    context = cluster(candidates, files_per_unit, extra_anchors)
    context.sort(key=lambda u: (-score(u), u["files"][0] if u["files"] else u["name"]))
    context_budget = max(
        int(max_units) - len(guaranteed),
        context_unit_floor(max_units) * max(1, int(context_pool_multiplier)),
    )
    context = context[:context_budget]

    # Interleave; concatenation would put every semantic unit behind all anchor samples.
    ordered = _weighted_round_robin(
        [sampled, semantic, context], (1, 1, int(context_merge_weight))
    )
    # After the interleave, so the walk sees units in the order the fan-out
    # truncates them.
    if coverage_first:
        ordered = _cover_before_repeat(
            ordered,
            candidates,
            max_units if cap_aware_backfill else None,
            extra_anchors,
        )
    # Last: it must see the final file assignment.
    if saturation_stop:
        ordered = _stop_at_saturation(ordered)
    return ordered
