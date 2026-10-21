"""Severity-first ordering of BL findings, so a capped fan-out keeps the most important ones.

Units are ordered by severity descending, then tier ascending (tier 1 is the
most important), then arrival index. Ordering never filters: the same units come
back, only reordered, and the caller applies any cap.

``severity`` and ``tier`` arrive in inconsistent shapes (mixed case, integer or
string tiers such as ``2``, ``"T2"``, or a severity word like ``"HIGH"``). None of
these functions raises: any unrecognised value maps to a neutral, mid-scale
bucket, so malformed input neither crowds out real criticals nor is dropped
first.
"""

import json
import math
from typing import Any

__all__ = ["order_by_priority"]

# Most severe first.
SEVERITY_ORDER: tuple[str, ...] = (
    "critical",
    "high",
    "medium",
    "low",
    "info",
)

_SEVERITY_ALIASES: dict[str, str] = {
    "critical": "critical",
    "crit": "critical",
    "high": "high",
    "medium": "medium",
    "med": "medium",
    "moderate": "medium",
    "low": "low",
    "info": "info",
    "informational": "info",
}

NEUTRAL_SEVERITY = "medium"

# Tiers run 1 (most important) to 3.
MIN_TIER = 1
MAX_TIER = 3
NEUTRAL_TIER = 2

_SEVERITY_RANK: dict[str, int] = {
    name: index for index, name in enumerate(SEVERITY_ORDER)
}


def normalize_severity(value: Any) -> str:
    """Map any severity value onto one of :data:`SEVERITY_ORDER`, case- and whitespace-insensitively.

    Anything unrecognised becomes :data:`NEUTRAL_SEVERITY`, including placeholders such as ``"unknown"``, ``"none"``,
    ``"n/a"`` and ``""``: a severity nobody assessed must not sort below a real ``info``.
    """
    if not isinstance(value, str):
        return NEUTRAL_SEVERITY
    return _SEVERITY_ALIASES.get(value.strip().lower(), NEUTRAL_SEVERITY)


def severity_rank(value: Any) -> int:
    """Sort rank for a severity: 0 is the most severe."""
    return _SEVERITY_RANK[normalize_severity(value)]


def normalize_tier(value: Any) -> int:
    """Map any ``tier`` value onto an integer in ``1..3`` where smaller is more important.

    Accepts ``2``, ``"1"``, ``"T2"`` and ``"tier 3"``. A parsed tier outside ``1..3`` is clamped into it, so ``0`` and
    ``-1`` become 1 and ``99`` becomes 3. Anything that is not a tier, including a severity word such as ``"HIGH"``,
    ``None`` and ``bool``, becomes :data:`NEUTRAL_TIER`.
    """
    return min(max(_parse_tier(value), MIN_TIER), MAX_TIER)


def _parse_tier(value: Any) -> int:
    """The raw integer tier, unclamped, or :data:`NEUTRAL_TIER` when ``value`` is not a tier."""
    # bool is an int subclass; True/False are not tiers.
    if isinstance(value, bool):
        return NEUTRAL_TIER
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # NaN/inf are not tiers, and int() either mangles or refuses them. A finite
        # float truncates toward zero, as int() does: 1.5 -> 1, 2.9 -> 2.
        return int(value) if math.isfinite(value) else NEUTRAL_TIER
    if isinstance(value, str):
        return _tier_from_text(value)
    return NEUTRAL_TIER


def _tier_from_text(value: str) -> int:
    """The integer after an optional ``tier``/``t`` label and separator, or :data:`NEUTRAL_TIER`.

    Separators are stripped only after a label: ``"T-1"`` and ``"tier -3"`` read as 1 and 3, while a plain numeric
    string keeps its sign, so ``"-5"`` is -5 (and clamps to 1) exactly like the int ``-5``.
    """
    text = value.strip().lower()
    for label in ("tier", "t"):
        if text.startswith(label):
            text = text[len(label) :].lstrip("-_ :")
            break
    try:
        return int(text)
    except ValueError:
        return NEUTRAL_TIER


def _as_mapping(unit: Any) -> dict:
    """The unit as a dict, decoding a JSON-object string; ``{}`` when no dict can be recovered."""
    if isinstance(unit, dict):
        return unit
    if isinstance(unit, str):
        try:
            parsed = json.loads(unit)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def priority_key(unit: Any, index: int) -> tuple[int, int, int]:
    """Sort key for one unit: ``(severity_rank, tier, arrival_index)``, all ints so the order is total."""
    obj = _as_mapping(unit)
    return (
        severity_rank(obj.get("severity")),
        normalize_tier(obj.get("tier")),
        index,
    )


def order_by_priority(units: list[Any]) -> list[Any]:
    """Return ``units`` reordered most-important-first; deterministic for a given input."""
    return [
        unit
        for _, unit in sorted(
            ((priority_key(unit, index), unit) for index, unit in enumerate(units)),
            key=lambda pair: pair[0],
        )
    ]
