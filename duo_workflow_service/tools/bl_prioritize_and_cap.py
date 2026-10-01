"""Pre-dispatch ordering and capping for a `for_each`-driven BL Security Analyzer stage.

`for_each` has no concept of severity ordering or effort-tier caps, so this tool
computes the final, already-ordered, already-capped unit list before
`for_each.items` reads it. Ordering uses `bl_security.priority.order_by_priority`:
same elements, same count, only the order changes, so a cap drops the least
severe units rather than an arbitrary arrival-order tail.
"""

import json
from typing import Any, ClassVar, List, Optional, Type

from packaging.version import Version
from pydantic import BaseModel, Field, NonNegativeInt, model_validator

from duo_workflow_service.bl_security.priority import order_by_priority
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig

# for_each parses the units dict and hands each unit to an agent prompt. Display
# truncation (200 KiB) would turn it into a string once the findings are large.
_NO_TRUNCATION = TruncationConfig(
    max_bytes=64 * 1024 * 1024, truncated_size=64 * 1024 * 1024
)

__all__ = [
    "BlPrioritizeAndCap",
    "BlPrioritizeAndCapInput",
    "resolve_effective_max_units",
]


class BlPrioritizeAndCapInput(BaseModel):
    units: List[Any] = Field(description="The raw, arrival-ordered unit list.")
    order_by_severity: bool = Field(
        default=False,
        description="When true, order units by severity/tier before capping so "
        "the cap drops the LEAST severe unit, not an arbitrary arrival-order "
        "tail. Same elements, same count -- only the order changes.",
    )
    max_units: Optional[NonNegativeInt] = Field(
        default=None,
        description="The default cap, used when no effort tier is selected or "
        "`max_units_tiers`/`scan_effort` don't resolve one.",
    )
    max_units_tiers: Optional[dict[str, NonNegativeInt]] = Field(
        default=None,
        description='Effort-tier cap table, e.g. {"low": 62, "standard": 150, '
        '"high": 400}. Consulted only when `scan_effort` names a declared tier.',
    )
    scan_effort: Optional[str] = Field(
        default=None,
        description="The effort tier name selected for this run, or omitted/"
        "unrecognized to fall back to `max_units`.",
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_json_string_fields(cls, data: Any) -> Any:
        """JSON-decode `units` and `max_units_tiers` when they arrive as strings.

        Flow-config literal inputs (`literal: true`) are always plain strings; there is no way to author a native YAML
        mapping or list through that mechanism.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for field in ("units", "max_units_tiers"):
            value = data.get(field)
            if isinstance(value, str):
                try:
                    data[field] = json.loads(value)
                except (ValueError, TypeError):
                    pass
        return data


def resolve_effective_max_units(
    *,
    max_units: Optional[int],
    max_units_tiers: Optional[dict],
    scan_effort: Optional[str],
) -> Optional[int]:
    """Return the cap for `scan_effort` from `max_units_tiers`; every miss falls back to `max_units`."""
    if not max_units_tiers or not scan_effort:
        return max_units
    effort = scan_effort.strip().lower()
    tiers = {k.lower(): v for k, v in max_units_tiers.items()}
    return tiers.get(effort, max_units)


class BlPrioritizeAndCap(DuoBaseTool):
    name: str = "bl_prioritize_and_cap"
    description: str = (
        "Order a discovery/finding list by severity (opt-in) and cap it to the "
        "resolved effort-tier or default max_units, BEFORE fan-out. Same "
        "elements, same count -- only the order and the cut-point change. "
        "Deterministic, no LLM call."
    )
    args_schema: Type[BaseModel] = BlPrioritizeAndCapInput
    # Below 1.0.0 so ListTools does not publish it: it only runs inside
    # the BL security flow.
    tool_version: ClassVar[Version] = Version("0.1.0")
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    truncation_config: TruncationConfig = _NO_TRUNCATION

    def format_display_message(
        self, args: Any, tool_response: Any = None
    ) -> Optional[str]:
        # The default dumps every arg, including the full pre-cap `units`
        # list, into the chat-log `content` field uncapped.
        if isinstance(tool_response, dict):
            return (
                f"Prioritized {tool_response.get('dispatched_count')}/"
                f"{tool_response.get('emitted_count')} units"
            )
        units = getattr(args, "units", None)
        count = len(units) if isinstance(units, list) else "?"
        return f"Prioritizing {count} units"

    async def _execute(
        self,
        units: List[Any],
        order_by_severity: bool = False,
        max_units: Optional[int] = None,
        max_units_tiers: Optional[dict] = None,
        scan_effort: Optional[str] = None,
    ) -> dict:
        ordered = order_by_priority(units) if order_by_severity else list(units)
        effective_cap = resolve_effective_max_units(
            max_units=max_units,
            max_units_tiers=max_units_tiers,
            scan_effort=scan_effort,
        )
        capped = ordered[:effective_cap] if effective_cap is not None else ordered
        return {
            "units": capped,
            "emitted_count": len(units),
            "dispatched_count": len(capped),
            "effective_max_units": effective_cap,
        }
