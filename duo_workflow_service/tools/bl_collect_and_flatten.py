"""Post-collect aggregation for a `for_each`-driven BL Security Analyzer stage.

Publishes two keys:

- `final_answer`: the JSON array of the finding dicts of every readable unit
    answer, flattened in unit order.
- `coverage`: the per-stage counts (emitted/dispatched/completed/errored)
    plus the human-readable `summary`/`truncation`/`loss` sentences, so
    a reader that renders sentences has something to render on a fully covered
    run as well as on a lossy one.
"""

import json
from typing import Any, ClassVar, List, Optional, Type

import structlog
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig

__all__ = ["BlCollectAndFlatten", "BlCollectAndFlattenInput"]

_log = structlog.stdlib.get_logger(__name__)

# Later flow steps parse the return value, and the triage step hands findings to
# an agent prompt. Generic display truncation would reshape the dict into a
# truncated string and break the `{final_answer, coverage}` contract.
_NO_TRUNCATION = TruncationConfig(
    max_bytes=64 * 1024 * 1024, truncated_size=64 * 1024 * 1024
)

# The fields a triage verdict adds to the finding it judges.
_VERDICT_FIELDS = ("verdict", "clause", "evidence", "triage_evidence")

# The key `for_each` puts in a unit's results entry instead of an answer when the
# unit raised. Mirrors `ITEM_ERROR_SUBKEY` in
# experimental/components/for_each/errors.py, spelled out here so a tool does not
# import the component package.
_UNIT_ERROR_KEY = "for_each_error"


def _unit_failed(entry: Any) -> bool:
    """Whether ``for_each`` recorded this unit as failed rather than answered."""
    return isinstance(entry, dict) and _UNIT_ERROR_KEY in entry


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _unit_findings(answer: Any) -> Optional[list]:
    """The finding dicts of one unit answer: the dicts under its ``findings`` list.

    A unit answers through a response-schema tool call, stored as ``json.dumps`` of ``{"reasoning": ...,
    "findings": [...]}``, so a string answer is decoded as strict JSON. ``None`` when the answer is not such an
    object, or when its ``findings`` list is non-empty but holds no finding dict. ``[]`` for an honest empty answer.
    """
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return None
    if not isinstance(answer, dict) or not isinstance(answer.get("findings"), list):
        return None
    items = answer["findings"]
    found = [it for it in items if isinstance(it, dict)]
    return found if found or not items else None


def _merge_verdict(answer: Any, item: Any) -> Any:
    """A triage verdict merged onto the finding it judged: ``{"reasoning": ..., "findings": [finding + verdict]}``.

    ``item`` is the unit's own fan-out item, the finding as triage received it. Verdict fields left unset are not
    merged. An answer that is not a structured object, or has no ``verdict``, is returned unchanged.
    """
    verdict = _maybe_json(answer)
    if not isinstance(verdict, dict) or "verdict" not in verdict:
        return answer
    finding = _maybe_json(item)
    if not isinstance(finding, dict):
        _log.warning(
            "bl_collect_and_flatten triage item is not a finding; emitting the verdict alone"
        )
        finding = {}
    fields = {k: verdict[k] for k in _VERDICT_FIELDS if verdict.get(k) is not None}
    return {
        "reasoning": verdict.get("reasoning", ""),
        "findings": [{**finding, **fields}],
    }


def _coverage_sentences(
    *,
    noun: str,
    emitted: Optional[int],
    dispatched: int,
    completed: int,
    errored: int,
    unread: int = 0,
) -> dict:
    """Human-readable coverage sentences for the stage: `summary`, and `truncation`/`loss` when they apply.

    `truncation` fires only when the pre-cap `emitted` exceeded what was
    dispatched -- a cap that BOUND is a choice worth disclosing. `loss` fires
    when units were discarded unread, which is a defect, and is kept separate
    so the two never read alike.
    """
    out: dict = {
        "summary": (
            f"Reviewed {completed - unread} of {dispatched} {noun}."
            if dispatched
            else f"No {noun} were dispatched for this stage."
        )
    }
    if emitted is not None and emitted > dispatched:
        out["truncation"] = (
            f"A cap bound this stage: {dispatched} of {emitted} {noun} were "
            f"dispatched; {emitted - dispatched} were never reviewed."
        )
    lost = errored + unread
    if lost:
        parts = []
        if unread:
            parts.append(f"{unread} completed but their findings could not be read")
        if errored:
            parts.append(f"{errored} errored")
        out["loss"] = (
            f"{lost} of {dispatched} {noun} were DISCARDED without their "
            f"findings being counted ({'; '.join(parts)})."
        )
    return out


class BlCollectAndFlattenInput(BaseModel):
    results: List[Any] = Field(
        description="`for_each`'s ordered per-unit result entries. A unit that "
        "ran holds its agent's `final_answer` (a dict, or a JSON string); a unit "
        "that failed holds a `for_each_error` record instead."
    )
    emitted_count: Optional[int] = Field(
        default=None,
        description="The pre-cap unit count from `bl_prioritize_and_cap`, for "
        "the coverage summary.",
    )
    verdict_items: Optional[List[Any]] = Field(
        default=None,
        description="For a triage stage: the fan-out's items (the findings "
        "judged), in order. Each completed unit's verdict is merged onto its "
        "own item.",
    )
    unit_noun: Optional[str] = Field(
        default=None,
        description="What this stage's units are (e.g. 'findings' for a stage "
        "that fans out over findings), for the coverage sentence. Defaults to "
        "'units'.",
    )


class BlCollectAndFlatten(DuoBaseTool):
    name: str = "bl_collect_and_flatten"
    description: str = (
        "Read each unit's `final_answer` from the fan-out's collected per-unit "
        "results, count the units that carry a `for_each_error` record as "
        "errored, and assemble the aggregate `final_answer` plus per-stage "
        "coverage. Deterministic, no LLM call."
    )
    args_schema: Type[BaseModel] = BlCollectAndFlattenInput
    # Below 1.0.0 so ListTools does not publish it: it only runs inside
    # the BL security flow.
    tool_version: ClassVar[Version] = Version("0.1.0")
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    truncation_config: TruncationConfig = _NO_TRUNCATION

    def format_display_message(
        self, args: Any, _tool_response: Any = None
    ) -> Optional[str]:
        # DuoBaseTool's default dumps every arg's str() into the chat-log
        # `content` field with no size cap (unlike tool_info.tool_response,
        # which is capped at TOOL_RESPONSE_MAX_DISPLAY_MSG). `results` is the
        # full for_each results list, which can push a checkpoint delta over
        # the gRPC transport cap.
        results = getattr(args, "results", None)
        count = len(results) if isinstance(results, list) else "?"
        return f"Collecting and flattening {count} unit results"

    @staticmethod
    def _resolve_entry(entry: Any) -> Any:
        """A completed unit's ``final_answer``; ``None`` for a failed unit or a non-dict entry."""
        if not isinstance(entry, dict) or _unit_failed(entry):
            return None
        return entry.get("final_answer")

    async def _execute(
        self,
        results: List[Any],
        emitted_count: Optional[int] = None,
        unit_noun: Optional[str] = None,
        verdict_items: Optional[List[Any]] = None,
    ) -> dict:
        successes = [
            (i, r)
            for i, r in enumerate(results)
            if isinstance(r, dict) and not _unit_failed(r)
        ]
        resolved = [(i, self._resolve_entry(r)) for i, r in successes]
        answered = [(i, answer) for i, answer in resolved if answer is not None]
        if verdict_items is not None:
            answered = [
                (
                    i,
                    _merge_verdict(
                        a, verdict_items[i] if i < len(verdict_items) else None
                    ),
                )
                for i, a in answered
            ]
        # A unit answer must hold a `findings` list (a dict, or strict JSON); one that
        # does not is counted unread, not reviewed.
        read = [(i, _unit_findings(a)) for i, a in answered]
        garbled = [i for i, found in read if found is None]
        if garbled:
            _log.warning(
                "bl_collect_and_flatten read unit answers that hold no readable findings",
                unit_indexes=garbled,
            )
        flat = [f for _, found in read for f in found or []]
        final_answer = json.dumps(flat, ensure_ascii=False)

        errored = sum(1 for r in results if _unit_failed(r))

        return {
            "final_answer": final_answer,
            "coverage": {
                "emitted": emitted_count,
                "dispatched": len(results),
                "completed": len(successes),
                "errored": errored,
                "unit_noun": unit_noun,
                **_coverage_sentences(
                    noun=unit_noun or "units",
                    emitted=emitted_count,
                    dispatched=len(results),
                    completed=len(successes),
                    errored=errored,
                    unread=len(successes) - len(answered) + len(garbled),
                ),
            },
        }
