import asyncio
import json

import pytest
from pydantic import ValidationError

from duo_workflow_service.tools.bl_prioritize_and_cap import (
    BlPrioritizeAndCap,
    BlPrioritizeAndCapInput,
    resolve_effective_max_units,
)
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD


class TestResolveEffectiveMaxUnits:
    def test_no_tiers_returns_default(self):
        assert (
            resolve_effective_max_units(
                max_units=150, max_units_tiers=None, scan_effort="high"
            )
            == 150
        )

    def test_no_scan_effort_returns_default(self):
        assert (
            resolve_effective_max_units(
                max_units=150, max_units_tiers={"high": 400}, scan_effort=None
            )
            == 150
        )

    def test_unrecognized_tier_falls_back_to_default(self):
        assert (
            resolve_effective_max_units(
                max_units=150, max_units_tiers={"high": 400}, scan_effort="bogus"
            )
            == 150
        )

    def test_recognized_tier_wins(self):
        assert (
            resolve_effective_max_units(
                max_units=150,
                max_units_tiers={"low": 62, "standard": 150, "high": 400},
                scan_effort="HIGH",
            )
            == 400
        )


class TestBlPrioritizeAndCap:
    @pytest.mark.asyncio
    async def test_default_keeps_arrival_order_and_count(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = await tool._execute(units=units)
        assert result["units"] == units
        assert result["emitted_count"] == 3
        assert result["dispatched_count"] == 3
        assert result["effective_max_units"] is None

    @pytest.mark.asyncio
    async def test_flag_off_keeps_arrival_order_even_for_units_with_severity(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [
            {"id": "low", "severity": "low"},
            {"id": "crit", "severity": "critical"},
        ]
        result = await tool._execute(units=units, max_units=1)
        assert result["units"] == [{"id": "low", "severity": "low"}]

    @pytest.mark.asyncio
    async def test_severity_ordering_reorders_but_keeps_same_elements(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [
            {"id": "low", "severity": "low"},
            {"id": "crit", "severity": "critical"},
            {"id": "med", "severity": "medium"},
        ]
        result = await tool._execute(units=units, order_by_severity=True)
        assert result["units"][0]["id"] == "crit"
        assert {u["id"] for u in result["units"]} == {"low", "crit", "med"}
        assert result["dispatched_count"] == 3

    @pytest.mark.asyncio
    async def test_severity_order_before_cap_drops_least_severe(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [
            {"id": "low", "severity": "low"},
            {"id": "crit", "severity": "critical"},
        ]
        result = await tool._execute(units=units, order_by_severity=True, max_units=1)
        assert result["units"] == [{"id": "crit", "severity": "critical"}]
        assert result["emitted_count"] == 2
        assert result["dispatched_count"] == 1

    @pytest.mark.asyncio
    async def test_effort_tier_cap_used_over_default(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [{"id": i} for i in range(5)]
        result = await tool._execute(
            units=units,
            max_units=150,
            max_units_tiers={"low": 2, "standard": 150},
            scan_effort="low",
        )
        assert result["dispatched_count"] == 2
        assert result["effective_max_units"] == 2

    @pytest.mark.asyncio
    async def test_no_cap_when_max_units_none(self):
        tool = BlPrioritizeAndCap(metadata={})
        units = [{"id": i} for i in range(10)]
        result = await tool._execute(units=units)
        assert result["dispatched_count"] == 10


class TestBlPrioritizeAndCapInputJsonStringCoercion:
    """`literal: true` flow inputs always arrive as plain strings; there is no way to author a native mapping/list
    through that mechanism.

    `units` and `max_units_tiers` must accept a JSON string as well as the native value.
    """

    def test_max_units_tiers_json_string_is_parsed(self):
        parsed = BlPrioritizeAndCapInput(
            units=[{"id": 1}],
            max_units_tiers='{"low": 62, "standard": 150, "high": 400}',
        )
        assert parsed.max_units_tiers == {"low": 62, "standard": 150, "high": 400}

    def test_units_json_string_is_parsed(self):
        parsed = BlPrioritizeAndCapInput(units='[{"id": 1}, {"id": 2}]')
        assert parsed.units == [{"id": 1}, {"id": 2}]

    def test_native_dict_and_list_pass_through_unchanged(self):
        parsed = BlPrioritizeAndCapInput(
            units=[{"id": 1}],
            max_units_tiers={"low": 62},
        )
        assert parsed.units == [{"id": 1}]
        assert parsed.max_units_tiers == {"low": 62}

    def test_the_caller_s_input_mapping_is_not_mutated(self):
        data = {"units": '[{"id": 1}]'}
        BlPrioritizeAndCapInput.model_validate(data)
        assert data == {"units": '[{"id": 1}]'}

    def test_invalid_json_string_left_for_pydantic_to_reject(self):
        with pytest.raises(Exception):
            BlPrioritizeAndCapInput(units=[{"id": 1}], max_units_tiers="not json")

    def test_a_non_mapping_input_passes_through_untouched(self):
        """The coercion only rewrites the keys of a mapping.

        A `mode="before"`
        validator also sees non-dict inputs -- pydantic hands it the raw object
        when a model is validated from attributes rather than from kwargs -- and
        those must pass straight through, not hit `.get` on a non-dict.
        """

        class _UnitsHolder:
            units = [{"id": 1}, {"id": 2}]
            order_by_severity = True
            max_units = 1
            max_units_tiers = None
            scan_effort = None

        parsed = BlPrioritizeAndCapInput.model_validate(
            _UnitsHolder(), from_attributes=True
        )

        assert parsed.units == [{"id": 1}, {"id": 2}]
        assert parsed.order_by_severity is True
        assert parsed.max_units == 1


class TestFormatDisplayMessage:
    """The default DuoBaseTool.format_display_message dumps every arg's str() into the chat-log `content` field with no
    size cap -- unlike tool_info.tool_response, which IS capped.

    `units` here is the full
    pre-cap discovery list and can be hundreds of KB; this override must
    never let it reach `content`.
    """

    def test_uses_response_counts_when_available(self):
        tool = BlPrioritizeAndCap()
        args = BlPrioritizeAndCapInput(units=[{"id": i} for i in range(500)])
        message = tool.format_display_message(
            args, {"dispatched_count": 150, "emitted_count": 500}
        )
        assert message == "Prioritized 150/500 units"
        assert "id" not in message

    def test_falls_back_to_arg_count_without_response(self):
        tool = BlPrioritizeAndCap()
        args = BlPrioritizeAndCapInput(units=[{"id": i} for i in range(3)])
        message = tool.format_display_message(args, None)
        assert message == "Prioritizing 3 units"


def test_the_tool_is_hidden_from_list_tools():
    # ListTools publishes only tools at or above STABLE_VERSION_THRESHOLD.
    assert BlPrioritizeAndCap.tool_version < STABLE_VERSION_THRESHOLD


def test_a_large_output_stays_a_dict():
    # 150 findings of ~1.4 KB pass the 200 KiB display cap, which would
    # otherwise turn the result into a string that for_each cannot read.
    units = [{"id": i, "severity": "high", "body": "x" * 1400} for i in range(150)]
    out = asyncio.run(BlPrioritizeAndCap()._arun(units=units, max_units=150))
    assert len(json.dumps(out)) > 200 * 1024
    assert isinstance(out, dict)
    assert len(out["units"]) == 150


class TestCapValues:
    @pytest.mark.parametrize(
        "field", [{"max_units": -1}, {"max_units_tiers": {"low": -1}}]
    )
    def test_a_negative_cap_is_rejected(self, field):
        with pytest.raises(ValidationError):
            BlPrioritizeAndCapInput(units=[{"id": 1}], **field)

    @pytest.mark.asyncio
    async def test_a_string_tier_value_caps_as_a_number(self):
        out = await BlPrioritizeAndCap().ainvoke(
            {
                "units": [{"id": 1}, {"id": 2}, {"id": 3}],
                "max_units_tiers": '{"low": "2"}',
                "scan_effort": "low",
            }
        )
        assert out["dispatched_count"] == 2

    def test_a_tier_key_matches_case_insensitively(self):
        assert (
            resolve_effective_max_units(
                max_units=150, max_units_tiers={"Low": 62}, scan_effort="low"
            )
            == 62
        )
