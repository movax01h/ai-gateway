"""Unit tests for severity-first ordering of BL findings.

Invariants: ordering is a permutation (nothing is dropped) and no input value
raises; unrecognised values land in the neutral buckets.
"""

import json

import pytest

from duo_workflow_service.bl_security.priority import (
    NEUTRAL_SEVERITY,
    NEUTRAL_TIER,
    SEVERITY_ORDER,
    normalize_severity,
    normalize_tier,
    order_by_priority,
    priority_key,
    severity_rank,
)


def _finding(severity=None, tier=None, **extra):
    """A finding-shaped dict; keys absent unless the test supplies them."""
    out: dict = dict(extra)
    if severity is not None:
        out["severity"] = severity
    if tier is not None:
        out["tier"] = tier
    return out


# --------------------------------------------------------------------------- #
# Severity normalization
# --------------------------------------------------------------------------- #
class TestNormalizeSeverity:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("critical", "critical"),
            ("high", "high"),
            ("medium", "medium"),
            ("low", "low"),
            ("info", "info"),
            ("crit", "critical"),
            ("med", "medium"),
            ("moderate", "medium"),
            ("informational", "info"),
        ],
    )
    def test_declared_words_pass_through(self, raw, expected):
        assert normalize_severity(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HIGH", "high"),
            ("Critical", "critical"),
            ("  LoW  ", "low"),
            ("MeDiUm", "medium"),
            ("INFORMATIONAL", "info"),
        ],
    )
    def test_case_and_whitespace_are_irrelevant(self, raw, expected):
        """The producers emit mixed case; a case-sensitive map would quietly neutralise every ``HIGH`` and hand the
        budget to the wrong findings."""
        assert normalize_severity(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "sev-9", "P1", 3, 2.5, True, [], {}, object()],
    )
    def test_unrecognised_values_become_neutral_and_never_raise(self, raw):
        assert normalize_severity(raw) == NEUTRAL_SEVERITY

    @pytest.mark.parametrize("raw", ["unknown", "UNKNOWN", "none", "n/a", ""])
    def test_placeholder_severities_are_neutral_not_below_info(self, raw):
        """A severity nobody assessed sorts with a missing one, not beneath a real ``info``."""
        assert normalize_severity(raw) == NEUTRAL_SEVERITY
        assert severity_rank(raw) == severity_rank(None) < severity_rank("info")

    def test_neutral_bucket_is_mid_scale_not_an_extreme(self):
        ranks = [severity_rank(name) for name in SEVERITY_ORDER]
        assert ranks == sorted(ranks), "SEVERITY_ORDER must run most-severe-first"
        assert min(ranks) < severity_rank(NEUTRAL_SEVERITY) < max(ranks)

    def test_severity_rank_orders_critical_before_low(self):
        assert severity_rank("CRITICAL") < severity_rank("high")
        assert severity_rank("high") < severity_rank("low")
        assert severity_rank("low") < severity_rank("info")


# --------------------------------------------------------------------------- #
# Tier normalization -- the schema-inconsistent field
# --------------------------------------------------------------------------- #
class TestNormalizeTier:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (1, 1),  # the integer form one producer declares
            (2, 2),
            (3, 3),
            ("1", 1),  # string-numeric
            (" 2 ", 2),
            ("T2", 2),
            ("t3", 3),
            ("tier 3", 3),
            ("Tier2", 2),
            (2.0, 2),  # a JSON number that round-tripped through a float
            (1.5, 1),  # a finite non-integral float truncates, as int() does
        ],
    )
    def test_every_observed_well_formed_shape_normalises(self, raw, expected):
        assert normalize_tier(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0, 1),
            (-1, 1),
            ("-1", 1),
            ("-2", 1),  # a numeric string keeps its sign, like the int -2
            ("-5", 1),
            ("T-1", 1),  # after a label the "-" is a separator
            ("tier -3", 3),
            (-7.5, 1),  # truncates to -7, then clamps
            (4, 3),
            (99, 3),
            ("99", 3),
            ("T99", 3),
            (3.9, 3),
        ],
    )
    def test_out_of_range_tiers_clamp_into_one_to_three(self, raw, expected):
        """A stray ``0`` or ``-1`` must not outrank every real tier-1 finding, nor ``99`` sink below tier 3."""
        assert normalize_tier(raw) == expected

    def test_the_neutral_tier_is_inside_the_clamped_range(self):
        assert normalize_tier(1) < NEUTRAL_TIER < normalize_tier(3)

    @pytest.mark.parametrize(
        "raw",
        [
            "HIGH",  # a SEVERITY word leaking into the tier field
            "critical",
            None,
            "",
            "   ",
            "T",
            "tier",
            "abc",
            True,  # bool is an int subclass, but not a tier
            False,
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_unrecognised_values_become_neutral_and_never_raise(self, raw):
        assert normalize_tier(raw) == NEUTRAL_TIER

    def test_a_severity_word_in_the_tier_field_does_not_imply_a_tier(self):
        """``'HIGH'`` says nothing about tier, so it must not be silently read as one.

        Severity is the primary sort key in its own right.
        """
        assert normalize_tier("HIGH") == normalize_tier(None) == NEUTRAL_TIER

    def test_mixed_int_and_string_tiers_are_mutually_comparable(self):
        """``sorted`` over the raw values raises ``TypeError``; normalized tiers are all ints."""
        raw = [2, "T2", "T3", "HIGH", "1", None]
        normalized = [normalize_tier(v) for v in raw]

        assert all(isinstance(v, int) for v in normalized)
        assert sorted(normalized) == sorted(normalized)  # comparison is total
        with pytest.raises(TypeError):
            sorted(raw, key=lambda v: v)


# --------------------------------------------------------------------------- #
# The sort key
# --------------------------------------------------------------------------- #
class TestPriorityKey:
    def test_key_is_all_integers(self):
        key = priority_key(_finding(severity="HIGH", tier="T2"), 7)
        assert key == (severity_rank("high"), 2, 7)
        assert all(isinstance(part, int) for part in key)

    def test_severity_outranks_tier(self):
        critical_tier3 = priority_key(_finding(severity="critical", tier=3), 0)
        low_tier1 = priority_key(_finding(severity="low", tier=1), 1)
        assert critical_tier3 < low_tier1

    def test_tier_breaks_a_severity_tie(self):
        assert priority_key(_finding(severity="high", tier=1), 5) < priority_key(
            _finding(severity="high", tier=3), 0
        )

    def test_arrival_index_breaks_a_full_tie(self):
        a = priority_key(_finding(severity="high", tier=1), 4)
        b = priority_key(_finding(severity="high", tier=1), 9)
        assert a < b

    def test_a_json_string_unit_is_parsed_not_neutralised(self):
        """A unit can arrive as an unparsed JSON string; reading it as opaque would neutralise every severity and make
        the ordering a no-op."""
        obj = _finding(severity="critical", tier=1)
        assert priority_key(json.dumps(obj), 3) == priority_key(obj, 3)

    @pytest.mark.parametrize("unit", ["not json at all", "[1,2]", 42, None])
    def test_non_dict_units_fall_back_to_the_neutral_buckets(self, unit):
        assert priority_key(unit, 2) == (
            severity_rank(NEUTRAL_SEVERITY),
            NEUTRAL_TIER,
            2,
        )


# --------------------------------------------------------------------------- #
# The ordering itself
# --------------------------------------------------------------------------- #
class TestOrderByPriority:
    def test_a_late_critical_beats_an_early_low(self):
        """With a cap of 2, arrival order would keep two ``low`` findings and discard the ``critical``."""
        units = [
            _finding(severity="low", tier=3, file="svc/alpha.rb"),
            _finding(severity="low", tier=3, file="svc/beta.rb"),
            _finding(severity="critical", tier=1, file="svc/gamma.rb"),
        ]

        assert [u["file"] for u in units[:2]] == ["svc/alpha.rb", "svc/beta.rb"]

        ordered = order_by_priority(units)
        assert ordered[0]["file"] == "svc/gamma.rb"
        assert [u["file"] for u in ordered[:2]] == ["svc/gamma.rb", "svc/alpha.rb"]

    def test_full_severity_ladder(self):
        units = [
            _finding(severity=name.upper(), tier=2, file=f"svc/{name}.rb")
            for name in reversed(SEVERITY_ORDER)
        ]
        ordered = order_by_priority(units)
        assert [u["file"] for u in ordered] == [
            f"svc/{name}.rb" for name in SEVERITY_ORDER
        ]

    def test_it_is_a_permutation_nothing_is_added_or_dropped(self):
        units = [
            _finding(severity="critical", tier=1, file="svc/a.rb"),
            _finding(severity="bogus", tier="T9", file="svc/b.rb"),
            "a bare string unit",
            _finding(file="svc/c.rb"),
        ]
        ordered = order_by_priority(units)

        assert len(ordered) == len(units)
        assert all(any(u is o for o in ordered) for u in units)

    def test_a_malformed_tier_is_neither_dropped_nor_fatal(self):
        """Every malformed-tier finding survives, in a defined position."""
        units = [
            _finding(severity="high", tier="T2", file="svc/a.rb"),
            _finding(severity="high", tier="HIGH", file="svc/b.rb"),
            _finding(severity="high", tier=None, file="svc/c.rb"),
            _finding(severity="high", tier=[1], file="svc/d.rb"),
            _finding(severity="high", file="svc/e.rb"),
        ]

        ordered = order_by_priority(units)

        assert {u["file"] for u in ordered} == {
            "svc/a.rb",
            "svc/b.rb",
            "svc/c.rb",
            "svc/d.rb",
            "svc/e.rb",
        }
        # tier "T2" == 2 == the neutral bucket, so all five tie and arrival
        # order is preserved among them.
        assert [u["file"] for u in ordered] == [u["file"] for u in units]

    def test_out_of_range_tiers_order_as_the_nearest_real_tier(self):
        """Clamped tiers tie with the real bound, so arrival order decides among them."""
        units = [
            _finding(severity="high", tier=1, file="svc/t1.rb"),
            _finding(severity="high", tier=99, file="svc/t99.rb"),
            _finding(severity="high", tier=-1, file="svc/tneg.rb"),
            _finding(severity="high", tier=3, file="svc/t3.rb"),
            _finding(severity="high", tier=0, file="svc/t0.rb"),
            _finding(severity="high", tier=2, file="svc/t2.rb"),
        ]
        ordered = order_by_priority(units)
        assert [u["file"] for u in ordered] == [
            "svc/t1.rb",
            "svc/tneg.rb",
            "svc/t0.rb",
            "svc/t2.rb",
            "svc/t99.rb",
            "svc/t3.rb",
        ]

    def test_a_malformed_severity_does_not_sink_below_a_real_low(self):
        units = [
            _finding(severity="low", tier=1, file="svc/low.rb"),
            _finding(severity="???", tier=1, file="svc/garbled.rb"),
            _finding(severity="critical", tier=1, file="svc/crit.rb"),
        ]
        ordered = order_by_priority(units)
        assert [u["file"] for u in ordered] == [
            "svc/crit.rb",
            "svc/garbled.rb",
            "svc/low.rb",
        ]

    def test_output_is_deterministic_across_repeated_runs(self):
        """An unstable order would make the capped set vary between runs."""
        units = [
            _finding(severity=sev, tier=tier, file=f"svc/f{i}.rb")
            for i, (sev, tier) in enumerate(
                [
                    ("high", 2),
                    ("high", "T2"),
                    ("critical", 3),
                    ("bogus", None),
                    ("high", 1),
                    ("low", "T1"),
                    ("critical", "1"),
                ]
            )
        ]
        runs = [[u["file"] for u in order_by_priority(units)] for _ in range(20)]
        assert all(run == runs[0] for run in runs)

    def test_equal_priority_preserves_arrival_order(self):
        units = [
            _finding(severity="high", tier=2, file=f"svc/f{i}.rb") for i in range(6)
        ]
        assert order_by_priority(units) == units

    def test_empty_list(self):
        assert order_by_priority([]) == []
