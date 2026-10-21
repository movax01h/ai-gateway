"""Unit tests for the pure BL discovery selection logic (``duo_workflow_service.bl_security.units``)."""

import collections

import pytest
import structlog

import duo_workflow_service.bl_security.units as bl_units
from duo_workflow_service.bl_security.units import (
    cluster,
    enumerate_candidates,
    handler_coverage_units,
    score,
    select_units,
)


# --------------------------------------------------------------------------- #
# _is_noise (via enumerate_candidates) + enumerate_candidates normalization
# --------------------------------------------------------------------------- #
class TestEnumerateCandidates:
    def test_empty_and_blank_paths_dropped(self):
        assert enumerate_candidates(["", "   "]) == []

    def test_lockfiles_and_noise_dirs_and_suffixes_dropped(self):
        groups = [
            "app/controllers/users_controller.rb\n"
            "spec/users_spec.rb\n"
            "app/models/user_spec.rb\n"
            "docs/readme.md\n"
            "app/assets/app.js\n"
            "yarn.lock\n"
            "vendor/lib/thing.rb\n"
            "app/models/user.png",
        ]
        out = enumerate_candidates(groups)
        # only the real controller survives the noise filter
        assert out == ["app/controllers/users_controller.rb"]

    def test_accepts_list_set_scalar_and_none_groups(self):
        groups = [
            None,
            ["app/models/user.rb"],
            {"app/policies/user_policy.rb"},
            "app/controllers/user_controller.rb",
            12345,  # scalar coerced via str()
        ]
        out = enumerate_candidates(groups)
        # sorted, de-duplicated union across heterogeneous group shapes
        assert out == sorted(out)
        assert "app/models/user.rb" in out
        assert "app/policies/user_policy.rb" in out
        assert "app/controllers/user_controller.rb" in out
        assert "12345" in out

    def test_deduplicates_repeated_paths(self):
        groups = [["a/x_controller.rb"], ["a/x_controller.rb"]]
        assert enumerate_candidates(groups) == ["a/x_controller.rb"]


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
class TestScore:
    def test_counts_distinct_security_keywords(self):
        unit = {"files": ["app/controllers/admin/users_controller.rb"]}
        # 'admin' + 'user' -> 2 (each keyword counted once)
        assert score(unit) == 2

    def test_zero_when_no_keywords(self):
        assert score({"files": ["lib/zzz.rb"]}) == 0

    def test_missing_files_key_is_zero(self):
        assert score({}) == 0


# --------------------------------------------------------------------------- #
# cluster (exercises _is_anchor / _dirname / _module_dir / _resource_tokens /
# _affinity / naming)
# --------------------------------------------------------------------------- #
class TestCluster:
    def test_anchor_becomes_unit_with_context_and_name(self):
        candidates = sorted(
            [
                "app/controllers/users_controller.rb",
                "app/models/user.rb",
                "app/policies/user_policy.rb",
                "app/models/order.rb",
            ]
        )
        units = cluster(candidates, files_per_unit=3)
        names = {u["name"] for u in units}
        assert "app.controllers.users_controller" in names
        unit = next(u for u in units if u["name"].endswith("users_controller"))
        # anchor is first, and its highest-affinity same-resource context joins it
        assert unit["files"][0] == "app/controllers/users_controller.rb"
        assert len(unit["files"]) <= 3
        assert "app/models/user.rb" in unit["files"]
        # the unrelated order model is NOT pulled into the users unit ahead of user.rb
        assert unit["files"][1] == "app/models/user.rb"

    def test_files_per_unit_floor_of_one(self):
        candidates = ["a/x_controller.rb", "a/x_helper.rb"]
        units = cluster(candidates, files_per_unit=0)  # floored to 1
        assert units[0]["files"] == ["a/x_controller.rb"]

    def test_camelcase_java_anchor_and_token_folding(self):
        candidates = sorted(
            [
                "src/UserController.java",
                "src/UserService.java",
                "src/UserResource.java",
            ]
        )
        units = cluster(candidates, files_per_unit=5)
        anchors = {u["files"][0] for u in units}
        # all three are anchors by the Java suffix rules
        assert "src/UserController.java" in anchors
        # UserController joins UserService/UserResource via shared 'user' token
        ctrl = next(u for u in units if u["files"][0] == "src/UserController.java")
        assert "src/UserService.java" in ctrl["files"]

    def test_no_anchor_yields_no_units(self):
        assert cluster(["lib/plain.txt"], files_per_unit=3) == []


# --------------------------------------------------------------------------- #
# Direct branch coverage for helpers short-circuited by callers
# --------------------------------------------------------------------------- #
class TestNoiseAndAffinityBranches:
    def test_is_noise_blank_path(self):
        # enumerate_candidates strips blanks before _is_noise sees them, so hit
        # the empty-guard directly.
        assert bl_units._is_noise("") is True
        assert bl_units._is_noise("   ") is True

    def test_same_module_dir_different_subdir_affinity_bonus(self):
        # anchor and a NON-anchor context share the top-2 module dir (src/app)
        # but sit in different full directories -> the module-dir affinity bonus
        # (rank += 40) fires even though the same-dir bonus does not.
        candidates = sorted(
            [
                "src/app/UserController.java",
                "src/app/helpers/user_helper.rb",
            ]
        )
        units = cluster(candidates, files_per_unit=2)
        ctrl = next(u for u in units if u["files"][0].endswith("UserController.java"))
        assert "src/app/helpers/user_helper.rb" in ctrl["files"]


# --------------------------------------------------------------------------- #
# RECALL COVERAGE — deterministic authz-handler seeding (guaranteed units)
# --------------------------------------------------------------------------- #
# Low-scoring authz handlers that must be reviewed even when a score-ranked
# selection would drop them past the max_units cap.
_GT_HANDLERS = [
    "routers/api/v1/org/member.go",
    "routers/api/v1/org/team.go",
    "routers/api/v1/user/star.go",
    "routers/api/v1/repo/pull.go",
    "routers/web/repo/view_home.go",
    "routers/web/auth/oauth2_provider.go",
]


def _forge_like_paths():
    # The 6 GT handlers + many HIGH-security-score decoy anchors (admin/auth/
    # token keyword-laden) that would win the score sort and push the low-score
    # GT handlers past a small max_units cap under a select-by-score-then-truncate
    # policy. Plus non-anchor context files.
    decoys = [f"routers/api/v1/admin/admin_auth_token_{i}.go" for i in range(30)]
    context = [
        "models/organization/team.go",
        "models/user/star.go",
        "services/pull/pull.go",
    ]
    return sorted(_GT_HANDLERS + decoys + context)


class TestHandlerCoverageSeeding:
    def test_every_anchor_handler_is_seeded_into_a_unit(self):
        cand = _forge_like_paths()
        anchors = [c for c in cand if bl_units._is_anchor(c)]
        units = handler_coverage_units(cand)
        seeded = {f for u in units for f in u["files"]}
        # EVERY anchor handler appears in exactly one guaranteed unit
        assert set(anchors) == seeded
        # each guaranteed unit is bounded to files_per_unit
        assert all(len(u["files"]) <= 8 for u in units)

    def test_low_score_gt_handlers_not_dropped_by_max_units_cap(self):
        # This is the recall-coverage regression guard: with a TINY max_units,
        # the score-sorted context tier alone would keep only a few high-score
        # decoy units and drop the low-score GT handlers. select_units must still
        # seed ALL of them via the guaranteed tier.
        cand = _forge_like_paths()
        units = select_units(cand, files_per_unit=2, max_units=3)
        reviewed = {f for u in units for f in u["files"]}
        for gt in _GT_HANDLERS:
            assert gt in reviewed, f"GT handler dropped by cap: {gt}"

    def test_focused_single_file_units_for_anchors_default(self):
        # EVERY anchor handler gets its OWN focused single-file unit; handlers
        # are NOT chunked with siblings.
        cand = _forge_like_paths()
        anchors = sorted(c for c in cand if bl_units._is_anchor(c))
        units = handler_coverage_units(cand)
        # one solo unit per anchor, each exactly its single anchor file
        assert len(units) == len(anchors)
        assert all(u["files"] == [a] for u, a in zip(units, anchors))
        for gt in _GT_HANDLERS:
            solo = [u for u in units if u["files"] == [gt]]
            assert len(solo) == 1, f"GT handler {gt} not a focused solo unit"

    def test_deterministic_across_runs(self):
        cand = _forge_like_paths()
        assert select_units(cand, 8, 150) == select_units(cand, 8, 150)

    def test_coverage_budget_scales_with_authz_surface(self):
        # More handlers -> strictly more guaranteed units (big repos cost more,
        # not weaker), and select_units' budget grows to fit them past max_units.
        small = [f"routers/api/v1/h{i}.go" for i in range(8)]
        big = [f"routers/api/v1/h{i}.go" for i in range(80)]
        assert len(handler_coverage_units(big)) > len(handler_coverage_units(small))
        # with a low max_units, the big surface still yields >max_units units
        sel_big = select_units(big, files_per_unit=8, max_units=3)
        assert len(sel_big) >= len(handler_coverage_units(big))


class TestMultiSampleAuthzUnits:
    """Each authz-anchor unit is emitted AUTHZ_SAMPLE_COUNT times so the review fan-out reviews it N times."""

    def test_multi_sample_helper_block_repeats_with_distinct_copies(self, monkeypatch):
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 3)
        units = [{"name": "a", "files": ["a.go"]}, {"name": "b", "files": ["b.go"]}]
        out = bl_units._multi_sample_authz_units(units)
        # block repetition: [all units] x N (graceful degradation under a cap)
        assert out == units * 3
        # each sample is a DISTINCT dict object (defensive copy, no aliasing)
        ids = [id(u) for u in out]
        assert len(ids) == len(set(ids))

    def test_each_authz_unit_emitted_sample_count_times(self, monkeypatch):
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 3)
        cand = _forge_like_paths()
        guaranteed = handler_coverage_units(cand)
        units = select_units(cand, files_per_unit=8, max_units=150)
        # every guaranteed authz unit appears EXACTLY AUTHZ_SAMPLE_COUNT times
        for g in guaranteed:
            n = sum(1 for u in units if u["files"] == g["files"])
            assert n == 3, f"authz unit {g['files']} appeared {n}x, expected 3"
        # BLOCK repetition survives the tier interleave: sample 1 of EVERY authz
        # unit precedes sample 2 of ANY of them, so a downstream fan-out cap
        # costs extra samples, never coverage. (Asserted on the authz
        # subsequence rather than on units[:n] -- the tiers are round-robin
        # interleaved so the anchor tier is no longer a contiguous prefix.)
        authz = [u for u in units if u["name"].startswith("authz_handlers.")]
        first_block = authz[: len(guaranteed)]
        assert {tuple(u["files"]) for u in first_block} == {
            tuple(g["files"]) for g in guaranteed
        }
        # the authz tier still leads the list overall
        assert units[0]["name"].startswith("authz_handlers.")

    def test_context_units_are_not_multiplied(self, monkeypatch):
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 3)
        cand = _forge_like_paths()
        guaranteed = handler_coverage_units(cand)
        units = select_units(cand, files_per_unit=8, max_units=150)
        # the score-ordered CONTEXT tier is emitted ONCE (cost bounded to authz).
        # Selected by NAME rather than by slice position: the tiers are round-robin
        # interleaved so the context tier is no longer a contiguous suffix.
        remainder = [
            u
            for u in units
            if not u["name"].startswith(("authz_handlers.", "semantic_"))
        ]
        files_seen = [tuple(u["files"]) for u in remainder]
        assert len(files_seen) == len(set(files_seen)), "context units were duplicated"
        # Non-vacuous + correct partition: a context tier must actually be left,
        # and the name filter must exclude EVERY guaranteed authz unit (those are
        # the ones deliberately repeated) -- otherwise the uniqueness assertion
        # above could pass while silently checking the multiplied tier.
        assert remainder, "no context units survived the name filter"
        guaranteed_files = {tuple(g["files"]) for g in guaranteed}
        assert guaranteed_files.isdisjoint(files_seen), (
            "guaranteed authz units leaked into the context tier"
        )

    def test_budget_accommodates_all_samples_past_tiny_max_units(self, monkeypatch):
        # Regression guard: the N authz samples must NEVER be truncated by
        # max_units (budget grows to fit them), else multi-sampling silently
        # degrades to fewer samples for low-effort presets.
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 3)
        cand = _forge_like_paths()
        guaranteed = handler_coverage_units(cand)
        units = select_units(cand, files_per_unit=8, max_units=3)  # tiny cap
        assert len(units) >= 3 * len(guaranteed)
        for g in guaranteed:
            assert sum(1 for u in units if u["files"] == g["files"]) == 3

    def test_sample_count_one_disables_sampling(self, monkeypatch):
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 1)
        cand = _forge_like_paths()
        guaranteed = handler_coverage_units(cand)
        units = select_units(cand, files_per_unit=8, max_units=150)
        for g in guaranteed:
            assert sum(1 for u in units if u["files"] == g["files"]) == 1


# --------------------------------------------------------------------------- #
# Language-agnostic semantic surface (_role_of / semantic_surface_units) and the
# per-tier budget reservation in select_units.
# --------------------------------------------------------------------------- #
class TestSemanticSurface:
    def test_semantic_units_skip_anchors_and_split_budget_across_roles(self):
        candidates = [
            "app/controllers/orders_controller.rb",  # anchor -> excluded
            "app/models/order.rb",
            "app/models/user.rb",
            "app/services/checkout.rb",
            "app/policies/order_policy.rb",
        ]
        units = bl_units.semantic_surface_units(candidates, files_per_unit=1, budget=30)
        names = {u["name"].split(".", 1)[0] for u in units}
        assert names == {"semantic_state", "semantic_logic", "semantic_access"}
        covered = {f for u in units for f in u["files"]}
        # the anchor is already guaranteed by handler_coverage_units -> not here
        assert "app/controllers/orders_controller.rb" not in covered
        assert covered == {
            "app/models/order.rb",
            "app/models/user.rb",
            "app/services/checkout.rb",
            "app/policies/order_policy.rb",
        }

    def test_semantic_units_zero_budget_is_empty(self):
        assert (
            bl_units.semantic_surface_units(["app/models/user.rb"], 2, budget=0) == []
        )

    def test_semantic_units_are_deterministic(self):
        candidates = ["app/models/%02d_user.rb" % i for i in range(20)]
        a = bl_units.semantic_surface_units(candidates, 3, budget=9)
        b = bl_units.semantic_surface_units(candidates, 3, budget=9)
        assert a == b

    def test_context_tier_is_never_starved_by_the_guaranteed_tier(self):
        # When the anchor surface is larger than max_units, the context tier must
        # still get units from its own floor rather than zero.
        candidates = ["app/controllers/c%02d_controller.rb" % i for i in range(50)] + [
            "app/models/m%02d.rb" % i for i in range(50)
        ]
        units = bl_units.select_units(candidates, files_per_unit=2, max_units=1)
        context_units = [
            u
            for u in units
            if not u["name"].startswith(("authz_handlers.", "semantic_"))
        ]
        assert len(context_units) >= 1


# --------------------------------------------------------------------------- #
# Prefix-representative ordering (_weighted_round_robin / _spread_by_module) -- what makes
# selection survive the downstream review fan-out cap.
# --------------------------------------------------------------------------- #
class TestPrefixRepresentativeOrdering:
    @pytest.mark.parametrize(
        ("sizes", "weights", "expected"),
        [
            ((3, 1), (1, 1), "a0 b0 a1 a2"),
            ((2, 5), (1, 3), "a0 b0 b1 b2 a1 b3 b4"),
            ((1, 4), (1, 2), "a0 b0 b1 b2 b3"),
            ((2, 2), (1, 0), "a0 b0 a1 b1"),
            ((2, 2), (1, -100), "a0 b0 a1 b1"),
            ((0, 0), (1, 3), ""),
        ],
        ids=[
            "plain",
            "weighted",
            "exhausted",
            "zero-clamped",
            "negative-clamped",
            "empty",
        ],
    )
    def test_weighted_round_robin(self, sizes, weights, expected):
        # A weight sets a group's share per pass, never its membership; an
        # exhausted group is skipped and a weight below 1 counts as 1.
        groups = [
            [{"name": f"{g}{i}", "files": []} for i in range(n)]
            for g, n in zip("ab", sizes)
        ]
        out = bl_units._weighted_round_robin(groups, weights)
        assert [u["name"] for u in out] == expected.split()

    def test_spread_by_module_breaks_up_a_dominant_directory(self):
        units = [{"name": "u", "files": ["routers/api/f%02d.go" % i]} for i in range(5)]
        units += [
            {"name": "u", "files": ["services/mail/g%02d.go" % i]} for i in range(5)
        ]
        out = bl_units._spread_by_module(units)
        # the first two units come from DIFFERENT module dirs, so a cap of 5
        # does not review routers/api exclusively.
        heads = [bl_units._module_dir(u["files"][0]) for u in out[:2]]
        assert heads[0] != heads[1]
        assert len(out) == 10  # permutation, nothing dropped

    def test_selection_prefix_spans_multiple_namespaces(self):
        # A repo whose anchor surface exceeds the fan-out cap must still get a
        # prefix that spans the repo, not just its alphabetically-first namespace.
        candidates = sorted(
            ["routers/api/a%03d_handler.go" % i for i in range(200)]
            + ["routers/web/w%03d_handler.go" % i for i in range(50)]
            + ["services/billing/s%03d.go" % i for i in range(50)]
            + ["models/org/m%03d.go" % i for i in range(50)]
        )
        units = bl_units.select_units(candidates, files_per_unit=4, max_units=150)
        reviewed = {f for u in units[:150] for f in u["files"]}
        assert any(p.startswith("routers/api/") for p in reviewed)
        assert any(p.startswith("routers/web/") for p in reviewed)
        assert any(p.startswith("services/") for p in reviewed)
        assert any(p.startswith("models/") for p in reviewed)

    def test_selection_interleaves_tiers_rather_than_prepending_guaranteed(self):
        """The tiers are interleaved, not concatenated."""
        # Large guaranteed tier (100 anchors), small semantic/context tiers.
        candidates = sorted(
            ["routers/api/h%03d_handler.go" % i for i in range(100)]
            + ["models/org/m%02d.go" % i for i in range(3)]
            + ["services/billing/s%02d.go" % i for i in range(3)]
        )
        cap = 20  # stands in for the review fan-out cap
        units = bl_units.select_units(candidates, files_per_unit=3, max_units=cap)

        guaranteed = [u for u in units if u["name"].startswith("authz_handlers.")]
        assert len(guaranteed) > cap, "fixture must overflow the cap to be meaningful"

        prefix = units[:cap]
        tiers = {
            (
                "guaranteed"
                if u["name"].startswith("authz_handlers.")
                else "semantic"
                if u["name"].startswith("semantic_")
                else "context"
            )
            for u in prefix
        }
        # Under prepending this set would be exactly {"guaranteed"}.
        assert tiers == {"guaranteed", "semantic", "context"}
        # ...and the guaranteed tier still LEADS: round-robin costs it position 0
        # to nothing.
        assert units[0]["name"].startswith("authz_handlers.")
        # Round-robin is tight: all three surfaces show up in the first few
        # units, not merely somewhere inside the cap.
        assert len({u["name"].split(".", 1)[0] for u in units[:3]}) == 3


class TestTopLevelNoise:
    def test_top_level_asset_dirs_are_noise(self):
        # Repo-root asset dirs are noise, not only nested ones.
        for path in (
            "public/app.js",
            "build/gen.go",
            "static/x.js",
            "assets/y.js",
            "dist/z.js",
        ):
            assert bl_units._is_noise(path) is True, path
        # ...and they are still noise when nested
        assert bl_units._is_noise("web/public/app.js") is True

    def test_presentation_layer_is_noise(self):
        for path in ("templates/user/auth.tmpl", "web/index.html", "ui/app.vue"):
            assert bl_units._is_noise(path) is True, path

    def test_real_source_under_similar_names_is_kept(self):
        # guard against over-filtering: a package merely NAMED build/public is
        # only excluded as a whole directory segment
        assert bl_units._is_noise("services/rebuild/run.go") is False
        assert bl_units._is_noise("models/publication.go") is False


# --------------------------------------------------------------------------- #
# scan_effort tiers (discovery side)
# --------------------------------------------------------------------------- #
def _big_repo_paths():
    """A synthetic repo large enough that a small unit budget actually truncates."""
    anchors = ["routers/api/v1/org/member.go", "routers/api/v1/repo/pull.go"] + [
        f"routers/api/v1/admin/admin_auth_token_{i}.go" for i in range(60)
    ]
    context = (
        [
            f"models/{pkg}/thing{i}.go"
            for pkg in ("organization", "user", "issues", "repo", "team")
            for i in range(120)
        ]
        + [
            f"services/{pkg}/svc{i}.go"
            for pkg in ("pull", "repo", "org")
            for i in range(120)
        ]
        + [
            f"modules/{pkg}/mod{i}.go"
            for pkg in ("setting", "git", "web")
            for i in range(120)
        ]
    )
    return sorted(anchors + context)


def _anchor_heavy_repo_paths():
    """A repo whose ANCHOR surface is large enough to bind low's context budget."""
    anchors = ["routers/api/v1/org/member.go", "routers/api/v1/repo/pull.go"] + [
        f"routers/api/v1/admin/admin_auth_token_{i}.go" for i in range(200)
    ]
    context = [
        f"models/{pkg}/thing{i}.go"
        for pkg in ("organization", "user", "issues", "repo", "team")
        for i in range(120)
    ]
    return sorted(anchors + context)


def _reviewed_files(units):
    return {f for u in units for f in u["files"]}


class TestResolveScanEffort:
    """``scan_effort`` -> the tier; it shapes the discovery POOL, not coverage (the fan-out cap truncates the pool)."""

    def test_a_declared_tier_maps_to_its_row_in_any_case(self):
        for name, row in bl_units.SCAN_EFFORT_TIERS.items():
            assert bl_units.resolve_scan_effort(f"  {name.upper()} ", 8, 150) == row

    def test_standard_tier_is_the_default_budget(self):
        # Raising the default pool is a cost decision, not a side effect of
        # naming the default tier.
        assert bl_units.SCAN_EFFORT_TIERS["standard"][1] == bl_units.DEFAULT_MAX_UNITS

    def test_tiers_are_monotone_in_both_knobs(self):
        low, standard, high = (
            bl_units.SCAN_EFFORT_TIERS[k] for k in ("low", "standard", "high")
        )
        assert low[0] <= standard[0] <= high[0]
        assert low[1] <= standard[1] <= high[1]

    def test_tier_coverage_nests(self):
        # Coverage is monotone in max_units, the only pool knob the tiers vary,
        # so raising effort can only add coverage. (Not so for files_per_unit.)
        cand = _big_repo_paths()
        low, standard, high = (
            _reviewed_files(
                bl_units.select_units(cand, *bl_units.SCAN_EFFORT_TIERS[k][:2])
            )
            for k in ("low", "standard", "high")
        )
        assert low <= standard <= high

    def test_higher_tiers_buy_context_depth_once_the_budget_binds(self):
        cand = _anchor_heavy_repo_paths()
        low, standard, high = (
            len(bl_units.select_units(cand, *bl_units.SCAN_EFFORT_TIERS[k][:2]))
            for k in ("low", "standard", "high")
        )
        assert low < standard <= high


def test_context_unit_floor_scales_below_the_default_and_is_bounded():
    assert (
        bl_units.context_unit_floor(bl_units.DEFAULT_MAX_UNITS)
        == bl_units.CONTEXT_UNIT_FLOOR
    )
    assert bl_units.context_unit_floor(550) == bl_units.CONTEXT_UNIT_FLOOR
    assert (
        bl_units.context_unit_floor(40)
        < bl_units.context_unit_floor(80)
        < bl_units.CONTEXT_UNIT_FLOOR
    )
    assert bl_units.context_unit_floor(0) == bl_units.MIN_CONTEXT_UNIT_FLOOR
    assert bl_units.context_unit_floor(-5) == bl_units.MIN_CONTEXT_UNIT_FLOOR


def test_a_wider_files_per_unit_grows_each_anchors_unit_as_a_superset():
    # cluster()'s per-anchor promise. The REVIEWED set does not nest across
    # files_per_unit (the interleave moves), so compare it at equal file slots.
    cand = _big_repo_paths()
    by_fpu = {
        fpu: {u["name"]: set(u["files"]) for u in bl_units.cluster(cand, fpu)}
        for fpu in (4, 8, 16)
    }
    names = set(by_fpu[4]) | set(by_fpu[8]) | set(by_fpu[16])
    assert names
    for name in names:
        assert by_fpu[4].get(name, set()) <= by_fpu[8].get(name, set())
        assert by_fpu[8].get(name, set()) <= by_fpu[16].get(name, set())
    # SUPERSET, not equality: the slice really does track the knob.
    assert any(
        by_fpu[8].get(name, set()) < by_fpu[16].get(name, set()) for name in names
    )


# --------------------------------------------------------------------------- #
# Reservation before repetition (coverage_first)
# --------------------------------------------------------------------------- #
def _duplication_shape_candidates():
    """A candidate surface that reproduces the context-tier slot-waste shape."""
    return sorted(
        # 40 anchors, all in one directory -> identical affinity lists
        ["routers/api/repo%02d_handler.go" % i for i in range(40)]
        # the handful of context files every one of those anchors will pick
        + ["internal/repo_policy%02d.go" % i for i in range(6)]
        # the semantic (business-logic) surface
        + ["models/org/m%03d.go" % i for i in range(40)]
        + ["services/billing/s%03d.go" % i for i in range(40)]
        # a large unclaimed pool, so the backfill queue never runs dry
        + ["pkg/util/u%04d.go" % i for i in range(400)]
    )


def _drained_surface_candidates():
    """A surface with NO spare pool, so the backfill queue runs dry."""
    return sorted(
        ["routers/api/repo%02d_handler.go" % i for i in range(40)]
        + ["internal/repo_policy%02d.go" % i for i in range(6)]
        + ["models/org/m%03d.go" % i for i in range(8)]
    )


def _tier_of(unit):
    if unit["name"].startswith("authz_handlers."):
        return "guaranteed"
    if unit["name"].startswith("semantic_"):
        return "semantic"
    return "context"


_CAP = 24  # stands in for the review fan-out cap


def _select_dup(**kwargs):
    return bl_units.select_units(
        _duplication_shape_candidates(), files_per_unit=8, max_units=60, **kwargs
    )


def _repeats(units):
    """File slots spent on a file an earlier slot already carries."""
    counts = collections.Counter(f for u in units for f in u["files"])
    return sum(n - 1 for n in counts.values())


class TestCoverageFirstSelection:
    """``coverage_first``: spend a slot on an unclaimed file before repeating one."""

    def test_it_buys_distinct_coverage_at_identical_cost(self):
        off, on = _select_dup(), _select_dup(coverage_first=True)
        # Same names, positions and unit sizes; solo anchor units untouched.
        shape = lambda us: [(u["name"], len(u["files"])) for u in us]  # noqa: E731
        assert shape(on) == shape(off)
        assert all(
            a["files"] == b["files"] for a, b in zip(off, on) if len(a["files"]) == 1
        )
        # Inside the cap no multi-file unit repeats a file, so it reaches more.
        multi = collections.Counter(
            f for u in on[:_CAP] if len(u["files"]) > 1 for f in u["files"]
        )
        assert max(multi.values()) == 1
        assert len(_reviewed_files(on[:_CAP])) > len(_reviewed_files(off[:_CAP]))

    def test_units_keep_their_size_when_nothing_is_left_unclaimed(self):
        """Queue dry -> units keep their own files rather than SHRINK."""
        candidates = sorted(
            ["routers/api/h%02d_handler.go" % i for i in range(12)]
            + ["models/org/m%02d.go" % i for i in range(4)]
        )
        off = bl_units.select_units(candidates, files_per_unit=8, max_units=30)
        on = bl_units.select_units(
            candidates, files_per_unit=8, max_units=30, coverage_first=True
        )
        assert [len(u["files"]) for u in on] == [len(u["files"]) for u in off]
        assert _reviewed_files(on) == _reviewed_files(off)

    def test_backfill_queue_is_anchors_first_then_security_score(self):
        queue = bl_units._backfill_queue(
            ["pkg/util/plain.go", "models/org/auth_token.go", "routers/api/x.go"]
        )
        assert queue == [
            "routers/api/x.go",
            "models/org/auth_token.go",
            "pkg/util/plain.go",
        ]


class TestCapAwareBackfill:
    """``cap_aware_backfill``: anchors whose solo unit is inside the cap go to the back of the backfill queue."""

    def test_it_removes_the_residual_repeats_inside_the_cap(self):
        # Inert without coverage_first.
        assert _select_dup(cap_aware_backfill=True) == _select_dup()
        off = _select_dup(coverage_first=True)[:_CAP]
        on = _select_dup(coverage_first=True, cap_aware_backfill=True)[:_CAP]
        assert _repeats(off) > 0, "fixture must leave a residual for the dial to fix"
        assert _repeats(on) < _repeats(off)
        assert len(_reviewed_files(on)) > len(_reviewed_files(off))

    def test_deprioritised_anchors_are_backfilled_not_dropped(self):
        candidates = _drained_surface_candidates()
        base = bl_units.select_units(candidates, files_per_unit=8, max_units=60)
        on = bl_units.select_units(
            candidates,
            files_per_unit=8,
            max_units=60,
            coverage_first=True,
            cap_aware_backfill=True,
        )
        borrowed = sum(
            1
            for before, after in zip(base, on)
            if len(after["files"]) > 1
            for f in set(after["files"]) - set(before["files"])
            if bl_units._is_anchor(f)
        )
        # A filter instead of a reorder would collapse reach.
        assert borrowed >= 10
        assert _reviewed_files(on[:_CAP]) == set(candidates)


class TestSaturationStop:
    """``saturation_stop``: the PER-REPO dynamic stop."""

    def test_it_cuts_only_a_redundant_tail(self):
        off, on = _select_dup(), _select_dup(saturation_stop=True)
        assert len(on) < len(off), "the stop must bite on this fixture"
        assert on == off[: len(on)]
        assert _reviewed_files(on) == _reviewed_files(off)

    def test_the_stop_is_tight_and_reach_is_unchanged_at_every_cap(self):
        off, on = _select_dup(), _select_dup(saturation_stop=True)
        assert set(on[-1]["files"]) - _reviewed_files(on[:-1])
        for cap in (1, 5, 12, 24, 40, 62, 100, 150, 400):
            assert _reviewed_files(on[:cap]) == _reviewed_files(off[:cap]), cap


class TestBothDialsTogether:
    def test_explicit_false_dials_equal_the_unset_defaults(self):
        explicit = _select_dup(
            coverage_first=False, cap_aware_backfill=False, saturation_stop=False
        )
        assert explicit == _select_dup()

    def test_combined_still_loses_no_reach_relative_to_its_own_uncut_list(self):
        uncut = _select_dup(coverage_first=True, cap_aware_backfill=True)
        cut = _select_dup(
            coverage_first=True, cap_aware_backfill=True, saturation_stop=True
        )
        assert _reviewed_files(cut) == _reviewed_files(uncut)
        assert cut == uncut[: len(cut)]


class TestResolveBoolDial:
    def test_recognized_tokens_and_real_bools(self):
        for value in ("true", "TRUE", " Yes ", "1", "on", True):
            assert bl_units._resolve_bool_dial(value) is True
        assert bl_units._resolve_bool_dial(False) is False


def _unrecognized_cases():
    """Every resolver: a value it does not understand keeps the default.

    Off for a switch, the configured values for an effort tier, no additions. A bare word (``off``,
    ``default``) is how a flow config declares a dial without moving it, and text is not a structured answer.
    """

    def effort(value):
        return bl_units.resolve_scan_effort(value, 8, 150)

    table = [
        (
            bl_units._resolve_bool_dial,
            (None, "false", "0", "no", "maybe", 1, 3.5, [], {"a": 1}),
            False,
        ),
        (effort, (None, "", "ludicrous", 400, ["high"]), (8, 150, 1, 1)),
        (
            bl_units.resolve_extra_globs,
            (
                None,
                "",
                True,
                5,
                5.5,
                [1, 2],
                {},
                ["off"],
                ["default"],
                '{"globs": ["a/*.rb"]}',
                "a/*.rb;b/*.rb",
            ),
            [],
        ),
        (
            bl_units.resolve_extra_anchor_patterns,
            (
                None,
                "",
                True,
                5,
                [1, 2],
                {},
                {"globs": ["*.rb"]},
                '{"anchor_patterns": ["_x$"]}',
            ),
            [],
        ),
    ]
    return [
        pytest.param(
            resolve,
            value,
            default,
            id=f"{getattr(resolve, '__name__', 'effort')}-{value!r}",
        )
        for resolve, values, default in table
        for value in values
    ]


@pytest.mark.parametrize(("resolve", "value", "default"), _unrecognized_cases())
def test_an_unrecognized_value_keeps_the_default(resolve, value, default):
    got = resolve(value)
    assert got == default and type(got) is type(default)


@pytest.mark.parametrize(
    ("path", "anchor"),
    [
        # Elixir / Phoenix: controllers, router.ex and Plug modules, .ex only.
        ("lib/app_web/controllers/account_controller.ex", True),
        ("lib/app_web/user_controller.ex", True),
        ("lib/app_web/router.ex", True),
        ("lib/app_web/plugs/ensure_authenticated.ex", True),
        ("lib/app_web/plug/authorize.ex", True),
        ("lib/app/workers/receiver_worker.ex", False),
        ("lib/app/object_validators/update_validator.ex", False),
        ("lib/app/visibility.ex", False),
        ("lib/app_web/controllers/account_controller.txt", False),
        ("lib/app_web/router.txt", False),
        ("lib/app_web/plugs/ensure_authenticated.txt", False),
        # Rust and method-routed Java have no name convention.
        ("src/api/core/organizations.rs", False),
        ("core/src/main/java/example/model/WidgetSet.java", False),
    ],
)
def test_is_anchor(path, anchor):
    assert bl_units._is_anchor(path) is anchor


@pytest.mark.parametrize(
    ("path", "role"),
    [
        # Segment- not suffix-matched, so a role survives a change of language;
        # the semantic tier carries what has no anchor convention.
        ("models/organization/team.go", "state"),
        ("app/models/user.rb", "state"),
        ("src/domain/account.py", "state"),
        ("internal/storage/repo.go", "state"),
        ("src/db/models/user.rs", "state"),
        ("core/src/main/java/example/model/WidgetSet.java", "state"),
        ("services/migrations/restore.go", "logic"),
        ("app/services/checkout.rb", "logic"),
        ("src/usecases/transfer.ts", "logic"),
        ("internal/workflows/deploy.go", "logic"),
        ("src/api/core/organizations.rs", "logic"),
        ("core/src/main/java/example/core/Runner.java", "logic"),
        ("routers/web/auth/oauth2.go", "access"),
        ("app/policies/user_policy.rb", "access"),
        ("src/middleware/guard.ts", "access"),
        ("internal/rbac/check.go", "access"),
        ("src/auth/session.rs", "access"),
        ("core/src/main/java/example/security/Realm.java", "access"),
        # access wins: services/auth/ is access control, not generic logic.
        ("services/auth/token.go", "access"),
        ("README", None),
        ("cmd/main.go", None),
    ],
)
def test_role_of(path, role):
    assert bl_units._role_of(path) == role


# --------------------------------------------------------------------------- #
# SCOPED SCAN: caller-supplied file scope (target_files)
# --------------------------------------------------------------------------- #
class TestTargetFileUnits:
    @pytest.mark.parametrize(
        ("files", "fpu", "expected"),
        [
            (["lib/a/update.ex"], 8, [["lib/a/update.ex"]]),
            (
                [f"a/f{i}.rb" for i in range(5)],
                2,
                [["a/f0.rb", "a/f1.rb"], ["a/f2.rb", "a/f3.rb"], ["a/f4.rb"]],
            ),
            (["a.rb", "b.rb"], 0, [["a.rb"], ["b.rb"]]),
        ],
        ids=["one-file-one-unit", "chunked-in-order", "floor-of-one"],
    )
    def test_units_chunk_the_named_files_in_order(
        self, monkeypatch, files, fpu, expected
    ):
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 1)
        assert [u["files"] for u in bl_units.target_file_units(files, fpu)] == expected

    def test_a_label_is_neutral_and_matches_the_context_tier(self, monkeypatch):
        # The reviewer sees the name; an "authz_handlers." label would prime it.
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 1)
        path = "lib/app_web/controllers/account_controller.ex"
        scoped = bl_units.target_file_units([path, "d/e.rb"], files_per_unit=8)[0][
            "name"
        ]
        clustered = cluster([path], files_per_unit=1)[0]["name"]
        assert scoped == clustered == "lib.app_web.controllers.account_controller"

    def test_a_non_anchor_scoped_file_still_gets_a_unit(self, monkeypatch):
        # The point of the dial: reach a file full discovery would never select.
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 1)
        path = "lib/pleroma/web/activity_pub/object_validators/update_validator.ex"
        assert not bl_units._is_anchor(path)
        assert bl_units._role_of(path) is None
        assert cluster([path], files_per_unit=1) == []
        units = bl_units.target_file_units([path], files_per_unit=8)
        assert [u["files"] for u in units] == [[path]]

    @pytest.mark.parametrize("samples", [1, 3])
    def test_multi_sampled_like_the_authz_tier(self, monkeypatch, samples):
        # So a scoped miss is the same event as a full-scan miss.
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", samples)
        units = bl_units.target_file_units(["a/b.rb"], files_per_unit=8)
        assert [u["files"] for u in units] == [["a/b.rb"]] * samples
        assert len({id(u) for u in units}) == samples


class TestCapScopedUnits:
    def test_within_the_cap_keeps_every_unit_without_warning(self):
        units = [{"files": ["a.rb"]}, {"files": ["b.rb"]}]
        with structlog.testing.capture_logs() as logs:
            assert bl_units.cap_scoped_units(units, 2) == units
        assert not logs

    def test_dropping_only_repeats_of_a_kept_group_does_not_warn(self):
        units = [{"files": ["a.rb"]}, {"files": ["a.rb"]}, {"files": ["a.rb"]}]
        with structlog.testing.capture_logs() as logs:
            assert bl_units.cap_scoped_units(units, 1) == units[:1]
        assert not logs

    def test_over_the_cap_keeps_the_first_units_and_counts_dropped_files(self):
        units = [{"files": ["a.rb"]}, {"files": ["b.rb", "c.rb"]}, {"files": ["d.rb"]}]
        with structlog.testing.capture_logs() as logs:
            assert bl_units.cap_scoped_units(units, 1) == units[:1]
        assert len(logs) == 1
        assert logs[0]["log_level"] == "warning"
        assert logs[0]["groups_dropped"] == 2
        assert logs[0]["files_dropped"] == 3
        assert logs[0]["max_units"] == 1


# --------------------------------------------------------------------------- #
# content_filter_candidates -- narrowing the candidates to the content net's hits
# --------------------------------------------------------------------------- #
_ANCHOR = "app/controllers/widgets_controller.rb"
_CONTEXT_HIT = "app/models/order.rb"
_CONTEXT_MISS = "app/lib/color_wheel.rb"


class TestContentFilterCandidates:
    def test_it_keeps_the_matched_files_and_drops_the_rest(self):
        cand = [_ANCHOR, _CONTEXT_HIT, _CONTEXT_MISS]
        kept = bl_units.content_filter_candidates(cand, {_CONTEXT_HIT})
        assert kept == [_ANCHOR, _CONTEXT_HIT]

    def test_every_anchor_survives_a_body_it_does_not_match(self):
        # Otherwise the filter would undo handler_coverage_units' guarantee.
        cand = [_ANCHOR, _CONTEXT_HIT, _CONTEXT_MISS]
        kept = bl_units.content_filter_candidates(cand, set())
        assert kept == [_ANCHOR]
        assert bl_units._is_anchor(_ANCHOR)

    def test_order_is_preserved_and_the_result_is_a_subset(self):
        cand = sorted(_big_repo_paths())
        kept = bl_units.content_filter_candidates(cand, set(cand[::3]))
        assert kept == [p for p in cand if p in set(kept)]
        assert set(kept) <= set(cand)

    def test_the_anchor_floor_still_holds_end_to_end(self):
        cand = bl_units.enumerate_candidates([_big_repo_paths()])
        anchors = {p for p in cand if bl_units._is_anchor(p)}
        kept = bl_units.content_filter_candidates(cand, set())
        assert {p for p in kept if bl_units._is_anchor(p)} == anchors
        units = bl_units.handler_coverage_units(kept)
        assert {f for u in units for f in u["files"]} == anchors

    def test_a_list_and_a_set_of_matches_behave_alike(self):
        cand = [_ANCHOR, _CONTEXT_HIT, _CONTEXT_MISS]
        assert bl_units.content_filter_candidates(
            cand, [_CONTEXT_HIT]
        ) == bl_units.content_filter_candidates(cand, {_CONTEXT_HIT})


# --------------------------------------------------------------------------- #
# Per-repository ADDITIVE extra globs / anchor patterns
# --------------------------------------------------------------------------- #
class TestResolveExtraGlobs:
    def test_a_list_and_the_answer_object_both_work(self):
        want = ["*_endpoint.rb", "app/gateway/**/*"]
        assert bl_units.resolve_extra_globs(want) == want
        assert (
            bl_units.resolve_extra_globs({"globs": want, "anchor_patterns": ["x"]})
            == want
        )
        assert bl_units.resolve_extra_globs(["a/*.rb", " ", "b/*.rb "]) == [
            "a/*.rb",
            "b/*.rb",
        ]

    def test_dot_and_doubled_separators_are_normalised(self):
        got = bl_units.resolve_extra_globs(
            ["./src/*.js", "lib//auth/*.rb", "/app/x/", "./app/controllers/**/*"]
        )
        # The last one is then a duplicate of a built-in glob.
        assert got == ["src/*.js", "lib/auth/*.rb", "app/x/*"]
        assert bl_units.resolve_extra_globs(["./*", ".//", "./off"]) == []

    def test_a_pattern_matching_the_whole_tree_is_rejected(self):
        for pattern in ("*", "**", "**/*", "*/*", "*.*", "*/**", "."):
            assert bl_units.resolve_extra_globs([pattern]) == [], pattern
        # A pattern rooted in a directory only looks broad.
        assert bl_units.resolve_extra_globs(["app/gateway/**/*"]) == [
            "app/gateway/**/*"
        ]

    def test_a_character_class_or_backslash_is_dropped(self):
        # A malformed class can fail the whole listing call.
        for pattern in ("foo[a-z].rb", "foo].rb", "foo?.rb", "foo" + chr(92) + "x.rb"):
            assert bl_units.resolve_extra_globs([pattern]) == [], pattern

    def test_a_pattern_already_built_in_is_not_fired_twice(self):
        assert bl_units.resolve_extra_globs([bl_units.AUTHZ_GLOBS[0]]) == []

    def test_over_the_cap_means_no_additions_rather_than_a_truncated_prefix(self):
        # A truncated prefix of a runaway generation would read as deliberate.
        at_cap = ["*_x%d.rb" % i for i in range(bl_units.MAX_EXTRA_GLOBS)]
        assert len(bl_units.resolve_extra_globs(at_cap)) == bl_units.MAX_EXTRA_GLOBS
        assert bl_units.resolve_extra_globs(at_cap + ["*_over.rb"]) == []

    def test_an_absurdly_long_pattern_is_dropped(self):
        assert (
            bl_units.resolve_extra_globs(
                ["*" + "a" * bl_units.MAX_EXTRA_PATTERN_CHARS + ".rb"]
            )
            == []
        )


class TestResolveExtraAnchorPatterns:
    def test_a_bad_pattern_costs_only_itself(self):
        # Malformed, repeated or over-long: dropped individually, never raised.
        over_long = "a" * (bl_units.MAX_EXTRA_PATTERN_CHARS + 1)
        out = bl_units.resolve_extra_anchor_patterns(
            [r"_endpoint\.rb$", "[", "(unclosed", over_long, r"_endpoint\.rb$"]
        )
        assert [p.pattern for p in out] == [r"_endpoint\.rb$"]

    def test_a_pattern_matching_every_path_is_rejected(self):
        for pattern in (".*", ".+", "^", "$", "/", "^.*$", "()"):
            assert bl_units.resolve_extra_anchor_patterns([pattern]) == [], pattern

    def test_over_the_cap_means_no_additions(self):
        at_cap = ["_x%d$" % i for i in range(bl_units.MAX_EXTRA_ANCHOR_PATTERNS)]
        assert (
            len(bl_units.resolve_extra_anchor_patterns(at_cap))
            == bl_units.MAX_EXTRA_ANCHOR_PATTERNS
        )
        assert bl_units.resolve_extra_anchor_patterns(at_cap + ["_over$"]) == []

    def test_it_is_idempotent_on_already_compiled_patterns(self):
        once = bl_units.resolve_extra_anchor_patterns([r"_endpoint\.rb$"])
        assert bl_units.resolve_extra_anchor_patterns(once) == once

    def test_one_answer_feeds_both_inputs(self):
        answer = {"globs": ["*_endpoint.rb"], "anchor_patterns": [r"_endpoint\.rb$"]}
        assert bl_units.resolve_extra_globs(answer) == ["*_endpoint.rb"]
        assert [p.pattern for p in bl_units.resolve_extra_anchor_patterns(answer)] == [
            r"_endpoint\.rb$"
        ]


class TestCapExtraAnchorFiles:
    CAP = bl_units.MAX_EXTRA_ANCHOR_FILES
    BUILT_IN = ["app/controllers/a_controller.rb", "app/controllers/z_controller.rb"]

    def _over_cap(self):
        # Reversed, so the kept set cannot be the input order by accident.
        return [f"lib/gw/f{i:04d}.go" for i in range(self.CAP + 25)][::-1]

    def test_within_the_cap_the_patterns_are_unchanged(self):
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)gw/"]))
        paths = [f"lib/gw/f{i}.go" for i in range(self.CAP)]
        with structlog.testing.capture_logs() as logs:
            assert bl_units.cap_extra_anchor_files(extra, paths) == extra
        assert not logs

    def test_over_the_cap_keeps_the_first_files_by_path_and_warns(self):
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)gw/"]))
        paths = self._over_cap()
        with structlog.testing.capture_logs() as logs:
            capped = bl_units.cap_extra_anchor_files(extra, paths)
        kept = [p for p in paths if bl_units._is_anchor(p, capped)]
        assert sorted(kept) == sorted(paths)[: self.CAP]
        assert len(kept) == self.CAP
        # Same input, same subset, whatever the order it arrives in.
        again = bl_units.cap_extra_anchor_files(extra, sorted(paths))
        assert [p for p in paths if bl_units._is_anchor(p, again)] == kept
        warned = [e for e in logs if "add too many files" in e["event"]]
        assert len(warned) == 1
        assert (warned[0]["kept"], warned[0]["dropped"]) == (self.CAP, 25)

    def test_built_in_anchors_are_untouched_and_not_counted(self):
        extra = tuple(
            bl_units.resolve_extra_anchor_patterns([r"(^|/)(gw|controllers)/"])
        )
        paths = self._over_cap() + self.BUILT_IN
        capped = bl_units.cap_extra_anchor_files(extra, paths)
        assert all(bl_units._is_anchor(p, capped) for p in self.BUILT_IN)
        added = [p for p in paths if p not in self.BUILT_IN]
        assert sum(bl_units._is_anchor(p, capped) for p in added) == self.CAP
        # Built-ins alone never trip the cap, however many there are.
        many_built_in = [
            f"app/controllers/c{i}_controller.rb" for i in range(self.CAP + 1)
        ]
        assert bl_units.cap_extra_anchor_files(extra, many_built_in) == extra


class TestExtraAnchorsAreAdditiveNotSubstitutive:
    """Additive-only must be a property of this code, not of the generating stage behaving."""

    PATHS = _big_repo_paths()

    def test_no_extra_pattern_can_unmake_a_built_in_anchor(self):
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)gateway/"]))
        built_in = [p for p in self.PATHS if bl_units._is_anchor(p)]
        assert built_in
        for path in built_in:
            assert bl_units._is_anchor(path, extra), path

    def test_the_anchor_set_only_grows(self):
        cand = bl_units.enumerate_candidates([self.PATHS + ["lib/gateway/thing.go"]])
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)gateway/"]))
        before = {p for p in cand if bl_units._is_anchor(p)}
        after = {p for p in cand if bl_units._is_anchor(p, extra)}
        assert before < after
        assert "lib/gateway/thing.go" in after - before

    def test_an_added_anchor_gains_its_own_guaranteed_unit(self):
        cand = bl_units.enumerate_candidates([self.PATHS + ["lib/gateway/thing.go"]])
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)gateway/"]))
        before = bl_units.handler_coverage_units(cand)
        after = bl_units.handler_coverage_units(cand, extra)
        assert len(after) == len(before) + 1
        assert ["lib/gateway/thing.go"] in [u["files"] for u in after]

    def test_built_in_anchors_come_before_pattern_only_anchors(self):
        extra = tuple(bl_units.resolve_extra_anchor_patterns([r"(^|/)aaa/"]))
        units = bl_units.handler_coverage_units(
            ["aaa/x.rb", "app/controllers/b_controller.rb"], extra_anchors=extra
        )
        assert [u["files"] for u in units] == [
            ["app/controllers/b_controller.rb"],
            ["aaa/x.rb"],
        ]


# Unusable as either extra input.
_JUNK_EXTRAS = (None, "", "off", True, 5, ["*"], ["["], "{not json", {"globs": [1, 2]})


class TestExtraInputsDefaultOffIsTodaysBehaviour:
    PATHS = _big_repo_paths()

    def test_select_units_is_identical_for_an_unrecognized_value(self):
        cand = bl_units.enumerate_candidates([self.PATHS])
        base = bl_units.select_units(cand, 8, 62, coverage_first=True)
        for junk in _JUNK_EXTRAS:
            got = bl_units.select_units(
                cand, 8, 62, coverage_first=True, extra_anchor_patterns=junk
            )
            assert got == base, junk


def _floor_bound_candidates():
    """A surface on which the context-tier FLOOR binds: more anchors than the budget, each with policy files."""
    out = []
    for d in range(10):
        out += ["routers/api/g%d/h%03d_handler.go" % (d, i) for i in range(20)]
        out += ["routers/api/g%d/p%03d_policy.go" % (d, i) for i in range(8)]
    out += ["models/org/m%03d.go" % i for i in range(40)]
    out += ["services/billing/s%03d.go" % i for i in range(40)]
    out += ["pkg/util/u%04d.go" % i for i in range(400)]
    return sorted(out)


class TestContextDials:
    """``context_pool_multiplier`` (how many context units exist) and ``context_merge_weight`` (how many prefix slots
    per pass they win).

    Both change which candidates win the cap's slots, never the unit membership.
    """

    FPU = 8
    MAX_UNITS = 62

    def _select(self, **kwargs):
        return bl_units.select_units(
            _floor_bound_candidates(),
            files_per_unit=self.FPU,
            max_units=self.MAX_UNITS,
            **kwargs,
        )

    @staticmethod
    def _context_units(units):
        return [u for u in units if _tier_of(u) == "context"]

    def test_more_pool_admits_more_context_into_the_reviewed_prefix(self):
        one = self._select(coverage_first=True)[: self.MAX_UNITS]
        two = self._select(coverage_first=True, context_pool_multiplier=2)[
            : self.MAX_UNITS
        ]
        assert len(one) == len(two) == self.MAX_UNITS, "the cap itself must not move"
        assert len(self._context_units(two)) > len(self._context_units(one))
        assert len(_reviewed_files(two)) > len(_reviewed_files(one))

    def test_weight_gives_the_context_tier_a_bigger_share_of_the_prefix(self):
        one = self._select(coverage_first=True, context_pool_multiplier=8)[
            : self.MAX_UNITS
        ]
        three = self._select(
            coverage_first=True, context_pool_multiplier=8, context_merge_weight=3
        )[: self.MAX_UNITS]
        assert len(one) == len(three) == self.MAX_UNITS, "the cap itself must not move"
        # Either side of half the prefix: the behaviour, not the arithmetic.
        assert len(self._context_units(one)) < self.MAX_UNITS // 2
        assert len(self._context_units(three)) > self.MAX_UNITS // 2
        assert three[0]["name"].startswith("authz_handlers.")

    @pytest.mark.parametrize(
        "dial", ["context_pool_multiplier", "context_merge_weight"]
    )
    def test_values_below_one_are_clamped_and_never_starve_the_tier(self, dial):
        one = self._select(coverage_first=True, **{dial: 1})
        for bad in (0, -1, -100):
            assert self._select(coverage_first=True, **{dial: bad}) == one

    @pytest.mark.parametrize(
        "dial", ["context_pool_multiplier", "context_merge_weight"]
    )
    def test_a_dial_at_one_is_identical_to_unset(self, dial):
        assert self._select(**{dial: 1}) == self._select()
