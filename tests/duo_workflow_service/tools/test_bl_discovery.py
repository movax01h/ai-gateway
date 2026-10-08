"""Unit tests for the BL discovery tool: its async ``_find`` / ``_search`` / ``_execute`` path with
``_execute_action`` mocked.

The pure selection logic is tested in ``tests/duo_workflow_service/bl_security/test_units.py``.
"""

import asyncio
import fnmatch
import hashlib
import json
import re

import pytest
import structlog.testing
from pydantic import ValidationError

import duo_workflow_service.bl_security.units as bl_units
import duo_workflow_service.tools.bl_discovery as bl
from duo_workflow_service.bl_security.units import select_units
from duo_workflow_service.tools.bl_discovery import (
    BlDiscoverAndCluster,
    BlDiscoverAndClusterInput,
)
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD
from tests.duo_workflow_service.bl_security.test_units import (
    _JUNK_EXTRAS,
    _big_repo_paths,
    _floor_bound_candidates,
)


def _tool():
    return BlDiscoverAndCluster(metadata={"outbox": object()})


def _execute(**kwargs):
    return asyncio.run(_tool()._execute(**kwargs))


def _glob_match(path, pattern):
    """How ``find_files`` matches, as observed with ``rg --files -g``.

    A ``/``-free pattern matches the basename; otherwise the repo-relative path, where ``*`` stays inside one segment
    and ``**`` spans zero or more directories.
    """
    if "/" not in pattern:
        return fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pattern)
    rx, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            rx, i = rx + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            rx, i = rx + ".*", i + 2
        elif pattern[i] == "*":
            rx, i = rx + "[^/]*", i + 1
        else:
            rx, i = rx + re.escape(pattern[i]), i + 1
    return re.fullmatch(rx, path) is not None


def _executor(
    paths=(), *, faithful=False, search=None, calls=None, raises=None, max_bytes=None
):
    """The one fake ``_execute_action``.

    ``findFiles`` answers with every path in ``paths`` or, when ``faithful``, the ones the pattern matches; a str (or
    any non-list) is returned as is. A listing over ``max_bytes`` gets the Node executor's size-limit cut.
    ``runCommand`` answers ``search``: ``None`` means no search program, a set is the ``rg -l`` hits, a str is raw
    output. ``calls`` records each action; ``raises`` is raised instead of answering.
    """

    async def _fake(metadata, action):
        if calls is not None:
            calls.append(action)
        if raises is not None:
            raise raises
        if action.HasField("runCommand"):
            if search is None or isinstance(search, str):
                return search or ""
            return "\n".join("./" + p for p in sorted(search))
        if not isinstance(paths, (list, tuple)):
            return paths
        pattern = action.findFiles.name_pattern
        listing = "\n".join(p for p in paths if not faithful or _glob_match(p, pattern))
        if max_bytes is None or len(listing) <= max_bytes:
            return listing
        half = max_bytes // 2
        return (
            f"Maximum allowed size exceeded. Result: {len(listing)} bytes. "
            f"Maximum allowed: {max_bytes} bytes\n{listing[:half]}"
            "\n[... output truncated: content exceeded the maximum allowed size ...]\n"
            f"{listing[-half:]}"
        )

    return _fake


def _install(monkeypatch, paths=(), **kwargs):
    """Install the fake executor; returns the list of actions it receives."""
    calls = []
    monkeypatch.setattr(bl, "_execute_action", _executor(paths, calls=calls, **kwargs))
    return calls


def _patterns(calls):
    return [a.findFiles.name_pattern for a in calls if a.HasField("findFiles")]


def _spy_select(monkeypatch):
    """Record what ``_execute`` hands ``select_units``, and how many times it ran."""
    seen = {"calls": 0}
    real = bl.select_units

    def _spy(candidates, files_per_unit, max_units, **kwargs):
        seen.update(kwargs)
        seen.update(
            calls=seen["calls"] + 1,
            candidates=list(candidates),
            files_per_unit=files_per_unit,
            max_units=max_units,
        )
        return real(candidates, files_per_unit, max_units, **kwargs)

    monkeypatch.setattr(bl, "select_units", _spy)
    return seen


def test_the_tool_is_hidden_from_list_tools():
    # ListTools publishes only tools at or above STABLE_VERSION_THRESHOLD.
    assert BlDiscoverAndCluster.tool_version < STABLE_VERSION_THRESHOLD


def test_the_flow_config_names_are_still_read_through_the_tool_module():
    for name in (
        "resolve_content_filter",
        "resolve_coverage_first",
        "resolve_cap_aware_backfill",
        "resolve_saturation_stop",
    ):
        assert getattr(bl, name) is bl_units._resolve_bool_dial
    assert bl.SCAN_EFFORT_TIERS is bl_units.SCAN_EFFORT_TIERS


def test_a_large_output_stays_a_list(monkeypatch):
    # Past the 200 KiB display cap, generic truncation would return a string.
    big = [{"name": f"u{i}", "files": ["x" * 100]} for i in range(3000)]

    async def _big(self, **_kwargs):
        return big

    monkeypatch.setattr(BlDiscoverAndCluster, "_execute", _big)
    assert asyncio.run(_tool()._arun(files_per_unit=8, max_units=40)) == big


# --------------------------------------------------------------------------- #
# BlDiscoverAndClusterInput validator (_coerce_int)
# --------------------------------------------------------------------------- #
class TestInputValidator:
    def test_coerces_string_ints(self):
        model = BlDiscoverAndClusterInput(files_per_unit=" 3 ", max_units="5")
        assert model.files_per_unit == 3
        assert model.max_units == 5

    @pytest.mark.parametrize("field", ["files_per_unit", "max_units"])
    @pytest.mark.parametrize("value", [0, -1, "0", "-1"])
    def test_rejects_counts_below_one(self, field, value):
        kwargs = {"files_per_unit": 3, "max_units": 5, field: value}
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            BlDiscoverAndClusterInput(**kwargs)


# --------------------------------------------------------------------------- #
# find_files patterns: directory globs go on the wire as `**/<glob>`
# --------------------------------------------------------------------------- #
class TestFindFilesPatterns:
    def test_the_fake_matches_like_find_files(self):
        # Pins the test-side matcher to what `rg --files -g` does, so the
        # faithful fake below can stand in for the executor.
        tree = ["views/root.py", "a/views/v.py", "a/views/x/v.py", "setup.py"]
        assert [p for p in tree if _glob_match(p, "**/*/views/**/*.py")] == tree[1:3]
        assert [p for p in tree if _glob_match(p, "*.py")] == tree
        assert [p for p in tree if _glob_match(p, "**/views/*.py")] == tree[:2]
        assert _glob_match("src/app.js", "**/src/*.js")
        assert not _glob_match("src/a/app.js", "**/src/*.js")

    def test_a_directory_glob_is_sent_rooted_anywhere(self):
        assert bl._wire_pattern("app/models/**/*") == "**/app/models/**/*"
        assert bl._wire_pattern("**/app/*.rb") == "**/app/*.rb"
        assert bl._wire_pattern("*_controller.rb") == "*_controller.rb"

    def test_execute_fires_each_glob_once_and_never_the_whole_tree(self, monkeypatch):
        calls = _install(monkeypatch, [], faithful=True)
        _execute(files_per_unit=4, max_units=20)
        sent = _patterns(calls)
        assert sent == [bl._wire_pattern(g) for g in bl.AUTHZ_GLOBS]
        assert len(sent) == len(set(sent))
        assert "*" not in sent

    def test_every_directory_glob_reaches_below_its_top_layer_and_no_further(self):
        # A nested file inside the named directory is found; the same file under
        # a differently named directory is not.
        for glob in [g for g in bl.AUTHZ_GLOBS if "/" in g]:
            segs = glob.split("/")
            dirs = [s.replace("*", "x") for s in segs[:-1] if s != "**"]
            name = segs[-1].replace("*", "x")
            shallow = "/".join(dirs + [name])
            nested = "/".join(dirs + ["sub", "deeper", name])
            wire = bl._wire_pattern(glob)
            assert _glob_match(shallow, wire) and _glob_match(nested, wire), glob
            named = [i for i, s in enumerate(dirs) if s not in ("x",)]
            dirs[named[-1]] = "not_" + dirs[named[-1]]
            assert not _glob_match("/".join(dirs + [name]), wire), glob

    def test_framework_documented_nested_layouts_are_all_discovered(self, monkeypatch):
        tree = [
            "shop/graphql/orders/mutations/refund.py",
            "shop/graphql/account/resolvers.py",
            "app/controllers/admin/reports_controller.rb",
            "app/models/concerns/orderable.rb",
            "app/services/billing/charge_service.rb",
            "shop/api/v2/views.py",
            "server/routes/admin/users.ts",
            "internal/services/billing/invoice.go",
            "backend/app/models/user.rb",
        ]
        _install(monkeypatch, tree, faithful=True)
        seen = _spy_select(monkeypatch)
        _execute(files_per_unit=8, max_units=40)
        assert set(tree) <= set(seen["candidates"])

    def test_a_directory_half_still_discriminates(self, monkeypatch):
        tree = ["srv/billing/graphql/types.py", "srv/billing/schema/types.rb"]
        _install(monkeypatch, tree, faithful=True)
        seen = _spy_select(monkeypatch)
        _execute(files_per_unit=8, max_units=40)
        assert seen["candidates"] == ["srv/billing/graphql/types.py"]


# --------------------------------------------------------------------------- #
# _find (async, _execute_action mocked)
# --------------------------------------------------------------------------- #
class TestFind:
    def test_find_returns_splitlines_on_success(self, monkeypatch):
        calls = _install(monkeypatch, "a/x_controller.rb\nb/y_controller.rb\n")
        out = asyncio.run(_tool()._find("*_controller.rb"))
        assert out == ["a/x_controller.rb", "b/y_controller.rb"]
        assert _patterns(calls) == ["*_controller.rb"]

    def test_find_returns_empty_on_exception(self, monkeypatch):
        _install(monkeypatch, raises=RuntimeError("executor down"))
        with structlog.testing.capture_logs() as logs:
            assert asyncio.run(_tool()._find("*.go")) == []
        warned = [e for e in logs if "candidates are missing" in e["event"]]
        assert [(e["log_level"], e["pattern"]) for e in warned] == [("warning", "*.go")]

    @pytest.mark.parametrize("answer", ["   ", 42])
    def test_find_returns_empty_on_non_string_or_blank(self, monkeypatch, answer):
        _install(monkeypatch, answer)
        assert asyncio.run(_tool()._find("*.rb")) == []


# --------------------------------------------------------------------------- #
# _execute (async end-to-end: fires all globs, clusters, scores, selects)
# --------------------------------------------------------------------------- #
class TestExecute:
    def test_execute_produces_scored_selected_units(self, monkeypatch):
        # Every glob returns the same small authz surface; enumerate de-dups it.
        _install(
            monkeypatch,
            "app/controllers/admin/users_controller.rb\n"
            "app/models/user.rb\n"
            "app/policies/user_policy.rb\n"
            "app/controllers/orders_controller.rb\n"
            "app/models/order.rb",
        )
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", 1)
        out = _execute(files_per_unit=2, max_units=1)

        # Both anchors get a solo unit even at max_units=1, and the guaranteed
        # tier leads the interleave in lexicographic anchor order.
        anchor_units = [u for u in out if u["name"].startswith("authz_handlers.")]
        assert len(anchor_units) == 2
        assert all(len(u["files"]) == 1 for u in anchor_units)
        assert {u["files"][0] for u in anchor_units} == {
            "app/controllers/admin/users_controller.rb",
            "app/controllers/orders_controller.rb",
        }
        assert out[0]["files"][0] == "app/controllers/admin/users_controller.rb"

        # The semantic tier guarantees the models and policies, disjoint from
        # the anchors.
        semantic_files = {
            f for u in out if u["name"].startswith("semantic_") for f in u["files"]
        }
        assert semantic_files == {
            "app/models/order.rb",
            "app/models/user.rb",
            "app/policies/user_policy.rb",
        }
        assert not (semantic_files & {u["files"][0] for u in anchor_units})

    def test_execute_max_units_floor_and_empty_repo(self, monkeypatch):
        _install(monkeypatch, "")
        assert _execute(files_per_unit=3, max_units=0) == []


@pytest.mark.parametrize("scan_effort", ["high", None, "ludicrous"])
def test_execute_hands_select_units_the_tier_or_the_configured_knobs(
    monkeypatch, scan_effort
):
    _install(monkeypatch, _big_repo_paths())
    seen = _spy_select(monkeypatch)
    _execute(files_per_unit=8, max_units=150, scan_effort=scan_effort)
    knobs = ("files_per_unit", "max_units")
    dials = ("context_pool_multiplier", "context_merge_weight")
    got = tuple(seen[k] for k in knobs + dials)
    assert got == bl_units.SCAN_EFFORT_TIERS.get(scan_effort, (8, 150, 1, 1))
    assert seen["extra_anchor_patterns"] == ()


@pytest.mark.parametrize(
    ("value", "on"), [(None, False), ("true", True), ("maybe", False)]
)
def test_coverage_first_reaches_select_units(monkeypatch, value, on):
    _install(monkeypatch, _big_repo_paths())
    seen = _spy_select(monkeypatch)
    _execute(files_per_unit=8, max_units=150, coverage_first=value)
    assert seen["coverage_first"] is on


# --------------------------------------------------------------------------- #
# Language-wide nets (Elixir, Rust, Java)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("glob", "path"),
    [
        ("*.ex", "lib/app/visibility.ex"),
        ("*.rs", "src/api/core/organizations.rs"),
        ("*.java", "core/src/main/java/example/model/WidgetSet.java"),
    ],
)
def test_a_language_net_is_present_and_extension_scoped(glob, path):
    # Like `*.go`: a language-wide net cannot encode one repository's layout,
    # and matches nothing on a tree without that language.
    assert glob in bl.AUTHZ_GLOBS
    assert _glob_match(path, glob)
    others = ["app/models/user.rb", "internal/routers/api.go", "web/x.ts"]
    assert not [p for p in others if _glob_match(p, glob)]


@pytest.mark.parametrize(
    "path",
    [
        "src/billing/controllers/invoice.controller.ts",
        "src/billing/services/invoice.service.ts",
        "server/plugins/quota/policies/is-owner.js",
        "server/plugins/quota/middlewares/rate-limit.js",
        "svc/orders/endpoints/order_endpoint.py",
        "svc/orders/schemas/order_schema.py",
    ],
)
def test_a_role_directory_glob_reaches_a_per_module_layout(path):
    # The role nouns with any parent and any extension.
    role_globs = [
        g for g in bl.AUTHZ_GLOBS if g.startswith("*/") and g.endswith("/**/*")
    ]
    assert [g for g in role_globs if _glob_match(path, bl._wire_pattern(g))]


class TestExecuteScopedScan:
    """The `_execute` short-circuit: scope in, discovery skipped."""

    def _run(self, monkeypatch, samples=1, **kw):
        calls = _install(monkeypatch, _big_repo_paths())
        seen = _spy_select(monkeypatch)
        monkeypatch.setattr(bl_units, "AUTHZ_SAMPLE_COUNT", samples)
        return _execute(**kw), _patterns(calls), seen["calls"]

    def test_target_files_skips_discovery_entirely(self, monkeypatch):
        path = "lib/pleroma/web/activity_pub/object_validators/update_validator.ex"
        out, fired, selections = self._run(
            monkeypatch, files_per_unit=8, max_units=150, target_files=path
        )
        assert (fired, selections) == ([], 0)
        assert [u["files"] for u in out] == [[path]]

    def test_scoped_pool_is_exactly_the_named_files(self, monkeypatch):
        named = ["a/one.rb", "b/two.go", "c/three.ex"]
        out, _, _ = self._run(
            monkeypatch, files_per_unit=8, max_units=150, target_files=",".join(named)
        )
        assert sorted({f for u in out for f in u["files"]}) == sorted(named)

    def test_scoped_run_respects_max_units(self, monkeypatch):
        named = [f"a/f{i}.rb" for i in range(4)]
        out, _, _ = self._run(
            monkeypatch, 3, files_per_unit=1, max_units=5, target_files=" ".join(named)
        )
        # 4 units x 3 samples, capped to 5: the cap costs samples, never files.
        assert len(out) == 5
        assert sorted({f for u in out for f in u["files"]}) == sorted(named)

    def test_scoped_run_warns_when_max_units_drops_files(self, monkeypatch):
        named = [f"a/f{i}.rb" for i in range(5)]
        with structlog.testing.capture_logs() as logs:
            self._run(
                monkeypatch, files_per_unit=2, max_units=2, target_files=",".join(named)
            )
        warned = [e for e in logs if "files are not reviewed" in e["event"]]
        assert [(e["groups_dropped"], e["files_dropped"]) for e in warned] == [(1, 1)]
        pool = [e for e in logs if e["event"] == "bl_discover_reviewed_pool"]
        assert [(e["n"], e["files"]) for e in pool] == [(4, "|".join(named[:4]))]

    def test_scan_effort_still_sizes_the_scoped_units(self, monkeypatch):
        named = [f"a/f{i}.rb" for i in range(16)]
        out, _, _ = self._run(
            monkeypatch,
            files_per_unit=2,
            max_units=150,
            scan_effort="high",  # tier files_per_unit = 8
            target_files=",".join(named),
        )
        assert len(out) == 2

    def test_a_blank_scope_runs_full_discovery_unchanged(self, monkeypatch):
        baseline, fired, selections = self._run(
            monkeypatch, files_per_unit=8, max_units=150
        )
        assert fired and selections == 1 and baseline
        for blank in ("", "   ", ",", None):
            out, _, selections = self._run(
                monkeypatch, files_per_unit=8, max_units=150, target_files=blank
            )
            assert (out, selections) == (baseline, 1), blank


class TestContentNet:
    def test_both_groups_are_in_the_one_pattern(self):
        for term in bl._AUTHZ_TERMS + bl._LOOKUP_TERMS:
            assert term in bl.CONTENT_NET_TERMS
        assert bl.CONTENT_NET_PATTERN.count("|") == len(bl.CONTENT_NET_TERMS) - 1

    def test_a_term_matches_whole_word_and_ignores_case(self):
        for text in (
            "if CURRENT_USER.admin?",
            "Order.objects.get(pk=pk)",
            "def retrieve(self, request):",
        ):
            assert re.search(bl.CONTENT_NET_PATTERN, text, re.IGNORECASE), text

    def test_a_term_inside_a_longer_word_is_not_a_match(self):
        # The boundary is what stops the net degenerating into "every file".
        for text in (
            "tokenizer = build_tokenizer()",
            "username_lookup_table = {}",
            "def paint(canvas): return canvas",
        ):
            assert not re.search(bl.CONTENT_NET_PATTERN, text, re.IGNORECASE), text


class TestExecuteContentFilter:
    PATHS = _big_repo_paths()

    def _run(self, monkeypatch, matched=None, **kwargs):
        _install(monkeypatch, self.PATHS, search=matched)
        return _execute(files_per_unit=8, max_units=40, **kwargs)

    @pytest.mark.parametrize("value", [None, "maybe"])
    def test_the_dial_off_reads_no_body_at_all(self, monkeypatch, value):
        calls = _install(monkeypatch, self.PATHS, search=set(self.PATHS))
        _execute(files_per_unit=8, max_units=40, content_filter=value)
        assert not [a for a in calls if a.HasField("runCommand")]

    def test_the_dial_on_narrows_the_reviewed_pool(self, monkeypatch):
        wide = self._run(monkeypatch, matched=None)
        cand = bl_units.enumerate_candidates([self.PATHS])
        keep = {p for p in cand if bl_units._is_anchor(p)} | set(sorted(cand)[:5])
        narrow = self._run(monkeypatch, matched=keep, content_filter="true")
        wide_files = {f for u in wide for f in u["files"]}
        narrow_files = {f for u in narrow for f in u["files"]}
        assert narrow_files < wide_files
        assert narrow_files <= keep

    def test_a_missing_search_program_degrades_to_todays_behaviour(self, monkeypatch):
        off = self._run(monkeypatch, matched=None)
        degraded = self._run(monkeypatch, matched=None, content_filter="true")
        assert degraded == off
        assert degraded


class TestExtraAnchorsAreAdditiveNotSubstitutive:
    """Additive-only must be a property of this code, not of the generating stage behaving."""

    PATHS = _big_repo_paths()

    def test_the_candidate_pool_only_grows_when_a_glob_is_added(self, monkeypatch):
        # Asserted on the candidate surface, not the reviewed files.
        paths = [
            "app/controllers/users_controller.rb",
            "lib/gateway/pipe.rb",
            "lib/gateway/valve.rb",
        ]
        _install(monkeypatch, paths, faithful=True)
        seen = _spy_select(monkeypatch)
        _execute(files_per_unit=8, max_units=400)
        base = set(seen["candidates"])
        _execute(files_per_unit=8, max_units=400, extra_globs=["lib/gateway/**/*"])
        added = set(seen["candidates"])
        assert base == {"app/controllers/users_controller.rb"}
        assert base < added
        assert {"lib/gateway/pipe.rb", "lib/gateway/valve.rb"} <= added


class TestAnchorPatternsAddCandidates:
    # The tuner names a few files as globs and the handlers as anchors only.
    PATHS = [
        "app/auth.py",
        "app/db.py",
        "app/orders.py",
        "app/users.py",
        "app/templates/base.html",
    ]
    GLOBS = ["app/auth.py", "app/db.py"]
    ANCHORS = [r"app/orders\.py$", r"app/users\.py$"]

    def _run(self, monkeypatch, **kwargs):
        self.calls = _install(monkeypatch, self.PATHS, faithful=True)
        seen = _spy_select(monkeypatch)
        units = _execute(
            files_per_unit=8, max_units=400, extra_globs=self.GLOBS, **kwargs
        )
        reviewed = {f for u in units for f in u["files"]}
        return set(seen["candidates"]), reviewed

    def test_anchor_matches_become_candidates_and_get_units(self, monkeypatch):
        candidates, reviewed = self._run(
            monkeypatch, extra_anchor_patterns=self.ANCHORS
        )
        assert {"app/orders.py", "app/users.py"} <= candidates
        assert {"app/orders.py", "app/users.py"} <= reviewed
        assert "app/templates/base.html" not in candidates
        # They need the whole tree, which the built-in globs no longer list.
        assert _patterns(self.calls).count("*") == 1

    def test_the_tuner_answer_object_reaches_the_handlers(self, monkeypatch):
        answer = {"globs": [], "anchor_patterns": self.ANCHORS}
        _, reviewed = self._run(monkeypatch, extra_anchor_patterns=answer)
        assert {"app/orders.py", "app/users.py"} <= reviewed

    def test_without_anchors_the_candidates_are_unchanged(self, monkeypatch):
        candidates, _ = self._run(monkeypatch)
        assert candidates == {"app/auth.py", "app/db.py"}
        assert "*" not in _patterns(self.calls)

    def test_a_truncated_listing_is_reported(self, monkeypatch):
        monkeypatch.setattr(bl, "executor_truncated", lambda text: True)
        with structlog.testing.capture_logs() as logs:
            self._run(monkeypatch, extra_anchor_patterns=self.ANCHORS)
        warned = [e for e in logs if "candidates are missing" in e["event"]]
        assert "anchor_patterns" in warned[0]["globs"]

    def test_a_broad_pattern_adds_nothing_and_warns(self, monkeypatch):
        many = [
            f"app/mod{i}.py" for i in range(bl_units.MAX_FILES_PER_ANCHOR_PATTERN + 1)
        ]
        monkeypatch.setattr(self, "PATHS", self.PATHS + many)
        with structlog.testing.capture_logs() as logs:
            candidates, _ = self._run(
                monkeypatch, extra_anchor_patterns=[r"\.py$", r"app/orders\.py$"]
            )
        assert not candidates & set(many)
        assert "app/orders.py" in candidates
        warned = [e for e in logs if "too many files" in e["event"]]
        assert warned[0]["pattern"] == r"\.py$"

    def test_a_rejected_pattern_marks_no_entry_points(self, monkeypatch):
        many = [
            f"app/mod{i}.py" for i in range(bl_units.MAX_FILES_PER_ANCHOR_PATTERN + 1)
        ]
        _install(monkeypatch, self.PATHS + many, faithful=True)

        def _units(anchors):
            return _execute(
                files_per_unit=8,
                max_units=10,
                extra_globs=["app/*.py"],
                extra_anchor_patterns=anchors,
            )

        # The over-cap pattern labels no glob hit as an entry point either.
        assert _units([r"\.py$", r"orders\.py$"]) == _units([r"orders\.py$"])

    def test_the_patterns_together_add_at_most_the_total_cap(self, monkeypatch):
        cap = bl_units.MAX_EXTRA_ANCHOR_FILES
        per = bl_units.MAX_FILES_PER_ANCHOR_PATTERN
        many = [f"gw{i // per}/f{i % per:02d}.py" for i in range(cap + per)]
        monkeypatch.setattr(self, "PATHS", self.PATHS + many)
        patterns = [rf"^gw{k}/" for k in range((cap + per) // per)]
        with structlog.testing.capture_logs() as logs:
            candidates, _ = self._run(monkeypatch, extra_anchor_patterns=patterns)
        assert sorted(candidates & set(many)) == sorted(many)[:cap]
        assert {"app/auth.py", "app/db.py"} <= candidates
        warned = [e for e in logs if "add too many files" in e["event"]]
        assert (warned[0]["kept"], warned[0]["dropped"]) == (cap, per)


class TestExtraInputsDefaultOffIsTodaysBehaviour:
    PATHS = _big_repo_paths()

    def test_the_tool_is_identical_for_a_missing_or_malformed_answer(self, monkeypatch):
        _install(monkeypatch, self.PATHS, faithful=True)
        base = _execute(files_per_unit=8, max_units=62)
        for junk in _JUNK_EXTRAS:
            got = _execute(
                files_per_unit=8,
                max_units=62,
                extra_globs=junk,
                extra_anchor_patterns=junk,
            )
            assert got == base, junk


# Pins the default (context dials unset) selection byte-for-byte:
# ``{cap: (unit_count, signature)}`` with ``coverage_first`` +
# ``cap_aware_backfill`` + ``saturation_stop`` on. Red means the DEFAULT moved.
_DEFAULT_SELECTION_GOLDEN = {
    40: (300, "22e3beef87bba2fcac61fba04771e14357a25828323422094c90bb4ba6c344b3"),
    62: (306, "e1b04ee25bada642ae6c87f7fd5d32eafb9a2e66ecf66cd11d94a55d2efd3622"),
    150: (150, "053c6ea34edbc310ed263646f17d40e37b4808fe82803c14b69f631389a79e2a"),
}


class TestDefaultSelectionGoldenSnapshot:
    """The default selection against committed literals (live-vs-live comparisons move together)."""

    FPU = 8

    @staticmethod
    def _sig(units):
        h = hashlib.sha256()
        for u in units:
            h.update(u["name"].encode())
            h.update(b"\0")
            for f in u["files"]:
                h.update(f.encode())
                h.update(b"\1")
            h.update(b"\2")
        return h.hexdigest()

    def _select(self, cap, **kwargs):
        return select_units(
            _floor_bound_candidates(),
            files_per_unit=self.FPU,
            max_units=cap,
            coverage_first=True,
            cap_aware_backfill=True,
            saturation_stop=True,
            **kwargs,
        )

    def test_the_unset_dial_reproduces_the_pristine_selection(self):
        for cap, (expected_count, expected_sig) in _DEFAULT_SELECTION_GOLDEN.items():
            units = self._select(cap)
            assert len(units) == expected_count, cap
            assert self._sig(units) == expected_sig, cap

    def test_explicit_neutral_values_are_the_same_path(self):
        for cap, (_, expected_sig) in _DEFAULT_SELECTION_GOLDEN.items():
            neutral = self._select(
                cap, context_merge_weight=1, context_pool_multiplier=1
            )
            assert self._sig(neutral) == expected_sig, cap

    def test_the_golden_can_actually_go_red(self):
        _, expected_sig = _DEFAULT_SELECTION_GOLDEN[62]
        turned = self._select(62, context_merge_weight=4, context_pool_multiplier=4)
        assert self._sig(turned) != expected_sig


# --------------------------------------------------------------------------- #
# _search / _content_hits -- DEGRADE, never crash
# --------------------------------------------------------------------------- #
class TestSearchReadsTheNodeExecutorEnvelope:
    """The Node executor prefixes ``Exit code: N`` and cuts output over 50 KB."""

    MARKER = "[... output truncated: showing first and last 25KB of 80KB ...]"

    def _hits(self, monkeypatch, out):
        _install(monkeypatch, search=out)
        return asyncio.run(_tool()._content_hits())

    def test_a_raising_executor_yields_none_rather_than_propagating(self, monkeypatch):
        # A missing search program must degrade the scan, not end it.
        _install(monkeypatch, raises=RuntimeError("rg: not found"))
        assert asyncio.run(_tool()._search(["-l", "-e", "x", "."])) is None

    def test_blank_lines_are_not_paths(self, monkeypatch):
        out = "./app/models/user.rb\n\n   \n./app/controllers/x.rb\n"
        assert self._hits(monkeypatch, out) == {
            "app/models/user.rb",
            "app/controllers/x.rb",
        }

    def test_the_exit_code_line_is_not_a_path(self, monkeypatch):
        assert self._hits(monkeypatch, "Exit code: 0\n./a.rb\nb.rb\n") == {
            "a.rb",
            "b.rb",
        }

    def test_no_match_is_no_content_signal(self, monkeypatch):
        assert self._hits(monkeypatch, "Exit code: 1\n") is None

    def test_a_search_error_is_no_content_signal(self, monkeypatch):
        with structlog.testing.capture_logs() as logs:
            assert self._hits(monkeypatch, "Exit code: 2\n./a.rb\n") is None
        assert any(e.get("exit_code") == "2" for e in logs)

    def test_truncated_output_is_no_content_signal(self, monkeypatch):
        out = f"Exit code: 0\n./a.rb\n\n{self.MARKER}\n\n./z.rb\n"
        with structlog.testing.capture_logs() as logs:
            assert self._hits(monkeypatch, out) is None
        assert any("truncated" in e["event"] for e in logs)

    def test_truncated_output_keeps_every_candidate(self, monkeypatch):
        paths = _big_repo_paths()
        cand = bl_units.enumerate_candidates([paths])
        head, tail = sorted(cand)[:3], sorted(cand)[-3:]
        search = "\n".join(["Exit code: 0"] + head + ["", self.MARKER, ""] + tail)

        _install(monkeypatch, paths, search=search)
        filtered = _execute(files_per_unit=8, max_units=40, content_filter="true")
        unfiltered = _execute(files_per_unit=8, max_units=40)
        assert filtered == unfiltered

    def test_a_size_limited_search_is_no_content_signal(self, monkeypatch):
        out = (
            "Maximum allowed size exceeded. Result: 5000000 bytes. "
            "Maximum allowed: 4194304 bytes\nExit code: 0\n./a.rb\n"
        )
        assert self._hits(monkeypatch, out) is None


class TestSizeLimitedFileListing:
    TREE = (
        [f"app/controllers/c{i}_controller.rb" for i in range(300)]
        + [f"app/models/m{i}.rb" for i in range(300)]
        + [f"docs/page{i}.md" for i in range(3000)]
    )

    def _run(self, monkeypatch, max_bytes):
        _install(monkeypatch, self.TREE, faithful=True, max_bytes=max_bytes)
        seen = _spy_select(monkeypatch)
        with structlog.testing.capture_logs() as logs:
            _execute(files_per_unit=8, max_units=40)
        return seen["candidates"], [e for e in logs if e["log_level"] == "warning"]

    def test_a_normal_listing_is_unchanged(self, monkeypatch):
        cands, warned = self._run(monkeypatch, max_bytes=10**9)
        assert not warned
        assert "app/models/m150.rb" in cands

    def test_a_truncated_listing_is_logged_and_keeps_only_whole_paths(
        self, monkeypatch
    ):
        cands, warned = self._run(monkeypatch, max_bytes=5_000)
        assert warned and "app/models/**/*" in warned[0]["globs"]
        # The notice, the marker and the paths the cut split never become paths.
        assert set(cands) <= set(self.TREE)
        assert "app/models/m150.rb" not in cands


# --------------------------------------------------------------------------- #
# The project's file exclusion policy
# --------------------------------------------------------------------------- #
def _excluding_tool(rules):
    return BlDiscoverAndCluster(
        metadata={"outbox": object(), "project": {"exclusion_rules": rules}}
    )


class TestFileExclusionPolicy:
    def test_find_drops_excluded_paths(self, monkeypatch):
        _install(monkeypatch, "app/models/user.rb\nsecret/keys.rb\n")
        found = asyncio.run(_excluding_tool(["secret/**"])._find("*.rb"))
        assert found == ["app/models/user.rb"]

    def test_excluded_paths_reach_no_unit_and_no_log(self, monkeypatch):
        tree = [
            "app/controllers/users_controller.rb",
            "app/controllers/secret_controller.rb",
            "app/models/user.rb",
        ]
        _install(monkeypatch, tree, faithful=True)
        with structlog.testing.capture_logs() as logs:
            units = asyncio.run(
                _excluding_tool(["*secret*"])._execute(files_per_unit=8, max_units=20)
            )
        files = {f for u in units for f in u["files"]}
        assert "app/controllers/users_controller.rb" in files
        assert not any("secret" in f for f in files)
        assert not any("secret" in json.dumps(e, default=str) for e in logs)

    def test_target_files_are_filtered_without_falling_back_to_discovery(
        self, monkeypatch
    ):
        fired = _install(monkeypatch, "")
        tool = _excluding_tool(["secret/**"])
        mixed = asyncio.run(
            tool._execute(
                files_per_unit=8, max_units=5, target_files="a/one.rb secret/two.rb"
            )
        )
        assert {f for u in mixed for f in u["files"]} == {"a/one.rb"}
        # Only excluded paths: no units, and no full scan either.
        only = asyncio.run(
            tool._execute(files_per_unit=8, max_units=5, target_files="secret/two.rb")
        )
        assert only == []
        assert fired == []


class TestLogging:
    def test_project_id_and_branch_are_logged(self, monkeypatch):
        _install(monkeypatch, "")
        with structlog.testing.capture_logs() as logs:
            _execute(files_per_unit=8, max_units=5, project_id=42, branch="main")
        out = next(e for e in logs if e["event"] == "bl_discover_and_cluster output")
        assert (out["project_id"], out["branch"]) == (42, "main")

    def test_only_the_tool_class_is_exported(self):
        assert bl.__all__ == ["BlDiscoverAndCluster"]
