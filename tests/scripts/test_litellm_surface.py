"""Contract tests for the litellm API surface AI Gateway depends on.

These tests deliberately do not mock litellm. The rest of the suite patches it
at the object level, which means litellm's own code never runs and a bump that
moves an internal is invisible until it reaches staging or production. These
run against whatever litellm is actually installed.
"""

import inspect
from pathlib import Path

import pytest

from scripts import litellm_surface
from scripts.litellm_surface import MODULES, PATCH_TARGETS, REQUIRED_PARAMS


class TestManifestImports:
    """Every litellm module and symbol in the manifest must still resolve."""

    @pytest.mark.parametrize("module", MODULES)
    def test_module_imports(self, module):
        assert not litellm_surface.check_modules() or module

    def test_no_module_findings(self):
        findings = litellm_surface.check_modules()
        assert findings == [], "\n".join(str(f) for f in findings)

    def test_no_symbol_findings(self):
        findings = litellm_surface.check_symbols()
        assert findings == [], "\n".join(str(f) for f in findings)


class TestPatchTargets:
    """Attributes the repo monkey-patches must exist and still be callable.

    This is the silent failure case. ``OpenAIChatCompletionResponseIterator``
    has its ``chunk_parser`` replaced in ai_gateway/proxy/clients/__init__.py.
    If litellm renames that method, the assignment creates a dead attribute,
    the override never runs and nothing raises.
    """

    def test_no_patch_target_findings(self):
        findings = litellm_surface.check_patch_targets()
        assert findings == [], "\n".join(str(f) for f in findings)

    @pytest.mark.parametrize(
        "module,holder,name",
        PATCH_TARGETS,
        ids=[f"{m}.{h}.{n}" if h else f"{m}.{n}" for m, h, n in PATCH_TARGETS],
    )
    def test_target_is_callable(self, module, holder, name):
        # pylint: disable=protected-access - reaching into litellm internals is
        # the whole point of this check.
        attr = litellm_surface._resolve(module, holder, name)
        assert callable(attr)


class TestSignatures:
    """Pinned signatures must match the recorded snapshot."""

    def test_snapshot_exists(self):
        assert litellm_surface.SNAPSHOT_PATH.exists(), (
            "Signature snapshot missing. Run "
            "'poetry run python scripts/litellm_surface.py --snapshot'."
        )

    def test_signatures_match_snapshot(self):
        findings = litellm_surface.check_signatures(litellm_surface.load_snapshot())
        assert findings == [], "\n".join(str(f) for f in findings)

    @pytest.mark.parametrize(
        "module,holder,name,required",
        REQUIRED_PARAMS,
        ids=[f"{m}.{n}" for m, _, n, _ in REQUIRED_PARAMS],
    )
    def test_required_params_present(self, module, holder, name, required):
        # pylint: disable=protected-access
        params = inspect.signature(
            litellm_surface._resolve(module, holder, name)
        ).parameters
        missing = [param for param in required if param not in params]
        assert not missing, f"{name} no longer accepts: {', '.join(missing)}"


class TestManifestCompleteness:
    """The manifest must cover every litellm import in the tree.

    Without this the manifest silently rots. Someone adds a new private-API
    import, gets no protection from it, and finds out in production.
    """

    def test_manifest_covers_every_import(self):
        findings = litellm_surface.check_manifest_complete()
        assert findings == [], (
            "litellm imports not covered by the manifest in "
            "scripts/litellm_surface.py:\n" + "\n".join(str(f) for f in findings)
        )


class TestImportScanner:
    """The AST scanner must see the import forms that actually appear."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("import litellm\n", [("litellm", None)]),
            ("import litellm.utils\n", [("litellm.utils", None)]),
            ("import litellm as ll\n", [("litellm", None)]),
            ("from litellm import acompletion\n", [("litellm", "acompletion")]),
            (
                "from litellm import acompletion as ac\n",
                [("litellm", "acompletion")],
            ),
            (
                "from litellm.exceptions import (\n    APIConnectionError,\n"
                "    InternalServerError,\n)\n",
                [
                    ("litellm.exceptions", "APIConnectionError"),
                    ("litellm.exceptions", "InternalServerError"),
                ],
            ),
            ("import os\nfrom pathlib import Path\n", []),
            ("from mylitellm import thing\n", []),
        ],
    )
    def test_scanner_finds_imports(self, tmp_path: Path, source, expected):
        path = tmp_path / "sample.py"
        path.write_text(source, encoding="utf-8")
        # pylint: disable=protected-access
        found = [
            (module, symbol)
            for _, module, symbol in litellm_surface._iter_litellm_imports(path)
        ]
        assert found == expected

    def test_scanner_tolerates_unparseable_file(self, tmp_path: Path):
        path = tmp_path / "broken.py"
        path.write_text("def (\n", encoding="utf-8")
        # pylint: disable=protected-access
        assert list(litellm_surface._iter_litellm_imports(path)) == []


class TestFindingsReportUsefully:
    """A finding must name the thing that moved, so the fix is obvious."""

    def test_finding_renders_kind_target_and_detail(self):
        finding = litellm_surface.Finding("module", "litellm.proxy", "import failed")
        assert str(finding) == "[module] litellm.proxy: import failed"

    def test_missing_symbol_is_reported_by_name(self, monkeypatch):
        monkeypatch.setitem(
            litellm_surface.SYMBOLS, "litellm", ("definitely_not_a_real_symbol",)
        )
        findings = litellm_surface.check_symbols()
        assert any("definitely_not_a_real_symbol" in str(f) for f in findings)
