"""Feature-owned serving-surface declaration + collector."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

from duo_workflow_service.agent_platform import serving_surface
from duo_workflow_service.agent_platform.serving_surface import (
    ServingSurface,
    collect_serving_surfaces,
)

_DECLARATION = (
    "from duo_workflow_service.agent_platform.serving_surface import ServingSurface\n"
    "SERVING_SURFACE = [ServingSurface(transport='grpc', deployable='dws')]\n"
)


def _write_serving(
    features: Path, domain: str, feature: str, content: str = _DECLARATION
) -> Path:
    feature_dir = features / domain / feature
    feature_dir.mkdir(parents=True, exist_ok=True)
    serving = feature_dir / "serving.py"
    serving.write_text(content)
    return serving


class TestCollectServingSurfaces:
    def test_collects_declared_surface(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "insights", "demo_flow")
        # a feature with no serving.py is skipped
        (features / "cli" / "no_serving").mkdir(parents=True)

        surfaces = collect_serving_surfaces(features)
        assert surfaces == {
            "demo_flow": [ServingSurface(transport="grpc", deployable="dws")]
        }

    def test_features_dir_falls_back_without_marker(self, monkeypatch, tmp_path):
        # No marker (wheel install, faked filesystem): fall back to the
        # fixed-depth derivation instead of failing the caller.
        orphan = tmp_path / "a" / "b" / "c" / "serving_surface.py"
        orphan.parent.mkdir(parents=True)
        monkeypatch.setattr(serving_surface, "__file__", str(orphan))

        assert (
            serving_surface._features_dir()
            == (tmp_path / "a" / "ai" / "features").resolve()
        )

    def test_missing_tree_is_silent(self, tmp_path: Path):
        assert collect_serving_surfaces(tmp_path / "nope") == {}

    def test_serving_without_declaration_is_skipped(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "cli", "no_attr", content="X = 1\n")
        _write_serving(features, "cli", "empty_list", content="SERVING_SURFACE = []\n")

        assert collect_serving_surfaces(features) == {}

    def test_broken_serving_is_skipped(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_serving(
            features, "cli", "broken", content="raise RuntimeError('boom')\n"
        )
        _write_serving(features, "insights", "healthy")

        surfaces = collect_serving_surfaces(features)
        assert "broken" not in surfaces
        assert "healthy" in surfaces

    def test_symlink_escaping_root_is_skipped(self, tmp_path: Path):
        outside = tmp_path / "outside.py"
        outside.write_text(_DECLARATION)
        features = tmp_path / "ai" / "features"
        feature = features / "insights" / "escapee"
        feature.mkdir(parents=True)
        (feature / "serving.py").symlink_to(outside)

        assert collect_serving_surfaces(features) == {}

    def test_unresolvable_symlink_is_skipped(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        feature = features / "insights" / "loopy"
        feature.mkdir(parents=True)
        # a self-referential symlink cannot resolve
        (feature / "serving.py").symlink_to(feature / "serving.py")

        assert collect_serving_surfaces(features) == {}

    def test_unloadable_spec_is_skipped(self, tmp_path: Path, monkeypatch):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "insights", "specless")
        monkeypatch.setattr(
            importlib.util, "spec_from_file_location", lambda *a, **k: None
        )

        assert collect_serving_surfaces(features) == {}

    def test_spec_without_loader_is_skipped(self, tmp_path: Path, monkeypatch):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "insights", "loaderless")
        real_spec = importlib.util.spec_from_file_location

        def spec_without_loader(*args, **kwargs):
            spec = real_spec(*args, **kwargs)
            spec.loader = None
            return spec

        monkeypatch.setattr(
            importlib.util, "spec_from_file_location", spec_without_loader
        )

        assert collect_serving_surfaces(features) == {}

    @pytest.mark.parametrize(
        "content",
        [
            # not a list
            "SERVING_SURFACE = 'grpc'\n",
            # item is not a ServingSurface
            "SERVING_SURFACE = [('grpc', 'dws')]\n",
            # typo'd transport value (Literal is not enforced at runtime)
            (
                "from duo_workflow_service.agent_platform.serving_surface"
                " import ServingSurface\n"
                "SERVING_SURFACE = [ServingSurface(transport='gprc', deployable='dws')]\n"
            ),
        ],
        ids=["not_a_list", "not_a_surface", "unknown_transport"],
    )
    def test_malformed_declaration_is_skipped(self, tmp_path: Path, content: str):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "cli", "malformed", content=content)
        _write_serving(features, "insights", "healthy")

        surfaces = collect_serving_surfaces(features)
        assert "malformed" not in surfaces
        assert "healthy" in surfaces

    def test_duplicate_feature_id_raises(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        for domain in ("insights", "cli"):
            _write_serving(features, domain, "dupe_feature")

        with pytest.raises(ValueError, match="Duplicate serving-surface"):
            collect_serving_surfaces(features)

    def test_real_moved_feature_surfaces(self):
        # Deliberately sweeps the real ai/features tree: this is the executable
        # form of the DoD "every moved feature declares its baseline surface".
        surfaces = collect_serving_surfaces()
        assert surfaces.get("glab_ask_git_command") == [
            ServingSurface(transport="rest", deployable="aigw")
        ]

    def test_sys_exit_in_serving_is_skipped(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "cli", "exiter", content="import sys\nsys.exit(3)\n")
        _write_serving(features, "insights", "healthy")

        surfaces = collect_serving_surfaces(features)
        assert "exiter" not in surfaces
        assert "healthy" in surfaces

    def test_falsy_wrong_type_declaration_warns_and_skips(self, tmp_path: Path):
        features = tmp_path / "ai" / "features"
        _write_serving(features, "cli", "falsy", content="SERVING_SURFACE = {}\n")

        with mock.patch(
            "duo_workflow_service.agent_platform.serving_surface.logger"
        ) as logger:
            assert collect_serving_surfaces(features) == {}
        assert logger.warning.called
