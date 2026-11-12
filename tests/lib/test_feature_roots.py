"""Locating the project root and ai/features with and without the pyproject.toml marker."""

from pathlib import Path

from lib import feature_roots
from lib.feature_roots import default_features_dir, default_project_root


class TestDefaultProjectRoot:
    def test_is_the_repo_root(self):
        root = default_project_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "ai" / "features").is_dir()

    def test_falls_back_without_marker(self, monkeypatch, tmp_path: Path):
        orphan = tmp_path / "a" / "b" / "feature_roots.py"
        orphan.parent.mkdir(parents=True)
        monkeypatch.setattr(feature_roots, "__file__", str(orphan))

        assert default_project_root() == (tmp_path / "a").resolve()


class TestDefaultFeaturesDir:
    def test_is_the_repo_tree(self):
        default = default_features_dir()
        assert default.name == "features"
        assert default.parent.name == "ai"
        assert default.is_dir()
        assert (default.parent.parent / "pyproject.toml").is_file()

    def test_falls_back_without_marker(self, monkeypatch, tmp_path: Path):
        # No marker (wheel install, faked filesystem): fall back to the
        # fixed-depth derivation instead of failing the caller.
        orphan = tmp_path / "a" / "b" / "feature_roots.py"
        orphan.parent.mkdir(parents=True)
        monkeypatch.setattr(feature_roots, "__file__", str(orphan))

        assert default_features_dir() == (tmp_path / "a" / "ai" / "features").resolve()
