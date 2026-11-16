"""Locating the project root and ai/features with and without the pyproject.toml marker."""

from pathlib import Path

from lib import feature_roots
from lib.feature_roots import (
    FeatureSubdir,
    default_features_dir,
    default_project_root,
    feature_subdirs,
)


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


class TestFeatureSubdirs:
    def test_lists_feature_dirs_then_shared_dirs(self, tmp_path: Path):
        features, shared = tmp_path / "ai" / "features", tmp_path / "ai" / "shared"
        for path in (
            features / "review" / "b_feature" / "schemas",
            features / "chat" / "a_feature" / "schemas",
            shared / "common" / "schemas",
            features / "chat" / "no_schemas" / "prompts",
            shared / "no_schemas",
        ):
            path.mkdir(parents=True)
        (features / "chat" / "a_file").mkdir()
        (features / "chat" / "a_file" / "schemas").write_text("not a dir")

        assert feature_subdirs("schemas", features) == [
            FeatureSubdir(
                features / "chat" / "a_feature" / "schemas", "a_feature", "chat"
            ),
            FeatureSubdir(
                features / "review" / "b_feature" / "schemas", "b_feature", "review"
            ),
            FeatureSubdir(shared / "common" / "schemas", "common", None),
        ]
        assert [found.shared for found in feature_subdirs("schemas", features)] == [
            False,
            False,
            True,
        ]

    def test_is_empty_without_trees(self, tmp_path: Path):
        assert not feature_subdirs("schemas", tmp_path / "ai" / "features")
