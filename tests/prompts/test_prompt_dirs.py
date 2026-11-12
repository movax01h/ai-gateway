# pylint: disable=file-naming-for-tests
"""Tests for the prompt directory helpers, on synthetic trees."""

from pathlib import Path

import pytest

from lib.subprompts_index import SUBPROMPTS_INDEX
from tests.prompts import prompt_dirs


@pytest.fixture
def features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    features = tmp_path / "ai" / "features"
    monkeypatch.setattr(prompt_dirs, "_FEATURES_DIR", features)
    monkeypatch.setattr(prompt_dirs, "_SHARED_DIR", tmp_path / "ai" / "shared")
    return features


def _write_version(directory: Path) -> None:
    (directory / "base").mkdir(parents=True)
    (directory / "base" / "1.0.0.yml").write_text("name: x\n")


class TestFeaturePromptDirs:
    def test_maps_listed_prompts_to_their_dirs(self, features: Path):
        prompts = features / "insights" / "my_flow" / "prompts"
        _write_version(prompts / "fetch")
        (prompts / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")
        _write_version(features / "cli" / "my_feature" / "prompts")

        assert prompt_dirs.feature_prompt_dirs() == {
            "my_flow_fetch": prompts / "fetch",
            "my_feature": features / "cli" / "my_feature" / "prompts",
            "cli/my_feature": features / "cli" / "my_feature" / "prompts",
        }


class TestFeatureDirOf:
    @pytest.mark.parametrize(
        "relative", ["insights/my_flow/prompts", "insights/my_flow/prompts/fetch"]
    )
    def test_returns_the_owning_feature_dir(self, features: Path, relative: str):
        assert prompt_dirs.feature_dir_of(features / relative) == (
            features / "insights" / "my_flow"
        )

    def test_rejects_a_dir_above_a_feature(self, features: Path):
        with pytest.raises(ValueError, match="is not under"):
            prompt_dirs.feature_dir_of(features / "insights" / "my_flow")
