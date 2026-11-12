# pylint: disable=file-naming-for-tests
"""Tests for scripts/update_prompt_version.sh."""

import subprocess
from pathlib import Path

import pytest

from lib.subprompts_index import SUBPROMPTS_INDEX

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_prompt_version.sh"


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    (tmp_path / "ai_gateway/prompts/definitions").mkdir(parents=True)
    feature = tmp_path / "ai/features/cli/my_feature/prompts/base"
    feature.mkdir(parents=True)
    (feature / "1.0.0.yml").write_text("name: feature\n")
    prompts = tmp_path / "ai/features/insights/my_flow/prompts"
    (prompts / "fetch/base").mkdir(parents=True)
    (prompts / "fetch/base/1.0.0.yml").write_text("name: fetch\n")
    return tmp_path


def _run(base_dir: Path, prompt_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), "-d", str(base_dir), prompt_name],
        input="y\n",
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_bumps_a_prompt_that_an_index_lists(base_dir: Path, newline: str):
    prompts = base_dir / "ai/features/insights/my_flow/prompts"
    (prompts / SUBPROMPTS_INDEX).write_bytes(
        f"# comment{newline}my_flow_fetch: fetch  # comment{newline}".encode()
    )

    result = _run(base_dir, "my_flow_fetch")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (prompts / "fetch/base/1.0.1.yml").read_text() == "name: fetch\n"


def test_feature_id_does_not_pick_up_listed_prompts(base_dir: Path):
    prompts = base_dir / "ai/features/insights/my_flow/prompts"
    (prompts / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")

    result = _run(base_dir, "my_flow")

    assert result.returncode == 1
    assert "No prompt definitions found for 'my_flow'" in result.stdout
    assert not (prompts / "fetch/base/1.0.1.yml").exists()


def test_bumps_a_feature_prompt(base_dir: Path):
    result = _run(base_dir, "my_feature")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (base_dir / "ai/features/cli/my_feature/prompts/base/1.0.1.yml").exists()
