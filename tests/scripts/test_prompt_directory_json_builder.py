"""Tests for scripts/prompt_directory_json_builder.py."""

import json
import subprocess
import sys
from pathlib import Path

from lib.subprompts_index import SUBPROMPTS_INDEX

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_version(directory: Path, version: str = "1.0.0") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{version}.yml").write_text("name: x\n")


def test_lists_legacy_feature_and_listed_prompts_without_site_packages(
    tmp_path: Path,
):
    _write_version(tmp_path / "ai_gateway/prompts/definitions/legacy_prompt/base")
    _write_version(tmp_path / "ai/features/cli/my_feature/prompts/base")
    prompts = tmp_path / "ai/features/insights/my_flow/prompts"
    _write_version(prompts / "fetch/base")
    _write_version(prompts / "fetch/base", "1.1.0")
    (prompts / "common/partial").mkdir(parents=True)
    (prompts / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")
    (tmp_path / "public").mkdir()
    for package in ("lib", "scripts"):
        (tmp_path / package).symlink_to(_REPO_ROOT / package)

    # -S -E: no site-packages, like the bare Python image of the `pages` job.
    subprocess.run(
        [sys.executable, "-S", "-E", "-m", "scripts.prompt_directory_json_builder"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert json.loads(
        (tmp_path / "public/prompt_directory_structure.json").read_text()
    ) == {
        "legacy_prompt": {"base": ["1.0.0"]},
        "my_feature": {"base": ["1.0.0"]},
        "my_flow_fetch": {"base": ["1.0.0", "1.1.0"]},
    }
