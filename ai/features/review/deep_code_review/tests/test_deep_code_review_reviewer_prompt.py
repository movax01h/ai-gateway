# pylint: disable=file-naming-for-tests
"""Guards for the deep_code_review_reviewer and standard_code_review_reviewer prompt definitions.

Managed Claude models must run at xhigh effort (gitlab#637410). They resolve to the ``claude`` family version and are
served through LiteLLM, which reads ``provider_params.litellm.reasoning_effort``. Self-hosted models resolve to
``base``, which must not set an effort they may not support.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.config.base import PromptConfig

_REVIEW_DIR = Path(__file__).parent.parent.parent
_PROMPTS_DIRS = [
    _REVIEW_DIR / "deep_code_review_reviewer" / "prompts",
    _REVIEW_DIR / "standard_code_review_reviewer" / "prompts",
]
_CLAUDE_YAML_FILES = sorted(
    f for d in _PROMPTS_DIRS for f in (d / "claude").glob("*.yml")
)
_BASE_YAML_FILES = sorted(f for d in _PROMPTS_DIRS for f in (d / "base").glob("*.yml"))


def _load(yaml_file: Path) -> PromptConfig:
    return PromptConfig.model_validate(yaml.safe_load(yaml_file.read_text()))


@pytest.mark.parametrize("prompts_dir", _PROMPTS_DIRS, ids=lambda d: d.parent.name)
def test_reviewer_has_claude_and_base_yaml(prompts_dir: Path) -> None:
    assert list((prompts_dir / "claude").glob("*.yml")), (
        f"no claude YAML under {prompts_dir}"
    )
    assert list((prompts_dir / "base").glob("*.yml")), (
        f"no base YAML under {prompts_dir}"
    )


@pytest.mark.parametrize(
    "yaml_file",
    _CLAUDE_YAML_FILES,
    ids=[f"{f.parents[2].name}/{f.name}" for f in _CLAUDE_YAML_FILES],
)
def test_claude_reviewer_runs_at_xhigh_effort(yaml_file: Path) -> None:
    config = _load(yaml_file)

    assert config.model.provider_params is not None
    assert config.model.provider_params.litellm is not None
    assert config.model.provider_params.litellm.reasoning_effort == "xhigh"


@pytest.mark.parametrize(
    "yaml_file",
    _BASE_YAML_FILES,
    ids=[f"{f.parents[2].name}/{f.name}" for f in _BASE_YAML_FILES],
)
def test_base_reviewer_sets_no_effort(yaml_file: Path) -> None:
    assert _load(yaml_file).model.provider_params is None
