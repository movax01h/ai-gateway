# pylint: disable=file-naming-for-tests
"""Guards for the deep_code_review_reviewer prompt definition.

Managed Claude models must run at xhigh effort (gitlab#637410). They resolve to the ``claude`` family version and are
served through LiteLLM, which reads ``provider_params.litellm.reasoning_effort``. Self-hosted models resolve to
``base``, which must not set an effort they may not support.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.config.base import PromptConfig

_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent / "deep_code_review_reviewer" / "prompts"
)
_CLAUDE_YAML_FILES = sorted((_PROMPTS_DIR / "claude").glob("*.yml"))
_BASE_YAML_FILES = sorted((_PROMPTS_DIR / "base").glob("*.yml"))


def _load(yaml_file: Path) -> PromptConfig:
    return PromptConfig.model_validate(yaml.safe_load(yaml_file.read_text()))


def test_reviewer_has_claude_and_base_yaml() -> None:
    assert _CLAUDE_YAML_FILES, f"no claude YAML under {_PROMPTS_DIR}"
    assert _BASE_YAML_FILES, f"no base YAML under {_PROMPTS_DIR}"


@pytest.mark.parametrize(
    "yaml_file", _CLAUDE_YAML_FILES, ids=[f.name for f in _CLAUDE_YAML_FILES]
)
def test_claude_reviewer_runs_at_xhigh_effort(yaml_file: Path) -> None:
    config = _load(yaml_file)

    assert config.model.provider_params is not None
    assert config.model.provider_params.litellm is not None
    assert config.model.provider_params.litellm.reasoning_effort == "xhigh"


@pytest.mark.parametrize(
    "yaml_file", _BASE_YAML_FILES, ids=[f.name for f in _BASE_YAML_FILES]
)
def test_base_reviewer_sets_no_effort(yaml_file: Path) -> None:
    assert _load(yaml_file).model.provider_params is None
