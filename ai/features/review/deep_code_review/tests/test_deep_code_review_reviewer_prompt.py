# pylint: disable=file-naming-for-tests
"""Guards for the deep_code_review_reviewer prompt definition.

The reviewer must run at xhigh effort (gitlab#637410). Deep models are served through LiteLLM, which reads
``provider_params.litellm.reasoning_effort``; without it Sonnet 5.5 silently runs at its default effort.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.config.base import PromptConfig

_BASE_DIR = (
    Path(__file__).parent.parent.parent
    / "deep_code_review_reviewer"
    / "prompts"
    / "base"
)
_BASE_YAML_FILES = sorted(_BASE_DIR.glob("*.yml"))


def test_reviewer_has_base_yaml() -> None:
    assert _BASE_YAML_FILES, f"no base YAML under {_BASE_DIR}"


@pytest.mark.parametrize(
    "yaml_file", _BASE_YAML_FILES, ids=[f.name for f in _BASE_YAML_FILES]
)
def test_reviewer_runs_at_xhigh_effort(yaml_file: Path) -> None:
    config = PromptConfig.model_validate(yaml.safe_load(yaml_file.read_text()))

    assert config.model.provider_params is not None
    assert config.model.provider_params.litellm is not None
    assert config.model.provider_params.litellm.reasoning_effort == "xhigh"
