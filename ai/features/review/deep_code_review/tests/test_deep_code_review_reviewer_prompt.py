# pylint: disable=file-naming-for-tests
"""Guards for the deep_code_review_reviewer prompt definition.

The reviewer must run at xhigh effort on every provider path (gitlab#637410): the direct Anthropic client reads
``params.output_config``, while LiteLLM providers (Vertex, Bedrock) have that key stripped in ``Prompt.__init__`` and
read ``provider_params.litellm.reasoning_effort`` instead. A version that sets only one of them silently runs at the
model's default effort on the other path.
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
def test_reviewer_runs_at_xhigh_effort_on_every_provider_path(yaml_file: Path) -> None:
    config = PromptConfig.model_validate(yaml.safe_load(yaml_file.read_text()))

    assert config.params is not None
    assert config.params.output_config == {"effort": "xhigh"}

    assert config.model.provider_params is not None
    assert config.model.provider_params.litellm is not None
    assert config.model.provider_params.litellm.reasoning_effort == "xhigh"
