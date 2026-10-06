# pylint: disable=file-naming-for-tests
"""Guards for the recommend_reviewers 3.1.0-dev config behind the Rails feature flag."""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


@pytest.mark.parametrize(
    ("flow_version", "expected_version", "expected_prompt_version"),
    [
        ("^3.0.0", "3.0.0", "^2.0.0"),
        ("3.1.0-dev", "3.1.0-dev", "2.1.0-dev"),
    ],
)
def test_dev_config_is_served_only_on_exact_request(
    flow_version: str, expected_version: str, expected_prompt_version: str
) -> None:
    """Range requests keep serving 3.0.0, so only the Rails feature flag selects the dev config."""
    config = FlowConfig.from_yaml_config("recommend_reviewers", flow_version)

    assert config.resolved_version == expected_version
    assert config.components[0]["prompt_version"] == expected_prompt_version
