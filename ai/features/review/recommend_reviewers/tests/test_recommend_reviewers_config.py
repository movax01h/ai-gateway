# pylint: disable=file-naming-for-tests
"""Guards for the version resolution of the recommend_reviewers flow configs."""

from pathlib import Path
from unittest.mock import patch

import pytest

from duo_workflow_service.agent_platform.v1.flows import flow_config
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from lib.version import resolve_version

PROMPT_VERSIONS_DIR = (
    Path(__file__).parent.parent.parent / "recommend_reviewers_assign/prompts/base"
)


@pytest.fixture(name="discover_flow_configs", autouse=True)
def discover_flow_configs_fixture():
    # The flow lives under ai/features/, which the service registers at startup.
    # Restore the registry so tests that list flow configs never see these roots.
    with patch.dict(flow_config._FEATURE_FLOW_ROOTS):
        flow_config.discover_feature_flow_configs()
        yield


@pytest.mark.parametrize(
    ("flow_version", "expected_version", "expected_prompt_version"),
    [
        ("^3.0.0", "3.1.0", "2.1.0"),
        # 3.0.0 pins the prompt to ^2.0.0, so it also gets the compact 2.1.0 prompt.
        ("3.0.0", "3.0.0", "2.1.0"),
    ],
)
def test_config_resolution(
    flow_version: str, expected_version: str, expected_prompt_version: str
) -> None:
    """Range requests from Rails serve the stable 3.1.0 config with the compact prompt."""
    config = FlowConfig.from_yaml_config("recommend_reviewers", flow_version)
    prompt_versions = [f.stem for f in PROMPT_VERSIONS_DIR.glob("*.yml")]

    assert config.resolved_version == expected_version
    assert (
        resolve_version(prompt_versions, config.components[0]["prompt_version"])
        == expected_prompt_version
    )
