# pylint: disable=file-naming-for-tests
"""Guards for the shipped resolve_dependency_bump flow config.

The flow is served from both registry roots while GitLab still sends
``resolve_dependency_bump/experimental``: instances keep resolving the
experimental copy until they upgrade to the release that sends
``resolve_dependency_bump/v1``. These tests pin what the promotion must not
lose and keep the two copies from drifting apart in the meantime.
"""

from pathlib import Path

import yaml

from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW_ID = "resolve_dependency_bump"
FLOW_VERSION = "1.0.0"
AGENT_COMPONENT = "resolve_dep_bump_pipeline_fix"


def _config_path(directory: Path) -> Path:
    return directory / FLOW_ID / f"{FLOW_VERSION}.yml"


def _agent_component(config: FlowConfig) -> dict:
    return next(c for c in config.components if c.get("name") == AGENT_COMPONENT)


class TestResolveDependencyBumpConfig:
    def test_serves_an_ambient_v1_flow(self):
        config = FlowConfig.from_yaml_config(FLOW_ID, FLOW_VERSION)

        assert config.version == "v1"
        assert config.environment == "ambient"

    def test_agent_component_opts_into_web_search(self):
        """``enable_web_search`` is the flow-level half of the web-search gate.

        Dropping it would leave the flow
        resolving breaking changes without changelog/advisory lookups, and would flip the ``tools_enabled["web_search"]``
        branch of the prompt with no other signal.
        """
        config = FlowConfig.from_yaml_config(FLOW_ID, FLOW_VERSION)

        assert _agent_component(config)["enable_web_search"] is True

    def test_agent_component_declares_run_command(self):
        """The prompt branches on ``tools_enabled["run_command"]``, which is toolset membership, so the tool has to stay
        declared for the symbol-introspection guidance to render."""
        config = FlowConfig.from_yaml_config(FLOW_ID, FLOW_VERSION)

        assert "run_command" in _agent_component(config)["toolset"]

    def test_v1_copy_matches_experimental_copy_except_version(self):
        """Both roots serve this flow until every supported GitLab sends ``/v1``.

        A fix applied to one copy only would otherwise ship as a silent behavior difference between two live references.
        """
        v1_config = yaml.safe_load(_config_path(FlowConfig.DIRECTORY_PATH).read_text())
        experimental_config = yaml.safe_load(
            _config_path(ExperimentalFlowConfig.DIRECTORY_PATH).read_text()
        )

        assert v1_config.pop("version") == "v1"
        assert experimental_config.pop("version") == "experimental"
        assert v1_config == experimental_config
