"""Guard for the developer flow configs that bind workspace agents.

Binding and dry-run compilation with items are covered generically in test_configs.py. This checks what those
cannot see: that the component meant to coordinate the agents is the one claiming them.
"""

import pytest

from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    looks_like_ref,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

DEVELOPER_VERSIONS = ["2.0.0", "2.0.0-interactive", "2.1.0-interactive", "3.0.0"]


@pytest.mark.parametrize("version", DEVELOPER_VERSIONS)
def test_developer_agent_claims_every_workspace_agent_the_flow_includes(version):
    """A claim that drifts from the declaration fails the build; a missing one silently drops the items."""
    config = FlowConfig.from_yaml_config("developer", version)
    developer_agent = next(
        c for c in config.components if c["name"] == "developer_agent"
    )
    claims = [
        CatalogItemRef(**entry)
        for entry in developer_agent.get("subagents") or []
        if looks_like_ref(entry)
    ]

    assert config.include is not None
    assert claims == config.include
