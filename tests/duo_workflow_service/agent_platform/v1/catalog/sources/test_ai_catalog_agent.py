from unittest.mock import Mock

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import CatalogItems
from duo_workflow_service.agent_platform.v1.catalog.sources.ai_catalog_agent import (
    AiCatalogAgentSource,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import BindRequest
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)

REFERENCE = {
    "source": "ai-catalog",
    "item_type": "agent",
    "item_id": "42",
    "version": "1.0.0",
}


def _ref(**overrides) -> CatalogItemRef:
    return CatalogItemRef.model_validate({**REFERENCE, **overrides})


def test_it_serves_ai_catalog_agents():
    assert AiCatalogAgentSource.SOURCE is CatalogItemSource.AI_CATALOG
    assert AiCatalogAgentSource.ITEM_TYPE is CatalogItemType.AGENT


class TestValidateRef:
    def test_an_id_with_a_version_is_accepted(self):
        AiCatalogAgentSource().validate_ref(_ref())

    @pytest.mark.parametrize("item_id", ["", "   "], ids=["empty", "whitespace"])
    def test_an_id_naming_no_agent_is_rejected(self, item_id):
        with pytest.raises(CatalogItemConfigError, match="must name an agent"):
            AiCatalogAgentSource().validate_ref(_ref(item_id=item_id))

    @pytest.mark.parametrize(
        "version", [None, "", "   "], ids=["none", "empty", "whitespace"]
    )
    def test_a_version_naming_nothing_is_rejected(self, version):
        with pytest.raises(CatalogItemConfigError, match="must declare a version"):
            AiCatalogAgentSource().validate_ref(_ref(version=version))


def test_bind_drops_the_claim_and_attaches_nothing():
    """Stub until ai-assist#2870. Statically named subagents stay and the reference goes.

    Nothing else about the claimant changes, so a flow that names an AI Catalog agent builds exactly as authored.
    """
    claimant = {
        "name": "coordinator",
        "subagents": [{"name": "static_helper"}, dict(REFERENCE)],
    }
    other = {"name": "step", "type": "DeterministicStepComponent"}

    with capture_logs() as logs:
        bound = AiCatalogAgentSource().bind(
            BindRequest(
                ref=_ref(),
                claimant_config=claimant,
                components_config=[claimant, other],
                items=CatalogItems(),
                tools_registry=Mock(),
            )
        )

    assert bound[0] == {"name": "coordinator", "subagents": [{"name": "static_helper"}]}
    assert bound[1] is other
    assert len(bound) == 2
    # The log is the only signal that a reference was accepted and then not acted on.
    assert [log for log in logs if "no agent is attached yet" in log["event"]]


def test_bind_leaves_no_empty_subagents_list():
    """With the reference gone and nothing else named, the key goes too, so the factory builds a plain agent."""
    claimant = {"name": "coordinator", "subagents": [dict(REFERENCE)]}

    bound = AiCatalogAgentSource().bind(
        BindRequest(
            ref=_ref(),
            claimant_config=claimant,
            components_config=[claimant],
            items=CatalogItems(),
            tools_registry=Mock(),
        )
    )

    assert bound == [{"name": "coordinator"}]
