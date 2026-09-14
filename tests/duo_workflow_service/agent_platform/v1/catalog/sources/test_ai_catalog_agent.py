from unittest.mock import Mock

import pytest

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

    def test_an_empty_id_is_rejected(self):
        with pytest.raises(CatalogItemConfigError, match="must name an agent"):
            AiCatalogAgentSource().validate_ref(_ref(item_id=""))

    def test_a_missing_version_is_rejected(self):
        with pytest.raises(CatalogItemConfigError, match="must declare a version"):
            AiCatalogAgentSource().validate_ref(_ref(version=None))


def test_bind_drops_the_claim_and_attaches_nothing():
    """Stub until ai-assist#2870: statically named subagents stay, the reference goes."""
    claimant = {
        "name": "coordinator",
        "subagents": [{"name": "static_helper"}, dict(REFERENCE)],
    }
    other = {"name": "step", "type": "DeterministicStepComponent"}

    bound = AiCatalogAgentSource().bind(
        BindRequest(
            ref=_ref(),
            claimant_config=claimant,
            components_config=[claimant, other],
            items=CatalogItems(),
            tools_registry=Mock(),
        )
    )

    assert bound[0]["subagents"] == [{"name": "static_helper"}]
    assert bound[1] is other
