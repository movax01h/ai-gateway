"""The AI Catalog agent source: agents a customer published to the AI Catalog.

A flow references one by id and version. GitLab resolves each referenced agent when the flow starts and sends it with
the request, the way workspace agents travel, so this source never looks an agent up itself.

Stub: no agents are sent yet, so binding drops the claim and the flow builds as though the entry were never there.
Attaching the pushed agents is ai-assist#2870.
"""

from typing import override

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import (
    _rewrite_claimant,
)

__all__ = ["AiCatalogAgentSource"]


class AiCatalogAgentSource(CatalogSource):
    """Agents published to the AI Catalog, referenced by id and version."""

    SOURCE = CatalogItemSource.AI_CATALOG
    ITEM_TYPE = CatalogItemType.AGENT

    @override
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Require an id and a version; the agent itself is GitLab's to resolve.

        Args:
            ref: A reference the flow declared.

        Raises:
            CatalogItemConfigError: If the id is empty or no version is declared.
        """
        if not ref.item_id:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must name an agent by id."
            )

        if not ref.version:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must declare a version."
            )

    @override
    def bind(self, request: BindRequest) -> list[dict]:
        """Drop the claim and leave the flow otherwise as authored.

        Args:
            request: The reference, the claiming component, and the items to bind.

        Returns:
            The flow's components with the claimant's reference removed. Nothing is attached until ai-assist#2870.
        """
        return [
            (
                _rewrite_claimant(comp_config, [])
                if comp_config is request.claimant_config
                else comp_config
            )
            for comp_config in request.components_config
        ]
