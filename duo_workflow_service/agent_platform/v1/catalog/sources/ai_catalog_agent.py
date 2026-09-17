"""The AI Catalog agent source: agents a customer published to the AI Catalog.

A flow references one agent by id and version. This source never looks an agent up itself. GitLab will resolve each
referenced agent when the flow starts and include it in the start request, the same way it includes workspace agents.

No agents are included yet, so this source is a stub. Binding removes the claim and attaches nothing, so the flow
builds as authored. Attaching the resolved agents is ai-assist#2870.
"""

from typing import override

import structlog

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
    without_claim,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)

__all__ = ["AiCatalogAgentSource"]

logger = structlog.stdlib.get_logger(__name__)


class AiCatalogAgentSource(CatalogSource):
    """Agents published to the AI Catalog, referenced by id and version.

    A stub: it validates a reference and then attaches nothing. See the module docstring.
    """

    SOURCE = CatalogItemSource.AI_CATALOG
    ITEM_TYPE = CatalogItemType.AGENT

    @override
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Require an id and a version. GitLab resolves the agent itself.

        Args:
            ref: A reference the flow declared.

        Raises:
            CatalogItemConfigError: If the id or the version is missing or blank. Blank is refused as well as
                missing, because a claim is matched to a declared entry by equality, and a value that is only
                whitespace names no agent while still reading like one.
        """
        if not ref.item_id.strip():
            raise CatalogItemConfigError(
                f"include entry '{ref}' must name an agent by id."
            )

        if not (ref.version or "").strip():
            raise CatalogItemConfigError(
                f"include entry '{ref}' must declare a version."
            )

    @override
    def bind(self, request: BindRequest) -> list[dict]:
        """Remove the claim and attach nothing.

        Args:
            request: The reference, the claiming component, and the items to bind.

        Returns:
            The flow's components, with the reference removed from the claiming component. Statically named
            subagents stay, and no component is attached until ai-assist#2870.
        """
        logger.info(
            "AI Catalog agent reference accepted, but no agent is attached yet",
            ref=str(request.ref),
            claimant=request.claimant_config.get("name"),
        )

        return [
            (
                without_claim(comp_config)
                if comp_config is request.claimant_config
                else comp_config
            )
            for comp_config in request.components_config
        ]
