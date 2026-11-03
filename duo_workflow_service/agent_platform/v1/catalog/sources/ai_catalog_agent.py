"""The AI Catalog agent source: agents a customer published to the AI Catalog.

The service fetches the referenced agents from GitLab before the flow is built, and GitLab resolves and checks each
one. This source matches the claiming reference to one of those agents by id and version and attaches it as a subagent.
"""

from typing import Optional, override

import structlog

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import AiCatalogAgent
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
    without_claim,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.promotion import (
    AgentComponentTemplate,
    item_prompt_config,
    promote_claimant,
    synthesized_agent_config,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)

__all__ = ["AiCatalogAgentSource", "ai_catalog_agent_prompt"]

logger = structlog.stdlib.get_logger(__name__)

_PROMPT_ID = f"{CatalogItemSource.AI_CATALOG}_{CatalogItemType.AGENT}_prompt"


def _matched_agent(
    ref: CatalogItemRef, agents: list[AiCatalogAgent]
) -> Optional[AiCatalogAgent]:
    wanted = ref.match_key()

    return next(
        (agent for agent in agents if (agent.item_id, agent.version) == wanted),
        None,
    )


def _validate_agent(agent: AiCatalogAgent, request: BindRequest) -> None:
    taken = {comp_config.get("name") for comp_config in request.components_config}

    if agent.name in taken:
        raise CatalogItemsError(
            f"AI Catalog subagent name '{agent.name}' collides with a component name "
            f"used by this flow."
        )

    try:
        request.tools_registry.toolset(list(agent.toolset))
    except ValueError as exc:
        raise CatalogItemsError(
            f"AI Catalog subagent '{agent.name}' declares an unusable tool: {exc}"
        ) from exc


def ai_catalog_agent_prompt() -> InMemoryPromptConfig:
    """Return the prompt every AI Catalog agent item is built against."""
    return item_prompt_config(_PROMPT_ID, "AI Catalog Agent")


class AiCatalogAgentSource(CatalogSource):
    """Builds an AI Catalog agent that GitLab served for this run into a subagent of the claiming component."""

    SOURCE = CatalogItemSource.AI_CATALOG
    ITEM_TYPE = CatalogItemType.AGENT

    @override
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Require an id and a version, since GitLab resolves the agent itself.

        Raises:
            CatalogItemConfigError: If the id or the version is missing or blank. A blank value is refused too, because
                bind matches the reference to a served agent by its stripped id and version, and a blank one never
                matches.
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
        """Promote the claimant to a supervisor over the agent the reference resolved to.

        When GitLab named a setting that turned the agents off, the claim is removed and nothing is attached, so
        the flow runs without its AI Catalog subagents.

        Args:
            request: The reference, the claiming component, and the agents GitLab served for this run.

        Returns:
            The flow's components, with the claimant rewritten and one component appended for the matched agent.

        Raises:
            CatalogItemConfigError: If GitLab served no agent for the claimed reference.
            CatalogItemsError: If the agent's name collides with a component of this flow, or it declares an unusable
                tool.
        """
        claimant_config = request.claimant_config
        reason = request.items.ai_catalog_skipped_reason

        if reason:
            logger.info(
                "Running without AI Catalog subagents",
                ref=str(request.ref),
                reason=reason,
            )
            return [
                (
                    without_claim(comp_config)
                    if comp_config is claimant_config
                    else comp_config
                )
                for comp_config in request.components_config
            ]

        agent = _matched_agent(request.ref, request.items.ai_catalog_agents)
        if agent is None:
            raise CatalogItemConfigError(
                f"Subagent '{request.ref.item_id}' at version '{request.ref.version}' "
                f"is not among this session's AI Catalog agents."
            )

        _validate_agent(agent, request)

        bound = [
            (
                promote_claimant(without_claim(comp_config), comp_config, [agent.name])
                if comp_config is claimant_config
                else comp_config
            )
            for comp_config in request.components_config
        ]
        bound.append(
            synthesized_agent_config(
                AgentComponentTemplate.default(_PROMPT_ID), agent, claimant_config
            )
        )

        return bound
