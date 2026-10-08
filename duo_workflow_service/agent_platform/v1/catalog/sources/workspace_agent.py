"""The workspace agent source: agent templates a customer authored in their repository.

Items arrive in full with the request rather than being fetched, so there is nothing to version and no id to address one
by: a flow references them all at once, with :data:`WILDCARD_ITEM_ID`.

Binding promotes the claiming component to a supervisor and synthesizes one ``AgentComponent`` per item beneath it. Item
prompts travel as ``literal`` inputs, never spliced into Jinja source, so client text is a template value rather than
code.
"""

from typing import override

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import (
    WorkspaceAgent,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
    without_claim,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.promotion import (
    AgentComponentTemplate,
    item_prompt_config,
    literal_input,
    promote_claimant,
    synthesized_agent_config,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)
from lib.feature_flags import FeatureFlag

__all__ = [
    "WILDCARD_ITEM_ID",
    "AgentComponentTemplate",
    "WorkspaceAgentSource",
    "agent_template_prompt",
]

# Reference every item of a kind, rather than one by id.
WILDCARD_ITEM_ID = "*"

# Named after the source and kind, so two sources offering the same kind never share a
# prompt.
_PROMPT_ID = f"{CatalogItemSource.WORKSPACE}_{CatalogItemType.AGENT_TEMPLATE}_prompt"

# Injected as a literal input on every claimant: whether the request carried any items
# for it to coordinate.
_HAS_AGENTS_VARIABLE = "has_workspace_agents"


def _validate_agents(request: BindRequest) -> None:
    """Check the items against this flow: a free name, and tools the registry can resolve.

    Args:
        request: The claim being bound.

    Raises:
        CatalogItemsError: If an item's name is taken, or it declares an unusable tool. Namespacing clears ordinary
            component names, leaving one reachable collision: a component the author declared inside the namespace.
    """
    taken = {comp_config.get("name") for comp_config in request.components_config}

    for agent in request.items.workspace_agents:
        if agent.name in taken:
            raise CatalogItemsError(
                f"Workspace agent name '{agent.name}' collides with a component "
                f"name used by this flow."
            )

        try:
            request.tools_registry.toolset(list(agent.toolset))
        except ValueError as exc:
            # The registry raises ValueError; re-raised naming the item, which it
            # cannot see.
            raise CatalogItemsError(
                f"Workspace agent '{agent.name}' declares an unusable tool: {exc}"
            ) from exc


def _rewrite_claimant(claimant_config: dict, agents: list[WorkspaceAgent]) -> dict:
    """Rewrite the claiming component for the items it resolved to.

    The catalog reference goes, statically named subagents stay, and ``has_workspace_agents`` is injected so the prompt
    can gate its delegation guidance. With items the claimant also gains a subagent entry per item and the delegation
    UI log events.

    Args:
        claimant_config: The claiming component's authored config.
        agents: The items it resolved to, possibly none.

    Returns:
        A new config; the authored one is left untouched.
    """
    rewritten = without_claim(claimant_config)
    # Optional, because every claimant gets the flag, including prompts with no
    # delegation guidance to gate.
    rewritten["inputs"] = list(claimant_config.get("inputs", [])) + [
        literal_input(
            "true" if agents else "",
            _HAS_AGENTS_VARIABLE,
            optional=True,
        )
    ]

    if not agents:
        return rewritten

    return promote_claimant(
        rewritten, claimant_config, [agent.name for agent in agents]
    )


def agent_template_prompt() -> InMemoryPromptConfig:
    """Return the prompt every workspace agent item is built against.

    Shipped here rather than declared per flow: a flow includes items without writing anything for them,
    and every item of this kind is built the same way. Letting a flow override it is a later phase.

    Returns:
        A fresh instance per call, so nothing accumulates across flows. The item's own prompt arrives as
        a value in ``item_prompt``, never spliced into the template source, so client text is data rather
        than template code. ``goal`` is the task the coordinator delegates.
    """
    return item_prompt_config(_PROMPT_ID, "Workspace Agent Template")


class WorkspaceAgentSource(CatalogSource):
    """Builds workspace agent templates into subagents of the claiming component.

    Experimental: gated by ``dap_workspace_agents``, pushed by GitLab per request. While it is off, the claimant
    builds as authored and any items the request carried are dropped.
    """

    SOURCE = CatalogItemSource.WORKSPACE
    ITEM_TYPE = CatalogItemType.AGENT_TEMPLATE
    EXPERIMENT_FLAG = FeatureFlag.DAP_WORKSPACE_AGENTS

    @override
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Reject a reference this source cannot serve.

        Args:
            ref: A reference the flow declared.

        Raises:
            CatalogItemConfigError: If a concrete id or a version is declared. Items arrive with the request, so
                there is neither to select by.
        """
        if ref.item_id != WILDCARD_ITEM_ID:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must use item_id '{WILDCARD_ITEM_ID}': "
                f"workspace items arrive with the request, so there is no id to "
                f"address one by."
            )

        if ref.version is not None:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must not declare a version: workspace items "
                f"arrive with the request, so there is no version to select."
            )

    @override
    def bind(self, request: BindRequest) -> list[dict]:
        """Promote the claimant to a supervisor over one component per item.

        Args:
            request: The reference, the claiming component, and the items to bind.

        Returns:
            The flow's components, with the claimant rewritten and one component appended per item. With no items
            the claimant builds as authored, so a flow that accepts items is unchanged by a request that sends none.
        """
        agents = request.items.workspace_agents

        _validate_agents(request)

        claimant_config = request.claimant_config
        bound = [
            (
                _rewrite_claimant(comp_config, agents)
                if comp_config is claimant_config
                else comp_config
            )
            for comp_config in request.components_config
        ]
        bound.extend(
            synthesized_agent_config(
                AgentComponentTemplate.default(_PROMPT_ID), agent, claimant_config
            )
            for agent in agents
        )

        return bound
