"""Which strategy serves which reference.

The only module that sees every source. A source or kind absent here is unusable, which is how a planned one stays
inert until its strategy exists. Adding one means writing the strategy in a module beside ``workspace_agent.py`` and
listing it in :data:`_STRATEGIES`.
"""

from typing import Mapping

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import CatalogSource
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import (
    WorkspaceAgentSource,
    agent_template_prompt,
)

__all__ = ["SOURCES", "default_prompts", "source_for"]

# The one list to extend when a source or a kind is added.
_STRATEGIES: tuple[CatalogSource, ...] = (WorkspaceAgentSource(),)

# Keyed by the pair a reference names, so the lookup is exact and no strategy has to
# decide whether a kind is its own.
SOURCES: Mapping[tuple[CatalogItemSource, CatalogItemType], CatalogSource] = {
    (strategy.SOURCE, strategy.ITEM_TYPE): strategy for strategy in _STRATEGIES
}


def default_prompts() -> list[InMemoryPromptConfig]:
    """Return the prompts the platform ships for included items.

    Returns:
        The prompts sources ship for their items. Registered for every flow, so a flow includes items
        without declaring anything for them. Listed here rather than asked of each source, because only
        a source that builds components needs one.
    """
    return [agent_template_prompt()]


def _supported() -> list[str]:
    """Return every ``source/item_type`` pair that resolves today, for error messages."""
    return sorted(f"{source.value}/{item_type.value}" for source, item_type in SOURCES)


def source_for(ref: CatalogItemRef) -> CatalogSource:
    """Return the strategy that can serve a reference.

    Args:
        ref: The reference to resolve.

    Returns:
        The strategy registered for the reference's source and kind.

    Raises:
        CatalogItemConfigError: If no strategy serves that pair.
    """
    strategy = SOURCES.get((ref.source, ref.item_type))
    if strategy is None:
        raise CatalogItemConfigError(
            f"include entry '{ref}' is not supported yet. "
            f"Supported: {', '.join(_supported())}."
        )

    return strategy
