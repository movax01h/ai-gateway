"""Attach a request's catalog items to a flow shipped in this repo.

A flow lists what it accepts in an ``include`` section, and one component claims an
entry by repeating it in its ``subagents`` list::

    components:
    -   name: "developer_agent"
        subagents:
        -   source: workspace
            item_type: agent_template
            item_id: "*"

    include:
    -   source: workspace
        item_type: agent_template
        item_id: "*"

That much is all any two sources share, so it is all that lives here: which component claimed which reference, and
whether the flow declared it. What a reference means, and what its items become, is delegated to the
:class:`~duo_workflow_service.agent_platform.v1.catalog.sources.base.CatalogSource` registered for it.
"""

from typing import Any, Optional

import structlog
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import CatalogItems
from duo_workflow_service.agent_platform.v1.catalog.sources import (
    BindRequest,
    CatalogItemRef,
    source_for,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry

__all__ = ["bind_catalog_items"]

logger = structlog.stdlib.get_logger(__name__)


def _as_ref(entry: Any) -> Optional[CatalogItemRef]:
    """Read a ``subagents`` entry as a catalog item reference.

    Args:
        entry: One element of a component's ``subagents`` list.

    Returns:
        The reference, or ``None`` for a statically named subagent.

    Raises:
        CatalogItemConfigError: If the entry is shaped like a reference but malformed. Reported rather than
            skipped, so a typo cannot reach the graph builder as a subagent named after nothing.
    """
    if not isinstance(entry, dict) or "source" not in entry:
        return None

    try:
        return CatalogItemRef.model_validate(entry)
    except ValidationError as exc:
        raise CatalogItemConfigError(
            f"A subagents entry is not a usable catalog item reference: {exc}"
        ) from exc


def _claims(components_config: list[dict]) -> list[tuple[dict, CatalogItemRef]]:
    """Find the components that claim a catalog item reference.

    Args:
        components_config: The flow's authored component config dicts.

    Returns:
        One ``(config, reference)`` pair per claiming component, in config order. The config itself rather than
        its name, so a source can rewrite it by identity. A component claiming the same reference twice is
        counted once.
    """
    found: list[tuple[dict, CatalogItemRef]] = []
    for comp_config in components_config:
        for entry in comp_config.get("subagents") or []:
            ref = _as_ref(entry)
            if ref is not None:
                found.append((comp_config, ref))
                break

    return found


def _resolve_claim(
    components_config: list[dict],
    include: Optional[list[CatalogItemRef]],
    items: CatalogItems,
) -> Optional[tuple[dict, CatalogItemRef]]:
    """Find the one component that claims a reference the flow declares.

    Args:
        components_config: The flow's authored component config dicts.
        include: The flow's ``include`` section, or ``None``.
        items: The items the request carried, used only to warn when they are dropped.

    Returns:
        The claiming component's config and the reference it claimed, or ``None`` when no component claims one.
        The flow then builds as authored and any items are dropped.

    Raises:
        CatalogItemConfigError: If a claim is not declared in ``include``, or if more than one component
            claims one.
    """
    declared = include or []
    claims = _claims(components_config)

    if not claims:
        # Compared against an empty registry rather than per kind, so this covers every
        # kind without naming one.
        if items != CatalogItems():
            logger.warning(
                "Ignoring catalog items: this flow does not accept them",
                declared_includes=len(declared),
            )

        return None

    if len(claims) > 1:
        # Attaching a subagent to a supervisor rewrites its state keys, so two claimants
        # cannot share one item. Supporting this needs an instance per claimant.
        names = [comp_config.get("name") for comp_config, _ in claims]
        raise CatalogItemConfigError(
            f"Components {names} all claim catalog items, but they can currently be "
            f"coordinated by only one component per flow."
        )

    claimant_config, ref = claims[0]
    if ref not in declared:
        raise CatalogItemConfigError(
            f"Component '{claimant_config.get('name')}' claims catalog items "
            f"'{ref}', but this flow's 'include' section does not declare them. A "
            f"claim must repeat a declared entry exactly, version included."
        )

    return claimant_config, ref


def bind_catalog_items(
    components_config: list[dict],
    include: Optional[list[CatalogItemRef]],
    items: CatalogItems,
    tools_registry: ToolsRegistry,
) -> list[dict]:
    """Return the component configs a flow should build for these items.

    Args:
        components_config: The flow's authored component config dicts.
        include: The flow's ``include`` section, or ``None``.
        items: The items the request carried.
        tools_registry: Resolves the tool names an item declares. The only check that an item's tools exist
            and are available to this run.

    Returns:
        The expanded configs, or *components_config* unchanged when no component claims
        a reference — the items are then dropped rather than failing the run.

    Raises:
        CatalogItemConfigError: If the flow declares a reference no source can serve, if
            a component claims one the flow does not declare, or if more than one
            component claims one.
        CatalogItemsError: If the items themselves cannot be used with this flow.
    """
    # Every declared entry, not just the claimed one: a flow declaring something no
    # source can serve is wrong whether or not a component takes it up.
    for entry in include or []:
        source_for(entry).validate_ref(entry)

    resolved = _resolve_claim(components_config, include, items)
    if resolved is None:
        return components_config

    # The claim equals a declared entry, so it needs no second validate_ref.
    claimant_config, ref = resolved
    return source_for(ref).bind(
        BindRequest(
            ref=ref,
            claimant_config=claimant_config,
            components_config=components_config,
            items=items,
            tools_registry=tools_registry,
        )
    )
