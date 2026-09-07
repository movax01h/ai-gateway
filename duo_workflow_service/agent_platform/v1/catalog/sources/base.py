"""The strategy a source implements to turn its items into part of a flow.

Rather than a shared pipeline with per-source branches, each source answers for itself: whether a reference is well
formed, and what the flow's components become once its items are bound.
:mod:`~duo_workflow_service.agent_platform.v1.catalog.binding` owns only what is shared, namely which component claims a
reference and whether the flow declared it.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

from duo_workflow_service.agent_platform.v1.catalog.items import CatalogItems
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry

__all__ = ["BindRequest", "CatalogSource"]


class BindRequest(NamedTuple):
    """Everything a source needs to bind one reference into one flow.

    Attributes:
        ref: The reference the claiming component declared.
        claimant_config: The claiming component's own config, an element of
            *components_config*, so a source can rewrite it by identity.
        components_config: The flow's authored component config dicts.
        items: The items the request carried, across all kinds.
        tools_registry: Resolves the tool names an item declares. The only check that an item's tools exist and
            are available to this run.
    """

    ref: CatalogItemRef
    claimant_config: dict
    components_config: list[dict]
    items: CatalogItems
    tools_registry: ToolsRegistry


class CatalogSource(ABC):
    """How one source's items of one kind are validated and built into a flow.

    One strategy per source and kind, so a source offering two kinds is two classes
    rather than one that branches on ``item_type``. A reference naming any other pair is
    refused before it reaches a strategy.

    Attributes:
        SOURCE: The source this serves.
        ITEM_TYPE: The kind it builds.
    """

    SOURCE: ClassVar[CatalogItemSource]
    ITEM_TYPE: ClassVar[CatalogItemType]

    @abstractmethod
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Reject a reference this source cannot serve.

        Called before :meth:`bind` and whether or not the request carried items, so a misconfigured flow fails the
        same way on every run.

        Args:
            ref: A reference the flow declared.

        Raises:
            CatalogItemConfigError: If the reference is not one this source accepts.
        """

    @abstractmethod
    def bind(self, request: BindRequest) -> list[dict]:
        """Build this source's items into the flow.

        Args:
            request: The reference, the claiming component, and the items to bind.

        Returns:
            The whole component list rather than an addition to it, because a source may rewrite the claiming
            component as well as add to it.

        Raises:
            CatalogItemConfigError: If the flow cannot accept these items.
            CatalogItemsError: If the items themselves cannot be used here.
        """
