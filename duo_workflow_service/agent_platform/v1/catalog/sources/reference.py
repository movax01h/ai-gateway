"""An ``include`` entry: what a flow says it can pull in, and from where.

A flow lists the items it accepts in an ``include`` section beside ``components`` and
``routers``::

    include:
    -   source: workspace
        item_type: agent_template
        item_id: "*"

These four fields are all any two sources have in common, so they are all that lives here. A reference is inert data:
whether it can be served, what a valid ``item_id`` looks like, and what its items become are each answered by the
:class:`~.base.CatalogSource` registered for it.
"""

from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

__all__ = ["CatalogItemRef", "CatalogItemSource", "CatalogItemType", "looks_like_ref"]


def looks_like_ref(entry: Any) -> bool:
    """Whether a ``subagents`` entry is a catalog reference rather than a statically named subagent.

    Args:
        entry: One element of a component's ``subagents`` list.

    Returns:
        ``True`` if the entry is shaped like a reference. Shared between binding and the sources so the two cannot
        disagree about which entries are claims.
    """
    return isinstance(entry, dict) and "source" in entry


class CatalogItemSource(StrEnum):
    """Where an included item comes from.

    A member here is a source this platform knows of, not one it can serve: that depends on whether a
    :class:`~.base.CatalogSource` is registered for it.
    """

    WORKSPACE = "workspace"
    AI_CATALOG = "ai-catalog"


class CatalogItemType(StrEnum):
    """What kind of item is included.

    There is no bare ``agent``: what a customer authors in their workspace is an agent *template*, which its source
    builds into a component.
    """

    AGENT_TEMPLATE = "agent_template"
    FLOW = "flow"


class CatalogItemRef(BaseModel):
    """One ``include`` entry: a reference to items a flow accepts.

    A component claims one by repeating it exactly, so equality is what links a claim to a declaration. Field-wise,
    which is pydantic's default.

    Attributes:
        source: Where the items come from.
        item_type: What kind of item is referenced.
        item_id: Which items, in whatever form the source addresses them.
        version: Which version to use. Only meaningful for a source that versions its items.
    """

    model_config = ConfigDict(extra="forbid")

    source: CatalogItemSource
    item_type: CatalogItemType
    item_id: str
    version: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.source.value}/{self.item_type.value}/{self.item_id}"
