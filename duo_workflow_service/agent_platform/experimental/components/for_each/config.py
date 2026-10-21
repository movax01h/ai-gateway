"""Declarative ``for_each`` configuration.

The config model alone lives here, free of any ``BaseComponent`` import, so
the wrapper and the nodes around it can depend on this module without an
import cycle.
"""

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from duo_workflow_service.agent_platform.experimental.state import IOKey

__all__ = [
    "MAX_CONCURRENCY_CEILING",
    "MAX_ITEMS_CEILING",
    "ForEachConfig",
]

#: Hard ceiling on ``max_items``. Not a product decision -- a guard rail so a typo
#: in a flow config cannot fan out unbounded.
MAX_ITEMS_CEILING = 1000

#: Hard ceiling on ``max_concurrency``. The same guard rail: a mistyped value
#: still leaves a gate rather than removing one.
MAX_CONCURRENCY_CEILING = 100


class ForEachConfig(BaseModel):
    """Fan-out configuration for a single wrapped component.

    ``items`` and ``as`` are ordinary IOKey strings, so they use exactly the
    same addressing as ``inputs``/``_outputs`` everywhere else in the platform.
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True, extra="forbid")

    #: IOKey string pointing at the list to iterate, e.g. ``"context:discover.files"``.
    items: str

    #: IOKey string under which each item is published to the wrapped component,
    #: e.g. ``"context:item"``. ``as`` is a Python keyword, hence the alias.
    as_: str = Field(alias="as")

    #: Defaults to the ceiling, so a flow drops the tail of a list only when it
    #: asks to, and what is dropped is published as ``truncated``.
    max_items: int = Field(default=MAX_ITEMS_CEILING, ge=1, le=MAX_ITEMS_CEILING)

    #: Branches of this one fan-out allowed to run at once. Not a flow-wide or
    #: fleet-wide model concurrency limit.
    max_concurrency: int = Field(default=10, ge=1, le=MAX_CONCURRENCY_CEILING)

    @field_validator("items", "as_")
    @classmethod
    def _must_be_parseable_iokey(cls, value: str) -> str:
        IOKey.parse_keys([value])  # raises with the platform's own error text
        return value

    @field_validator("as_")
    @classmethod
    def _as_must_be_a_context_namespace(cls, value: str) -> str:
        """``as`` is a WRITE target, so it cannot be any readable key.

        Each of these validates at construction and then breaks at run time:
        ``as: "ui_chat_log"`` kills every item with a ``TypeError``,
        ``as: "conversation_history"`` with an ``AttributeError``,
        ``as: "status"`` silently overwrites the branch's workflow status, and
        ``as: "context"`` replaces the whole context channel for every branch.
        A bare ``context:<name>`` lands on top of whichever component owns that
        namespace.
        """
        key = IOKey.parse_keys([value])[0]
        if key.target == "context" and key.subkeys:
            return value
        raise ValueError(
            f"for_each.as is set to '{value}', but it has to be a 'context:' path "
            f"with at least one subkey -- for example 'context:item'. Unlike an "
            f"input, 'as' is where the fan-out WRITES each item before running the "
            f"body, so it needs a namespace of its own: writing a whole channel "
            f"replaces that channel for every item, which fails mid-run for "
            f"'ui_chat_log' and 'conversation_history' and silently overwrites "
            f"'status' and 'context'."
        )

    @model_validator(mode="after")
    def _as_must_not_shadow_items(self) -> Self:
        """``as`` must not write into the namespace ``items`` reads from.

        The collision is silent data loss rather than an error: each item would
        overwrite the very list being iterated.
        """
        item_root = (self.item_key.subkeys or [""])[0]
        items_key = self.items_key
        items_subkeys = list(items_key.subkeys or [])
        producer = (
            items_subkeys[0]
            if items_key.target == "context" and items_subkeys
            else None
        )
        if producer is not None and item_root == producer:
            raise ValueError(
                f"for_each.as is set to '{self.as_}', which writes into "
                f"'context:{producer}' -- the namespace for_each.items "
                f"('{self.items}') reads from, so each item would overwrite the "
                f"list being iterated. Give the item a namespace of its own, for example "
                f"'context:item'."
            )
        return self

    @property
    def items_key(self) -> IOKey:
        """The parsed IOKey for the source list."""
        return IOKey.parse_keys([self.items])[0]

    @property
    def item_key(self) -> IOKey:
        """The parsed IOKey under which one item is published to the branch."""
        return IOKey.parse_keys([self.as_])[0]
