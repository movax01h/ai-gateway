"""Fan-out as a wrapper around a component.

``ForEachComponent`` delegates the component protocol to what it wraps, so no
component type needs to know ``for_each`` exists to be fanned out over a list.
"""

import asyncio
from typing import Annotated, Any, ClassVar, Optional, cast, override

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pydantic import PrivateAttr

from duo_workflow_service.agent_platform.constants import NODE_ROLE_SEPARATOR
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.config import (
    ForEachConfig,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.errors import (
    TERMINAL_EXCEPTIONS,
    AllItemsFailedError,
    failed_item_errors,
    item_error_record,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.unit import (
    compile_as_unit,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
    RuntimeIOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent as V1BaseComponent,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = [
    "BRANCHES_SUBKEY",
    "ERRORS_SUBKEY",
    "FAILED_SUBKEY",
    "ITEM_INDEX_CONTEXT_KEY",
    "PROCESSED_ITEMS_SUBKEY",
    "PUBLISHED_SUBKEYS",
    "RESULTS_SUBKEY",
    "SUCCEEDED_SUBKEY",
    "TOTAL_ITEMS_SUBKEY",
    "TRUNCATED_SUBKEY",
    "ForEachComponent",
]

log = structlog.stdlib.get_logger("for_each")

#: Sub-key of the component's own namespace carrying a branch's position in the
#: iterated list: a branch is invoked through its own ``Send`` payload, not
#: through shared state.
ITEM_INDEX_CONTEXT_KEY = "for_each_index"

#: Sub-key holding the outcome of every item that ran, in input order.
RESULTS_SUBKEY = "results"

#: Sub-key holding the failures among them, in input order.
ERRORS_SUBKEY = "errors"

#: Sub-keys holding the counts a reader needs to judge the results it got.
TOTAL_ITEMS_SUBKEY = "total_items"
PROCESSED_ITEMS_SUBKEY = "processed_items"
SUCCEEDED_SUBKEY = "succeeded"
FAILED_SUBKEY = "failed"
TRUNCATED_SUBKEY = "truncated"

#: Sub-key the branches write into. Scratch rather than output: it is keyed by
#: index because concurrent branches each need a leaf of their own, and the
#: barrier turns it into the ordered shape above and then nulls it.
BRANCHES_SUBKEY = "_branches"

#: What a later component may read, in the order a reader meets it.
PUBLISHED_SUBKEYS = (
    RESULTS_SUBKEY,
    ERRORS_SUBKEY,
    TOTAL_ITEMS_SUBKEY,
    PROCESSED_ITEMS_SUBKEY,
    SUCCEEDED_SUBKEY,
    FAILED_SUBKEY,
    TRUNCATED_SUBKEY,
)

#: A branch's UI entries reach the flow's own log, which is flat and
#: append-reduced rather than component-scoped.
_UI_CHAT_LOG_KEY = IOKey(target="ui_chat_log")

# Channels a branch never takes from its parent, even when declared. ``status``
# is the runtime's control channel. ``ui_chat_log`` is output: the fan-out adds
# each branch's log to the parent's, so an inherited copy would repeat it once
# per item.
_NOT_INHERITED = ("status", "ui_chat_log")

#: What reading an input the parent cannot satisfy raises: a missing required
#: key, a path step that is not a dict, or a runtime key's factory failing on
#: state it did not expect. The body then meets the input as missing inside
#: its branch.
_UNREADABLE = (KeyError, TypeError, AttributeError)

#: How much of an offending value an error message quotes back.
_PREVIEW_CHARS = 100


def _preview(value: Any) -> str:
    """Render a value for an error message, truncated."""
    text = repr(value)
    return text if len(text) <= _PREVIEW_CHARS else f"{text[:_PREVIEW_CHARS]}..."


def _with_subagents(component: Any) -> list[Any]:
    """``component`` and every subagent under it, at any depth."""
    subagents = getattr(component, "subagent_components", None) or {}
    return [component, *(c for sub in subagents.values() for c in _with_subagents(sub))]


class ForEachComponent(BaseComponent):
    """Runs the wrapped component's body once per item of a list.

    The topology is ``#foreach_dispatch -> Send -> #foreach_unit ->
    #foreach_collect``. The wrapper adopts the wrapped component's identity, so
    a router that names the component reaches the fan-out without knowing it is
    there.
    """

    # The wrapper reads `for_each.items` directly rather than through `inputs`,
    # so it leaves any input a flow config declared to the component below it.
    _allowed_input_targets: ClassVar[tuple[str, ...]] = tuple(
        FlowState.__annotations__.keys()
    )

    # Everything the wrapper writes, the scratch slot it clears at the barrier
    # included. The wrapped component's own keys are deliberately absent: a
    # branch's data lands nested inside one `results` entry, so naming those
    # keys here would advertise top-level paths that resolve to nothing.
    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = tuple(
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, subkey],
        )
        for subkey in (*PUBLISHED_SUBKEYS, BRANCHES_SUBKEY)
    ) + (IOKeyTemplate(target="ui_chat_log"),)

    component: BaseComponent | V1BaseComponent
    for_each: ForEachConfig

    _gate: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    @classmethod
    def wrapping(
        cls,
        component: BaseComponent | V1BaseComponent,
        for_each: ForEachConfig,
    ) -> "ForEachComponent":
        """Build a fan-out that stands in for the component it wraps.

        Routers address a component by name and enter it through
        ``__entry_hook__``, so the wrapper takes the component's identity.
        """
        for inner in _with_subagents(component):
            if isinstance(inner, ForEachComponent):
                raise ValueError(
                    f"for_each cannot be nested: '{inner.name}' already runs "
                    f"once per item."
                )

        return cls(
            component=component,
            for_each=for_each,
            name=component.name,
            flow_id=component.flow_id,
            flow_type=component.flow_type,
            user=component.user,
        )

    @override
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return f"{self.name}{NODE_ROLE_SEPARATOR}foreach_dispatch"

    @property
    def _unit_node(self) -> str:
        return f"{self.name}{NODE_ROLE_SEPARATOR}foreach_unit"

    @property
    def _collect_node(self) -> str:
        return f"{self.name}{NODE_ROLE_SEPARATOR}foreach_collect"

    def _own_key(self, subkey: str) -> IOKey:
        """One leaf of the namespace this component owns."""
        return IOKey(target="context", subkeys=[self.name, subkey])

    @property
    def _namespace_key(self) -> IOKey:
        """The whole namespace this component owns, the index included.

        The fan-out writes the index inside it, so a branch inherits nothing
        under this root: the root, not the index leaf, is what a parent key
        has to be tested against.
        """
        return IOKey(target="context", subkeys=[self.name])

    @property
    def _index_key(self) -> IOKey:
        return self._own_key(ITEM_INDEX_CONTEXT_KEY)

    @property
    def _branches_key(self) -> IOKey:
        return self._own_key(BRANCHES_SUBKEY)

    @property
    def _body_output_key(self) -> IOKey:
        """Where the wrapped body leaves its own output inside a branch.

        Optional: a body that published nothing still ran, and that is an outcome to record rather than a key to raise
        on.
        """
        return IOKey(target="context", subkeys=[self.component.name], optional=True)

    def _own_update(self, values: dict[str, Any]) -> dict[str, Any]:
        """A state update writing the named leaves of this component's namespace.

        Every write goes through an ``IOKey``, so the namespace layout is
        decided once here rather than spelled out as a nested literal at each
        of the three nodes that writes one.
        """
        update: dict[str, Any] = {}
        for subkey, value in values.items():
            update = merge_nested_dict(
                update, self._own_key(subkey).to_nested_dict(value)
            )
        return update

    @override
    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        graph.add_node(self.__entry_hook__(), self._dispatch)
        graph.add_node(self._unit_node, self._run_unit)
        graph.add_node(self._collect_node, self._collect)
        # Static, so LangGraph collapses the branch triggers into one task.
        graph.add_edge(self._unit_node, self._collect_node)
        graph.add_conditional_edges(self._collect_node, router.route)

    async def _dispatch(self, state: FlowState) -> Command:
        """Resolve the item list and start one branch per item.

        Dispatch is this node's own ``Command`` rather than a conditional edge,
        so the branches and the slot they write into are published in one
        update: an edge chooses a target but cannot write state. An empty list
        is a special case only in where it hands off to.

        ``total_items`` and ``truncated`` are published from here because here
        is where they are known -- before any branch has run. A downstream
        component can then tell that the list behind the results it reads was
        longer than the run, which a service-log warning never told it.
        """
        all_items = self._resolve_items(state)
        items = all_items[: self.for_each.max_items]
        truncated = len(items) < len(all_items)

        if truncated:
            log.warning(
                "for_each is running only the first max_items of its list",
                component=self.name,
                total_items=len(all_items),
                max_items=self.for_each.max_items,
            )

        sends = [
            Send(self._unit_node, self._branch_state(state, index, item))
            for index, item in enumerate(items)
        ]
        return Command(
            update=self._own_update(
                {
                    BRANCHES_SUBKEY: {},
                    TOTAL_ITEMS_SUBKEY: len(all_items),
                    TRUNCATED_SUBKEY: truncated,
                }
            ),
            goto=sends or self._collect_node,
        )

    def _resolve_items(self, state: FlowState) -> list[Any]:
        """Read the whole list to iterate, before ``max_items`` is applied."""
        items = self.for_each.items_key.value_from_state(state)

        if not isinstance(items, list):
            raise TypeError(
                f"for_each on component '{self.name}' reads its items from "
                f"'{self.for_each.items}', which has to hold a list but holds a "
                f"{type(items).__name__}: {_preview(items)}"
            )

        return items

    def _branch_state(self, state: FlowState, index: int, item: Any) -> FlowState:
        """Build the isolated ``FlowState`` one branch is invoked with.

        Empty channels, overlaid with whatever the body declares it reads, so
        a branch sees nothing of its siblings and carries no more of the
        parent than it needs.

        Only declared inputs, rather than a copy of the whole state, because
        every pending branch payload is checkpointed: a branch carrying the
        full parent state multiplies it by the item count, and that once took
        a real flow past the executor's 4 MiB message cap. A body that reads
        anything from the parent has to say so in ``inputs``.

        ``inputs`` is the single source of truth for what crosses into a
        branch, whichever channel it names: a component may declare inputs on
        any ``FlowState`` channel -- ``AgentComponent`` allows all of them --
        so restricting the forward to ``context`` would silently empty a
        declared ``conversation_history`` input.

        ``status`` and ``ui_chat_log`` are the exceptions. ``status`` is the
        runtime's own control channel, and a branch has to start executing, so
        it is pinned here. ``ui_chat_log`` is output: each branch's log is added
        to the parent's, so an inherited copy would repeat it once per item.

        The item and the index are written by the fan-out itself, so the
        inputs that address them are not looked up in the parent. That also
        leaves the component's own namespace free for the index.
        """
        branch: dict[str, Any] = {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
            "agent_context_limits": {},
        }
        branch = merge_nested_dict(branch, self._inherited_state(state))

        context = merge_nested_dict(
            branch["context"], self.for_each.item_key.to_nested_dict(item)["context"]
        )
        branch["context"] = merge_nested_dict(
            context, self._index_key.to_nested_dict(index)["context"]
        )
        return cast(FlowState, branch)

    def _inherited_state(self, state: FlowState) -> dict[str, Any]:
        """The slice of the parent state the body's declared inputs address.

        Each input is rebuilt as a nested dict holding just its own leaf under
        its own channel, and the leaves are deep-merged, so two inputs under
        one root arrive together.

        Every channel an input may name is carried, not just ``context``: a
        component decides its own ``_allowed_input_targets``, and an
        ``AgentComponent`` allows all of them, so a declared
        ``conversation_history:`` input has to cross into the branch as much
        as a ``context:`` one does. ``status`` and ``ui_chat_log`` are never
        carried; see ``_branch_state``.

        An input the parent cannot satisfy is skipped rather than raised on,
        so it never fails the whole fan-out: a key that is absent, a path that
        runs into a non-dict on the way, or a ``RuntimeIOKey`` whose factory
        fails against the parent. The body then reads that input as missing
        inside its branch. A required one fails that item alone, as it did when
        the whole context was copied; a non-dict path now fails there as a
        missing key rather than a ``TypeError``. An optional one reads as
        ``None``, unless another input carried the same non-dict root.
        """
        inherited: dict[str, Any] = {}
        for declared in self._body_inputs():
            if declared.literal:
                continue

            try:
                key = (
                    declared.to_iokey(state)
                    if isinstance(declared, RuntimeIOKey)
                    else declared
                )
            except _UNREADABLE:
                continue
            if key.target in _NOT_INHERITED or self._written_by_fan_out(key):
                continue

            if not key.subkeys:
                # A bare channel input reads all of it, less every namespace
                # the fan-out writes: a stale dict item would otherwise
                # deep-merge into the new one.
                inherited = merge_nested_dict(
                    inherited, {key.target: self._whole_channel(state, key.target)}
                )
                continue

            try:
                value = key.value_from_state(state)
            except _UNREADABLE:
                continue
            if value is None and key.optional:
                continue

            inherited = merge_nested_dict(inherited, key.to_nested_dict(value))
        return inherited

    def _whole_channel(self, state: FlowState, target: str) -> dict[str, Any]:
        """One channel of the parent, less anything the fan-out writes itself.

        Only dict channels get here: ``status`` and ``ui_chat_log`` are never inherited.
        """
        return {
            name: entry
            for name, entry in cast(dict[str, Any], state.get(target) or {}).items()
            if not self._written_by_fan_out(IOKey(target=target, subkeys=[name]))
        }

    def _body_inputs(self) -> list[IOKey | RuntimeIOKey]:
        """The inputs of the body and of every subagent under it.

        A supervisor runs its subagents on the branch state too, at any depth, and their inputs are not part of its own,
        so they are collected as well.
        """
        return [
            key
            for component in _with_subagents(self.component)
            for key in getattr(component, "inputs", [])
        ]

    def _written_by_fan_out(self, key: IOKey) -> bool:
        """Whether ``key`` addresses an item, or the namespace holding the index."""
        item = self.for_each.item_key
        # A key above the item (`context:item` for `as: context:item.value`)
        # would carry a stale copy of it into the branch.
        return (
            key in item
            or key in self._namespace_key
            or (bool(key.subkeys) and item in key)
        )

    def _concurrency_gate(self) -> asyncio.Semaphore:
        """The semaphore bounding how many branches of this fan-out run at once.

        Per component by construction: there is one wrapper instance per
        component per flow run, and every ``Send`` branch of one superstep runs
        on that run's single event loop.

        Bound on first use rather than at construction, because a component is
        built inside ``asyncio.to_thread``, in a thread with no running loop.
        The check below cannot interleave -- it reaches the assignment with no
        ``await`` in between -- so one branch creates it and the rest find it.

        ``step_timeout`` must stay unset on the compiled graph, and nothing in
        the service sets it: it bounds a whole superstep, and a branch parked
        here extends that superstep one for one, so it would fail exactly the
        capped fan-outs and leave the uncapped ones alone.

        An approval inside a capped fan-out is reached after roughly
        ``ceil(N / max_concurrency)`` unit runs rather than one. That is
        inherent to capping a fan-out at all, LangGraph's own cap included.
        """
        if self._gate is None:
            self._gate = asyncio.Semaphore(self.for_each.max_concurrency)
        return self._gate

    async def _run_unit(
        self, state: FlowState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run the wrapped component's body over one item.

        The body is compiled per branch, not once in ``attach``: a compiled
        unit's nodes own mutable state -- ``UIHistory`` drains its accumulated
        UI entries wholesale, ``AgentNode`` holds a ``ModelErrorHandler``
        counting retries -- and branches sharing a unit share it. The compile
        sits inside the gate so a parked branch holds no compiled graph while
        it waits.

        An item's own failure is returned as data rather than raised: an
        uncapped ``Send`` fan-out cancels every sibling around whichever branch
        raises, so one item's exception costs the whole list's work.
        """
        index = self._index_key.value_from_state(state)

        async with self._concurrency_gate():
            unit = compile_as_unit(self.component)

            try:
                result = await self._invoke_unit(unit, state, config)
            except GraphBubbleUp:
                # LangGraph's own control flow -- an `interrupt()` awaiting the user,
                # or a parent-targeted `Command`. It subclasses plain `Exception`, so
                # catching it below would report a paused flow as a finished one.
                raise
            except TERMINAL_EXCEPTIONS:
                # Not this item's failure to answer for -- see `TERMINAL_EXCEPTIONS`.
                raise
            except Exception as error:  # pylint: disable=broad-except
                log.error(
                    "for_each is recording a failed item and continuing",
                    component=self.name,
                    item_index=index,
                    error=repr(error),
                )
                # No `ui_chat_log`: `ainvoke` raised, so the branch's state -- and
                # every UI entry it produced -- never came back.
                return self._branch_update(index, item_error_record(error))

        produced = dict(self._body_output_key.value_from_state(result) or {})
        # The body writes into `context:<name>` and so does the index, so the
        # item's own outcome is what is left once the index is taken back out.
        produced.pop(ITEM_INDEX_CONTEXT_KEY, None)

        return merge_nested_dict(
            self._branch_update(index, produced),
            _UI_CHAT_LOG_KEY.to_nested_dict(
                _UI_CHAT_LOG_KEY.value_from_state(result) or []
            ),
        )

    def _branch_update(self, index: int, entry: dict[str, Any]) -> dict[str, Any]:
        """The update that puts one item's outcome in the scratch slot reserved for it.

        Success and failure go to the same place, so the barrier sees one entry per item whatever each item did.
        """
        return self._own_update({BRANCHES_SUBKEY: {str(index): entry}})

    async def _invoke_unit(
        self, unit: CompiledStateGraph, state: FlowState, config: RunnableConfig
    ) -> FlowState:
        """Run one compiled branch to completion.

        The single point at which a branch's execution is entered, so anything that has to hold for exactly the span of
        one branch has one place to go.
        """
        # A unit's state schema is `FlowState`; `ainvoke` is untyped at this seam.
        return cast(FlowState, await unit.ainvoke(state, config=config))

    async def _collect(self, state: FlowState) -> dict[str, Any]:
        """Fan the branches back in, and publish what the fan-out produced.

        This is the barrier: every branch has written its scratch slot by the
        time it runs, so it is the first point at which the whole outcome
        exists. It publishes those entries as a list in input order -- branches
        finish in whatever order they finish, and a reader should not have to
        re-sort them -- plus the failures as one list, and the counts. The
        scratch slot is nulled so the same data is not checkpointed twice.

        A failure on *every* item is a fault in the fan-out itself, not data
        about the items, and returning it as results would hand a downstream
        stage N error records to succeed on. Zero items is not that case: an
        empty list has nothing to fail, and the flow carries on.

        Raises:
            AllItemsFailedError: If every item of a non-empty list failed.
        """
        branches = self._branches_key.value_from_state(state) or {}
        errors = failed_item_errors(branches)

        if branches and len(errors) == len(branches):
            raise AllItemsFailedError.for_component(self.name, errors)

        ordered = sorted(branches, key=int)
        return self._own_update(
            {
                RESULTS_SUBKEY: [branches[index] for index in ordered],
                ERRORS_SUBKEY: [
                    {"index": int(index), **errors[index]}
                    for index in ordered
                    if index in errors
                ],
                PROCESSED_ITEMS_SUBKEY: len(branches),
                SUCCEEDED_SUBKEY: len(branches) - len(errors),
                FAILED_SUBKEY: len(errors),
                BRANCHES_SUBKEY: None,
            }
        )
