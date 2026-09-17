import asyncio
import functools
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from pydantic import Field

from duo_workflow_service.agent_platform.experimental.components import (
    BaseComponent,
    EndComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.components import (
    for_each as for_each_package,
)
from duo_workflow_service.agent_platform.experimental.components.for_each import (
    BRANCHES_SUBKEY,
    ERRORS_SUBKEY,
    FAILED_SUBKEY,
    ITEM_ERROR_SUBKEY,
    ITEM_INDEX_CONTEXT_KEY,
    PROCESSED_ITEMS_SUBKEY,
    PUBLISHED_SUBKEYS,
    RESULTS_SUBKEY,
    SUCCEEDED_SUBKEY,
    TERMINAL_EXCEPTIONS,
    TOTAL_ITEMS_SUBKEY,
    TRUNCATED_SUBKEY,
    AllItemsFailedError,
    ForEachComponent,
    ForEachConfig,
    TerminalRouter,
)
from duo_workflow_service.agent_platform.experimental.components.for_each import (
    component as for_each_module,
)
from duo_workflow_service.agent_platform.experimental.routers import Router
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorType
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
from duo_workflow_service.security.exceptions import SecurityException
from lib.events import GLReportingEventContext
from lib.usage_quota.errors import UsageQuotaError
from lib.usage_quota.service import InsufficientCredits

_ITEM_KEY = IOKey(target="context", subkeys=["item"])

#: How long a surviving branch waits for the failing one. Only reached when the
#: failure cancelled it, which is the bug these tests are about, so it is a
#: deadlock guard rather than a timing assumption.
_GATE_TIMEOUT_SECONDS = 10

#: How many times a counted branch hands control back before it gives up on
#: seeing its siblings. Generous, and free: every round is a bare ``sleep(0)``.
_YIELD_ROUNDS = 200


def _log_entry(content: str) -> dict[str, Any]:
    return {
        "message_type": "agent",
        "content": content,
        "timestamp": "2026-01-01T00:00:00Z",
        "status": None,
        "correlation_id": None,
        "tool_info": None,
    }


class ShoutComponent(BaseComponent):
    """One node, one output: the smallest component with a body to fan out."""

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "shout"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    #: Every state a branch was invoked with, in completion order. Shared across
    #: branches on purpose -- it is how a test sees what each branch saw.
    observed: list = Field(default_factory=list)

    def __entry_hook__(self) -> str:
        return f"{self.name}#shout"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        async def run(state: FlowState) -> dict:
            self.observed.append(state)
            # Written straight into the branch's own context dict: a parent that
            # still shows it afterwards handed out a reference, not a copy.
            state["context"]["branch_scribble"] = True
            return out.to_nested_dict(str(_ITEM_KEY.value_from_state(state)).upper())

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class _DrainingNode:
    """A node that accumulates UI entries and hands them over wholesale.

    ``UIHistory`` works this way, and it is why a compiled unit cannot be shared
    between branches: whichever branch drains first carries off every branch's
    entries.
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._logs: list[dict[str, Any]] = []

    async def run(self, state: FlowState) -> dict:
        item = _ITEM_KEY.value_from_state(state)
        self._logs.append(_log_entry(f"logged-{item}"))
        # Yield, so that every branch has appended before any branch drains: a
        # shared node then hands one branch all of the entries and the rest none.
        await asyncio.sleep(0)
        drained, self._logs = self._logs, []
        return {
            "ui_chat_log": drained,
            "context": {
                self.component_name: {
                    "drained": [entry["content"] for entry in drained]
                }
            },
        }


class DrainingComponent(BaseComponent):
    """Keeps its per-run state on the node object, as a real component does."""

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "drained"],
        ),
        IOKeyTemplate(target="ui_chat_log"),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    def __entry_hook__(self) -> str:
        return f"{self.name}#drain"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        node = _DrainingNode(self.name)
        graph.add_node(self.__entry_hook__(), node.run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class CyclingComponent(BaseComponent):
    """Keeps its cycle budget in graph state, the way an agent component does.

    Its node loops back to itself until the count reaches the item's own value, so each branch spends a different number
    of cycles.
    """

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "cycle_count"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    def __entry_hook__(self) -> str:
        return f"{self.name}#cycle"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        count_key = IOKey(
            target="context", subkeys=[self.name, "cycle_count"], optional=True
        )

        async def run(state: FlowState) -> dict:
            count = count_key.value_from_state(state) or 0
            # Hand control back, so the branches really do interleave.
            await asyncio.sleep(0)
            return count_key.to_nested_dict(count + 1)

        def route(state: FlowState) -> str:
            budget = int(_ITEM_KEY.value_from_state(state))
            if (count_key.value_from_state(state) or 0) < budget:
                return self.__entry_hook__()
            return router.route(state)

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), route)


class SummarizeComponent(BaseComponent):
    """Downstream stage: reads the collected results the fan-out published."""

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "summary"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    results_key: IOKey

    def __entry_hook__(self) -> str:
        return f"{self.name}#summarize"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        async def run(state: FlowState) -> dict:
            results = self.results_key.value_from_state(state)
            return out.to_nested_dict(",".join(entry["shout"] for entry in results))

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class FailingComponent(BaseComponent):
    """Shouts its item, or raises the exception configured for that item.

    Every branch runs against this one component object, so ``raised`` is shared
    across them: a shouting branch does not finish until the failing branch has
    raised. A run that collects those shouts therefore ran them alongside the
    failure, not before it.
    """

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "shout"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    #: Item value -> the exception raised for it. Every other item shouts.
    failures: dict[str, Any] = Field(default_factory=dict)

    #: An ``asyncio.Event``, opened by the first branch to raise. Typed loosely
    #: because a component model takes no arbitrary types.
    raised: Any = Field(default_factory=asyncio.Event)

    def __entry_hook__(self) -> str:
        return f"{self.name}#shout"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        async def run(state: FlowState) -> dict:
            item = str(_ITEM_KEY.value_from_state(state))
            if item in self.failures:
                self.raised.set()
                raise self.failures[item]
            if self.failures:
                await asyncio.wait_for(
                    self.raised.wait(), timeout=_GATE_TIMEOUT_SECONDS
                )
            return out.to_nested_dict(item.upper())

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class CountingComponent(BaseComponent):
    """Records how many of its branches are inside the body at the same time.

    A branch hands control back to the loop until ``expect`` branches have
    arrived, or until it has done so ``_YIELD_ROUNDS`` times. Set ``expect`` to
    the number of items and an uncapped fan-out reaches it -- every branch is
    inside before any leaves -- while a capped one cannot. The peak is then the
    cap itself rather than an artefact of the order the loop happened to
    schedule the branches in.
    """

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "shout"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    #: ``{"now": ..., "peak": ...}``, shared across branches on purpose.
    in_flight: dict = Field(default_factory=lambda: {"now": 0, "peak": 0})

    #: How many branches a branch waits to see before it stops waiting.
    expect: int = 1

    def __entry_hook__(self) -> str:
        return f"{self.name}#count"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        async def run(state: FlowState) -> dict:
            self.in_flight["now"] += 1
            self.in_flight["peak"] = max(self.in_flight["peak"], self.in_flight["now"])
            for _ in range(_YIELD_ROUNDS):
                if self.in_flight["now"] >= self.expect:
                    break
                await asyncio.sleep(0)
            self.in_flight["now"] -= 1
            return out.to_nested_dict(str(_ITEM_KEY.value_from_state(state)).upper())

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class ReverseOrderComponent(BaseComponent):
    """Finishes its branches in reverse input order, deliberately.

    Branch ``i`` waits for branch ``i + 1``, so the last item writes first and
    the first item writes last. Any published order that follows completion
    instead of input is then the reverse of what it should be.
    """

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "shout"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    #: One ``asyncio.Event`` per item position, created before the run starts.
    done: Any = Field(default_factory=dict)

    #: The positions, in the order the branches actually finished.
    finished: list = Field(default_factory=list)

    def __entry_hook__(self) -> str:
        return f"{self.name}#reverse"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )
        index_key = IOKey(target="context", subkeys=[self.name, ITEM_INDEX_CONTEXT_KEY])

        async def run(state: FlowState) -> dict:
            index = index_key.value_from_state(state)
            successor = self.done.get(index + 1)
            if successor is not None:
                await asyncio.wait_for(successor.wait(), timeout=_GATE_TIMEOUT_SECONDS)
            self.finished.append(index)
            self.done[index].set()
            return out.to_nested_dict(str(_ITEM_KEY.value_from_state(state)).upper())

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


class ApprovingComponent(BaseComponent):
    """Pauses inside a branch, the way a tool call awaiting approval does."""

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="context",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "approved"],
        ),
    )
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context",)

    def __entry_hook__(self) -> str:
        return f"{self.name}#approve"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        out = self._outputs[0].to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        async def run(state: FlowState) -> dict:
            answer = interrupt(f"approve {_ITEM_KEY.value_from_state(state)}?")
            return out.to_nested_dict(answer)

        graph.add_node(self.__entry_hook__(), run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)


@pytest.fixture(name="flow_type")
def flow_type_fixture() -> GLReportingEventContext:
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="identity")
def identity_fixture(flow_type, user) -> dict[str, Any]:
    return {"flow_id": "flow-1", "flow_type": flow_type, "user": user}


@pytest.fixture(name="shouter")
def shouter_fixture(identity) -> ShoutComponent:
    return ShoutComponent(name="review_one", **identity)


def fan_out(component: BaseComponent, **config: Any) -> ForEachComponent:
    settings: dict[str, Any] = {"items": "context:discover.files", "as": "context:item"}
    settings.update(config)
    return ForEachComponent.wrapping(component, ForEachConfig.model_validate(settings))


def base_state(**overrides: Any) -> FlowState:
    state: FlowState = {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {},
        "agent_context_limits": {},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def run_fan_out(fanned: ForEachComponent, items: list, **overrides: Any) -> dict:
    """Attach the fan-out to a real graph and run it end to end."""
    graph = StateGraph(FlowState)
    fanned.attach(graph, TerminalRouter())
    graph.set_entry_point(fanned.__entry_hook__())

    context = {"discover": {"files": items}, **(overrides.pop("context", None) or {})}
    state = base_state(context=context, **overrides)
    return asyncio.run(graph.compile().ainvoke(state, config={"recursion_limit": 25}))


def collected(result: dict, name: str = "review_one") -> list:
    return result["context"][name][RESULTS_SUBKEY]


def published(result: dict, subkey: str, name: str = "review_one") -> Any:
    """One leaf of what the barrier left in the component's own namespace."""
    return result["context"][name][subkey]


def results_key(fanned: ForEachComponent) -> IOKey:
    """The declared key a downstream component reads the outcomes through."""
    return next(
        key for key in fanned.outputs if (key.subkeys or [])[-1:] == [RESULTS_SUBKEY]
    )


def declared_paths(component: BaseComponent) -> set[str]:
    """The ``target:sub.keys`` strings a flow config names a component's outputs by."""
    return {
        ":".join([key.target, ".".join(key.subkeys or [])]) for key in component.outputs
    }


def written_paths(update: Any) -> set[str]:
    """The paths one node update writes, cut to the depth ``_outputs`` declares."""
    if isinstance(update, Command):
        update = update.update or {}

    paths: set[str] = set()
    for target, value in update.items():
        if target == "context":
            for name, leaves in value.items():
                paths.update(f"context:{name}.{leaf}" for leaf in leaves)
        else:
            paths.add(f"{target}:")
    return paths


def run_fan_out_with_checkpointer(fanned: ForEachComponent, items: list) -> tuple:
    """Run the fan-out on a checkpointed graph, which is what can pause."""
    graph = StateGraph(FlowState)
    fanned.attach(graph, TerminalRouter())
    graph.set_entry_point(fanned.__entry_hook__())
    compiled = graph.compile(checkpointer=InMemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}
    state = base_state(context={"discover": {"files": items}})

    async def run_to_completion():
        # A branch that never gets past the gate would otherwise hang the suite
        # rather than fail it.
        return await asyncio.wait_for(
            compiled.ainvoke(state, config=config), timeout=_GATE_TIMEOUT_SECONDS
        )

    return asyncio.run(run_to_completion()), compiled, config


def terminal_cases() -> dict[type[Exception], Exception]:
    """One raisable instance per member of ``TERMINAL_EXCEPTIONS``."""
    return {
        GraphRecursionError: GraphRecursionError("recursion limit of 25 reached"),
        InvalidRequestException: InvalidRequestException("resume carried no goal"),
        ModelError: ModelError(ModelErrorType.API_ERROR, 500, "provider is down"),
        NotifiableAgentException: NotifiableAgentException("shown to the user"),
        NotifiableException: NotifiableException("also shown to the user"),
        SecurityException: SecurityException("prompt injection in a tool result"),
        InsufficientCredits: InsufficientCredits(),
        UsageQuotaError: UsageQuotaError("quota check unavailable"),
    }


def failing(identity: dict, failures: dict[str, Any]) -> ForEachComponent:
    return fan_out(FailingComponent(name="review_one", failures=failures, **identity))


def error_record(result: dict, index: int) -> dict:
    return collected(result)[index][ITEM_ERROR_SUBKEY]


class TestFanOut:
    def test_one_branch_per_item(self, shouter):
        result = run_fan_out(fan_out(shouter), ["a", "b", "c"])

        assert len(shouter.observed) == 3
        assert len(collected(result)) == 3

    def test_results_are_a_list_in_input_order(self, shouter):
        result = run_fan_out(fan_out(shouter), ["a", "b", "c"])

        assert collected(result) == [
            {"shout": "A"},
            {"shout": "B"},
            {"shout": "C"},
        ]

    def test_the_collect_node_runs_once_for_the_whole_fan_out(self, identity):
        """The static edge collapses N branch triggers into a single task."""
        fanned = fan_out(ShoutComponent(name="review_one", **identity))
        collect_calls = []
        original = ForEachComponent._collect

        async def counting(self, state):
            collect_calls.append(state)
            return await original(self, state)

        with patch.object(ForEachComponent, "_collect", counting):
            run_fan_out(fanned, ["a", "b", "c", "d"])

        assert len(collect_calls) == 1

    def test_a_downstream_stage_reads_the_collected_results(self, shouter, identity):
        fanned = fan_out(shouter)
        summarize = SummarizeComponent(
            name="summarize", results_key=results_key(fanned), **identity
        )
        end = EndComponent(name="end", **identity)

        graph = StateGraph(FlowState)
        end.attach(graph)
        Router(from_component=fanned, to_component=summarize).attach(graph)
        Router(from_component=summarize, to_component=end).attach(graph)
        graph.set_entry_point(fanned.__entry_hook__())

        result = asyncio.run(
            graph.compile().ainvoke(
                base_state(context={"discover": {"files": ["a", "b"]}}),
                config={"recursion_limit": 25},
            )
        )

        assert result["context"]["summarize"]["summary"] == "A,B"

    def test_an_empty_item_list_publishes_no_results(self, shouter):
        result = run_fan_out(fan_out(shouter), [])

        assert collected(result) == []
        assert shouter.observed == []

    def test_items_past_max_items_are_not_run(self, shouter):
        with patch.object(for_each_module.log, "warning") as warning:
            result = run_fan_out(fan_out(shouter, max_items=2), ["a", "b", "c", "d"])

        assert collected(result) == [{"shout": "A"}, {"shout": "B"}]
        assert warning.call_args.kwargs == {
            "component": "review_one",
            "total_items": 4,
            "max_items": 2,
        }

    def test_items_that_do_not_resolve_to_a_list_are_rejected(self, shouter):
        with pytest.raises(TypeError, match="has to hold a list but holds a str"):
            run_fan_out(fan_out(shouter), "not-a-list")

    def test_a_long_offending_items_value_is_truncated_in_the_message(self, shouter):
        with pytest.raises(TypeError) as excinfo:
            run_fan_out(fan_out(shouter), "x" * 500)

        # LangGraph appends its own "During task ..." line to the cause.
        message = str(excinfo.value).splitlines()[0]
        assert message.endswith("...")
        assert len(message) < 250


class TestBranchIsolation:
    @pytest.fixture(name="noisy_result")
    def noisy_result_fixture(self, shouter) -> dict:
        """Run the fan-out over a parent state deliberately full of noise."""
        return run_fan_out(
            fan_out(shouter),
            ["a", "b"],
            conversation_history={"someone_else": [HumanMessage("parent turn")]},
            ui_chat_log=[_log_entry("parent-entry")],
            context={
                "parent_note": "visible to every branch",
                "review_one": {RESULTS_SUBKEY: ["stale"]},
            },
        )

    @pytest.mark.usefixtures("noisy_result")
    def test_a_branch_starts_with_an_empty_conversation_history(self, shouter):
        assert [state["conversation_history"] for state in shouter.observed] == [{}, {}]

    @pytest.mark.usefixtures("noisy_result")
    def test_a_branch_starts_with_an_empty_ui_chat_log(self, shouter):
        assert [state["ui_chat_log"] for state in shouter.observed] == [[], []]

    @pytest.mark.usefixtures("noisy_result")
    def test_a_branch_inherits_the_parent_context(self, shouter):
        assert all(
            state["context"]["parent_note"] == "visible to every branch"
            for state in shouter.observed
        )

    @pytest.mark.usefixtures("noisy_result")
    def test_a_branch_does_not_inherit_the_fan_outs_own_namespace(self, shouter):
        """Only the index the wrapper seeds is in there -- nothing the flow left."""
        assert all(
            set(state["context"]["review_one"]) == {ITEM_INDEX_CONTEXT_KEY}
            for state in shouter.observed
        )

    def test_a_branch_writes_into_its_own_copy_of_the_context(self, noisy_result):
        assert "branch_scribble" not in noisy_result["context"]

    def test_the_parent_ui_chat_log_is_not_repeated_once_per_branch(self, noisy_result):
        contents = [entry["content"] for entry in noisy_result["ui_chat_log"]]
        assert contents.count("parent-entry") == 1

    def test_a_branch_knows_its_own_index(self, shouter):
        run_fan_out(fan_out(shouter), ["a", "b", "c"])

        indexes = {
            state["context"]["review_one"][ITEM_INDEX_CONTEXT_KEY]
            for state in shouter.observed
        }
        assert indexes == {0, 1, 2}


class TestPerBranchCompilation:
    def test_every_branch_compiles_a_unit_of_its_own(self, shouter):
        original = for_each_module.compile_as_unit
        units = []

        def counting(component):
            unit = original(component)
            units.append(unit)
            return unit

        with patch.object(for_each_module, "compile_as_unit", counting):
            run_fan_out(fan_out(shouter), ["a", "b", "c", "d"])

        assert len(units) == 4, "expected one compiled unit per branch"
        assert len({id(unit) for unit in units}) == 4, "branches shared a unit"

    def test_a_branch_carries_off_only_its_own_ui_entries(self, identity):
        """The leak a shared compiled unit produces, as a regression test."""
        items = ["a", "b", "c", "d"]

        result = run_fan_out(
            fan_out(DrainingComponent(name="review_one", **identity)), items
        )

        assert collected(result) == [{"drained": [f"logged-{item}"]} for item in items]
        assert len(result["ui_chat_log"]) == len(items)


class TestConcurrentWrites:
    """What a branch returns has to stay in a slot of its own."""

    def test_a_branchs_own_state_does_not_reach_the_parent_namespace(
        self, shouter, identity
    ):
        result = run_fan_out(
            fan_out(CyclingComponent(name="review_one", **identity)), ["1", "2", "3"]
        )

        # The fan-out's namespace holds what it publishes and nothing else, and
        # the branch's own working keys stay inside the branch: the inner
        # graph's context is republished per branch, never merged in.
        assert set(result["context"]["review_one"]) == set(PUBLISHED_SUBKEYS) | {
            BRANCHES_SUBKEY
        }
        assert set(result["context"]) == {"discover", "review_one"}

    def test_each_branch_gets_a_cycle_budget_of_its_own(self, identity):
        """A budget kept in state is isolated by the branch payload, not by compiling.

        Three branches run the same inner component, so a budget spliced back into the parent would be one allowance
        shared three ways -- and three concurrent writes to one key besides.
        """
        result = run_fan_out(
            fan_out(CyclingComponent(name="review_one", **identity)), ["1", "2", "3"]
        )

        assert collected(result) == [
            {"cycle_count": 1},
            {"cycle_count": 2},
            {"cycle_count": 3},
        ]


class TestDelegation:
    def test_the_wrapped_components_own_keys_are_not_republished(self, shouter):
        """Only the wrapper's own writes are readable by name from the flow.

        The wrapped component declares an output of its own, so the assertion
        below says something: the wrapper drops it rather than passing it on.
        A branch's key lands nested inside one ``results`` entry, so declaring
        it here would advertise a top-level path that resolves to nothing.
        """
        fanned = fan_out(shouter)

        assert shouter.outputs
        assert declared_paths(fanned).isdisjoint(declared_paths(shouter))
        assert results_key(fanned) == IOKey(
            target="context", subkeys=["review_one", RESULTS_SUBKEY]
        )

    def test_the_wrapper_takes_the_wrapped_components_identity(self, shouter):
        fanned = fan_out(shouter)

        assert (fanned.name, fanned.flow_id, fanned.user) == (
            shouter.name,
            shouter.flow_id,
            shouter.user,
        )

    def test_a_router_pointing_at_the_component_enters_the_fan_out(
        self, shouter, identity
    ):
        fanned = fan_out(shouter)
        router = Router(
            from_component=EndComponent(name="end", **identity), to_component=fanned
        )

        assert router.route(base_state()) == fanned.__entry_hook__()
        assert router.route(base_state()).startswith("review_one#")

    def test_a_router_leaving_the_component_is_attached_to_the_collect_node(
        self, shouter, identity
    ):
        fanned = fan_out(shouter)
        graph = StateGraph(FlowState)

        Router(
            from_component=fanned, to_component=EndComponent(name="end", **identity)
        ).attach(graph)

        assert set(graph.nodes) == {
            "review_one#foreach_dispatch",
            "review_one#foreach_unit",
            "review_one#foreach_collect",
        }


class TestItemFailureContainment:
    """One item's failure is data; it does not cost the siblings their work."""

    def test_a_failing_branch_does_not_cancel_its_siblings(self, identity):
        """An uncapped ``Send`` fan-out otherwise loses every branch to one.

        The surviving branches do not return until the failing one has raised, so this is N-1 results produced
        *alongside* a failure rather than before it.
        """
        result = run_fan_out(
            failing(identity, {"b": RuntimeError("b is unreadable")}),
            ["a", "b", "c"],
        )

        assert len(collected(result)) == 3
        assert collected(result)[0] == {"shout": "A"}
        assert collected(result)[2] == {"shout": "C"}
        assert error_record(result, 1) == {
            "type": "RuntimeError",
            "message": "b is unreadable",
        }

    def test_the_fan_in_still_fires_exactly_once(self, identity):
        fanned = failing(identity, {"b": RuntimeError("boom")})
        collect_calls = []
        original = ForEachComponent._collect

        async def counting(self, state):
            collect_calls.append(state)
            return await original(self, state)

        with patch.object(ForEachComponent, "_collect", counting):
            run_fan_out(fanned, ["a", "b", "c", "d"])

        assert len(collect_calls) == 1

    def test_an_error_record_does_not_shift_the_other_items(self, identity):
        """The failing item's own slot, not the next free one."""
        result = run_fan_out(
            failing(identity, {"b": ValueError("bad")}), ["a", "b", "c"]
        )

        assert collected(result) == [
            {"shout": "A"},
            {ITEM_ERROR_SUBKEY: {"type": "ValueError", "message": "bad"}},
            {"shout": "C"},
        ]

    def test_several_items_can_fail_independently(self, identity):
        result = run_fan_out(
            failing(identity, {"a": ValueError("a bad"), "c": KeyError("c bad")}),
            ["a", "b", "c"],
        )

        assert error_record(result, 0)["type"] == "ValueError"
        assert collected(result)[1] == {"shout": "B"}
        assert error_record(result, 2)["type"] == "KeyError"

    def test_a_contained_failure_is_logged_against_its_item(self, identity):
        with patch.object(for_each_module.log, "error") as error:
            run_fan_out(failing(identity, {"b": ValueError("bad")}), ["a", "b", "c"])

        assert error.call_args.kwargs == {
            "component": "review_one",
            "item_index": 1,
            "error": "ValueError('bad')",
        }


class TestWhatStaysUncontained:
    def test_a_pause_reaches_the_flow_instead_of_becoming_a_record(self, identity):
        """``GraphBubbleUp`` subclasses ``Exception``, so it has to be let out first.

        Containing it would publish a result for an item whose tool call is still waiting on the user, and report the
        paused flow as finished.
        """
        fanned = fan_out(ApprovingComponent(name="review_one", **identity))

        result, compiled, config = run_fan_out_with_checkpointer(fanned, ["a", "b"])

        assert "__interrupt__" in result
        assert compiled.get_state(config).next == ("review_one#foreach_unit",) * 2
        assert RESULTS_SUBKEY not in result["context"]["review_one"]

    def test_each_paused_item_gets_an_interrupt_of_its_own(self, identity):
        """A branch's pause is independently resumable, so it needs its own ID."""
        fanned = fan_out(ApprovingComponent(name="review_one", **identity))

        result, _, _ = run_fan_out_with_checkpointer(fanned, ["a", "b"])

        pending = result["__interrupt__"]
        assert {paused.value for paused in pending} == {"approve a?", "approve b?"}
        assert len({paused.id for paused in pending}) == 2

    def test_a_paused_fan_out_resumes_into_real_results(self, identity):
        """The pause is a pause: the items still produce their outcomes after it."""
        fanned = fan_out(ApprovingComponent(name="review_one", **identity))
        result, compiled, config = run_fan_out_with_checkpointer(fanned, ["a", "b"])

        resumed = asyncio.run(
            compiled.ainvoke(
                Command(
                    resume={paused.id: "yes" for paused in result["__interrupt__"]}
                ),
                config=config,
            )
        )

        assert collected(resumed) == [{"approved": "yes"}, {"approved": "yes"}]

    @pytest.mark.parametrize(
        "terminal", list(terminal_cases()), ids=lambda case: case.__name__
    )
    def test_a_terminal_exception_ends_the_whole_fan_out(self, identity, terminal):
        error = terminal_cases()[terminal]

        with pytest.raises(terminal):
            run_fan_out(failing(identity, {"b": error}), ["a", "b", "c"])

    def test_every_member_of_the_terminal_tuple_is_exercised(self):
        """A member added without a case here would go untested silently."""
        assert set(terminal_cases()) == set(TERMINAL_EXCEPTIONS)

    def test_a_non_terminal_exception_is_contained(self, identity):
        result = run_fan_out(
            failing(identity, {"b": RuntimeError("just this item")}), ["a", "b", "c"]
        )

        assert error_record(result, 1)["type"] == "RuntimeError"


class TestTheAllFailedGuard:
    """Containment turns a fan-out-wide fault into N records; that stays loud."""

    def test_every_item_failing_raises(self, identity):
        failures = {
            item: ValueError("prompt_id 'nope' is not registered") for item in "abc"
        }

        with pytest.raises(AllItemsFailedError):
            run_fan_out(failing(identity, failures), ["a", "b", "c"])

    def test_the_raise_carries_a_representative_cause(self, identity):
        failures = {item: ValueError(f"{item} is not registered") for item in "abc"}

        with pytest.raises(AllItemsFailedError) as excinfo:
            run_fan_out(failing(identity, failures), ["a", "b", "c"])

        message = str(excinfo.value)
        assert "item 0 raised ValueError: a is not registered" in message
        assert "all 3 of its items" in message
        assert "1 distinct error type(s)" in message

    def test_an_empty_item_list_does_not_trip_the_guard(self, identity):
        """Zero items is zero failures, and a clean read for the next stage."""
        result = run_fan_out(failing(identity, {"a": ValueError("unused")}), [])

        assert collected(result) == []

    def test_a_single_failing_item_is_every_item_failing(self, identity):
        """A one-item fan-out publishes nothing usable, so it fails like any other.

        Exempting it would make the flow's behaviour depend on how long the list happened to be, which no flow author
        can predict.
        """
        with pytest.raises(AllItemsFailedError, match="all 1 of its items"):
            run_fan_out(failing(identity, {"a": ValueError("bad")}), ["a"])

    def test_one_surviving_item_is_enough_to_carry_on(self, identity):
        result = run_fan_out(
            failing(identity, {"a": ValueError("bad"), "c": ValueError("bad")}),
            ["a", "b", "c"],
        )

        assert collected(result)[1] == {"shout": "B"}


class TestTheConcurrencyGate:
    """The cap is per component: one wrapper instance, one semaphore, one loop."""

    @staticmethod
    def counting(identity: dict, items: list) -> CountingComponent:
        """A counter that waits for every item, so an uncapped run peaks at all of them."""
        return CountingComponent(name="review_one", expect=len(items), **identity)

    def test_the_cap_bounds_how_many_branches_are_in_flight(self, identity):
        """Five items each waiting to see five; a cap of two means it never happens."""
        items = ["a", "b", "c", "d", "e"]
        counter = self.counting(identity, items)

        result = run_fan_out(fan_out(counter, max_concurrency=2), items)

        assert counter.in_flight["peak"] == 2
        assert len(collected(result)) == 5

    def test_the_default_cap_is_the_one_that_applies_when_none_is_set(self, identity):
        """Twelve items and no configured cap: ten run at once, not twelve."""
        items = [str(index) for index in range(12)]
        counter = self.counting(identity, items)

        run_fan_out(fan_out(counter), items)

        assert counter.in_flight["peak"] == 10

    def test_a_paused_branch_leaves_the_gate_as_it_found_it(self, identity):
        """``async with`` releases on the way out, ``GraphBubbleUp`` included."""
        fanned = fan_out(
            ApprovingComponent(name="review_one", **identity), max_concurrency=1
        )

        result, _, _ = run_fan_out_with_checkpointer(fanned, ["a", "b"])

        assert {paused.value for paused in result["__interrupt__"]} == {
            "approve a?",
            "approve b?",
        }

    def test_a_capped_fan_out_resumes_on_a_fresh_instance(self, identity):
        """A resume is a new run: new components, a new gate, the same checkpoint.

        ``Flow._compile`` builds every component again per run, so the branch
        that was parked behind the pause is dispatched against a gate that has
        never been used, and the branches that finished before the pause are
        already in the checkpoint.
        """
        saver = InMemorySaver()
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}

        def compiled_fan_out():
            fanned = fan_out(
                ApprovingComponent(name="review_one", **identity), max_concurrency=1
            )
            graph = StateGraph(FlowState)
            fanned.attach(graph, TerminalRouter())
            graph.set_entry_point(fanned.__entry_hook__())
            return graph.compile(checkpointer=saver)

        paused = asyncio.run(
            compiled_fan_out().ainvoke(
                base_state(context={"discover": {"files": ["a", "b"]}}), config=config
            )
        )
        resumed = asyncio.run(
            compiled_fan_out().ainvoke(
                Command(resume={pause.id: "yes" for pause in paused["__interrupt__"]}),
                config=config,
            )
        )

        assert collected(resumed) == [{"approved": "yes"}, {"approved": "yes"}]


class TestInputOrder:
    """Branches finish in whatever order they finish; results do not."""

    def test_results_follow_input_order_when_completion_is_reversed(self, identity):
        """Eleven items, so lexicographic and numeric order disagree at 2 and 10."""
        items = [f"item-{index}" for index in range(11)]
        reverser = ReverseOrderComponent(name="review_one", **identity)
        reverser.done = {index: asyncio.Event() for index in range(len(items))}

        result = run_fan_out(fan_out(reverser, max_concurrency=11), items)

        assert reverser.finished == list(reversed(range(len(items))))
        assert collected(result) == [{"shout": item.upper()} for item in items]

    def test_the_barrier_orders_by_index_rather_than_by_insertion(self, shouter):
        """Sorting the keys as text would put item 10 between items 1 and 2."""
        fanned = fan_out(shouter)
        scrambled = {
            "10": {"shout": "K"},
            "2": {"shout": "C"},
            "0": {"shout": "A"},
        }

        update = asyncio.run(
            fanned._collect(
                base_state(context={"review_one": {BRANCHES_SUBKEY: scrambled}})
            )
        )

        assert update["context"]["review_one"][RESULTS_SUBKEY] == [
            {"shout": "A"},
            {"shout": "C"},
            {"shout": "K"},
        ]


class TestWhatTheBarrierPublishes:
    """A reader gets the outcomes, the failures and the counts to judge them by."""

    def test_the_counts_describe_the_run(self, identity):
        result = run_fan_out(
            failing(identity, {"b": ValueError("bad")}), ["a", "b", "c"]
        )

        assert published(result, TOTAL_ITEMS_SUBKEY) == 3
        assert published(result, PROCESSED_ITEMS_SUBKEY) == 3
        assert published(result, SUCCEEDED_SUBKEY) == 2
        assert published(result, FAILED_SUBKEY) == 1
        assert published(result, TRUNCATED_SUBKEY) is False

    def test_the_errors_list_names_every_failure_in_input_order(self, identity):
        result = run_fan_out(
            failing(identity, {"a": ValueError("a bad"), "c": KeyError("c bad")}),
            ["a", "b", "c"],
        )

        assert published(result, ERRORS_SUBKEY) == [
            {"index": 0, "type": "ValueError", "message": "a bad"},
            {"index": 2, "type": "KeyError", "message": "'c bad'"},
        ]

    def test_a_run_with_no_failures_publishes_an_empty_errors_list(self, shouter):
        result = run_fan_out(fan_out(shouter), ["a", "b"])

        assert published(result, ERRORS_SUBKEY) == []
        assert published(result, FAILED_SUBKEY) == 0
        assert published(result, SUCCEEDED_SUBKEY) == 2

    def test_an_empty_item_list_publishes_zeroes_rather_than_nothing(self, shouter):
        result = run_fan_out(fan_out(shouter), [])

        assert collected(result) == []
        assert published(result, TOTAL_ITEMS_SUBKEY) == 0
        assert published(result, PROCESSED_ITEMS_SUBKEY) == 0
        assert published(result, TRUNCATED_SUBKEY) is False

    def test_truncation_is_published_rather_than_only_logged(self, shouter):
        """A downstream reader can tell that 2 of 4 items never ran."""
        result = run_fan_out(fan_out(shouter, max_items=2), ["a", "b", "c", "d"])

        assert published(result, TOTAL_ITEMS_SUBKEY) == 4
        assert published(result, PROCESSED_ITEMS_SUBKEY) == 2
        assert published(result, TRUNCATED_SUBKEY) is True

    def test_a_list_within_max_items_is_not_truncated(self, shouter):
        result = run_fan_out(fan_out(shouter, max_items=4), ["a", "b", "c", "d"])

        assert published(result, TOTAL_ITEMS_SUBKEY) == 4
        assert published(result, TRUNCATED_SUBKEY) is False

    def test_the_default_runs_the_whole_list(self, shouter):
        """Nothing is dropped unless a flow asks for it to be."""
        result = run_fan_out(fan_out(shouter), ["a", "b", "c", "d"])

        assert published(result, TRUNCATED_SUBKEY) is False
        assert len(collected(result)) == 4

    def test_the_scratch_slot_is_nulled_once_the_barrier_has_read_it(self, shouter):
        """Otherwise every branch entry is checkpointed twice over."""
        result = run_fan_out(fan_out(shouter), ["a", "b"])

        assert published(result, BRANCHES_SUBKEY) is None


class TestTheItemIndex:
    def test_the_index_is_scoped_to_the_component(self, shouter):
        """A nested fan-out writes its own namespace, so the two cannot collide."""
        run_fan_out(fan_out(shouter), ["a", "b", "c"])

        assert all(
            ITEM_INDEX_CONTEXT_KEY not in state["context"] for state in shouter.observed
        )
        assert all(
            ITEM_INDEX_CONTEXT_KEY in state["context"]["review_one"]
            for state in shouter.observed
        )

    def test_the_index_is_not_part_of_the_recorded_item_result(self, shouter):
        """The body and the wrapper share the namespace; only the body's part lands."""
        result = run_fan_out(fan_out(shouter), ["a", "b"])

        assert collected(result) == [{"shout": "A"}, {"shout": "B"}]


class TestDeclaredOutputs:
    def test_outputs_declares_exactly_what_the_nodes_write(self, identity):
        """Asserted against the real writes, so a new one cannot slip past undeclared.

        The run below exercises all three nodes and both branch outcomes, which between them write every path the
        component ever writes.
        """
        fanned = failing(identity, {"b": ValueError("bad")})
        written: set[str] = set()

        def recording(original):
            # `wraps` keeps the signature LangGraph reads to decide whether a
            # node is handed the RunnableConfig as well as the state.
            @functools.wraps(original)
            async def wrapper(self, *args, **kwargs):
                update = await original(self, *args, **kwargs)
                written.update(written_paths(update))
                return update

            return wrapper

        with (
            patch.object(
                ForEachComponent, "_dispatch", recording(ForEachComponent._dispatch)
            ),
            patch.object(
                ForEachComponent, "_run_unit", recording(ForEachComponent._run_unit)
            ),
            patch.object(
                ForEachComponent, "_collect", recording(ForEachComponent._collect)
            ),
        ):
            run_fan_out(fanned, ["a", "b", "c"])

        assert written == declared_paths(fanned)

    def test_the_scratch_slot_is_declared_like_any_other_write(self, shouter):
        """A cleanup write is still a mutation, so it belongs in ``_outputs``."""
        assert f"context:review_one.{BRANCHES_SUBKEY}" in declared_paths(
            fan_out(shouter)
        )


def test_the_package_exports_the_wrapper():
    assert for_each_package.ForEachComponent is ForEachComponent
