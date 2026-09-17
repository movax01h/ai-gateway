import asyncio
from typing import Optional, cast
from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.experimental.components.deterministic_step.component import (
    DeterministicStepComponent,
)
from duo_workflow_service.agent_platform.experimental.components.for_each import (
    TerminalRouter,
    compile_as_unit,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient


class EchoTool(BaseTool):
    """Real BaseTool, so a compiled unit can be invoked rather than only built."""

    name: str = "echo_tool"
    description: str = "echoes the item it is given"
    args_schema: Optional[type] = None

    def _run(self, *args, **kwargs):  # pragma: no cover - async only
        raise NotImplementedError

    async def _arun(self, *args, **kwargs) -> str:
        return f"processed:{kwargs.get('item')}"


class StubSubagent:
    """Minimal subagent, so building a supervisor needs no second real agent.

    It attaches a node of its own: the supervisor's delegation router names the
    subagent's entry hook, and ``compile()`` rejects an edge to a node that
    nothing added.
    """

    _is_subagent_component = True

    def __init__(self, name: str):
        self.name = name
        self.description = "a stub subagent"

    def bind_to_supervisor(self, **_kwargs) -> None:
        """No-op -- the stub reads no state, so it needs no key factories."""

    def attach(self, graph: StateGraph, router) -> None:
        graph.add_node(self.__entry_hook__(), lambda state: state)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)

    def __entry_hook__(self) -> str:
        return f"{self.name}#agent"


@pytest.fixture(name="flow_type")
def flow_type_fixture() -> GLReportingEventContext:
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="toolset")
def toolset_fixture() -> Toolset:
    return Toolset(pre_approved={"echo_tool"}, all_tools={"echo_tool": EchoTool()})


@pytest.fixture(name="prompt_registry")
def prompt_registry_fixture() -> Mock:
    registry = Mock(spec=LocalPromptRegistry)
    prompt = Mock()
    prompt.model = Mock()
    prompt.model.model_name = "claude-3-sonnet"
    registry.get_on_behalf.return_value = prompt
    registry.get_required_variables.return_value = set()
    return registry


@pytest.fixture(name="deterministic_step")
def deterministic_step_fixture(flow_type, user, toolset) -> DeterministicStepComponent:
    # ``inputs`` is declared as parsed IOKeys, but the raw string form is what
    # ``build_base_component`` parses; it cannot take already-parsed keys. The
    # cast is what says so to mypy.
    inputs = cast(list[IOKey | RuntimeIOKey], ["context:item"])
    return DeterministicStepComponent(
        name="review_one",
        flow_id="flow-1",
        flow_type=flow_type,
        user=user,
        tool_name="echo_tool",
        toolset=toolset,
        internal_event_client=Mock(spec=InternalEventsClient),
        inputs=inputs,
    )


@pytest.fixture(name="agent")
def agent_fixture(flow_type, user, toolset, prompt_registry) -> AgentComponent:
    return AgentComponent(
        name="review_one",
        flow_id="flow-1",
        flow_type=flow_type,
        user=user,
        prompt_id="p",
        toolset=toolset,
        prompt_registry=prompt_registry,
        internal_event_client=Mock(spec=InternalEventsClient),
    )


@pytest.fixture(name="supervisor")
def supervisor_fixture(flow_type, user, toolset, prompt_registry):
    return SupervisorAgentComponent(
        name="review_one",
        flow_id="flow-1",
        flow_type=flow_type,
        user=user,
        inputs=[],
        prompt_id="p",
        toolset=toolset,
        prompt_registry=prompt_registry,
        internal_event_client=Mock(spec=InternalEventsClient),
        schema_registry=Mock(spec=BaseResponseSchemaRegistry),
        subagents=[{"name": "developer"}],
        subagent_components={"developer": StubSubagent("developer")},
        max_delegations=5,
    )


def test_terminal_router_routes_to_end():
    assert TerminalRouter().route({}) == END


def test_terminal_router_attaches_nothing():
    graph = StateGraph(FlowState)
    TerminalRouter().attach(graph)
    assert not graph.nodes


@pytest.mark.parametrize(
    "component_name",
    ["deterministic_step", "agent", "supervisor"],
)
def test_compile_as_unit_covers_every_component_type(request, component_name):
    """One helper, three unrelated component types.

    Whether the fan-out is generic rests entirely on this: the helper touches
    only ``attach`` and ``__entry_hook__``, and both are declared on the base
    class rather than on any one component type.
    """
    component = request.getfixturevalue(component_name)

    compiled = compile_as_unit(component)

    assert isinstance(compiled, CompiledStateGraph)
    assert component.__entry_hook__() in compiled.nodes


def test_a_compiled_unit_carries_no_checkpointer_of_its_own(deterministic_step):
    """Inheriting the caller's checkpointer is what lets an interrupt bubble up."""
    assert compile_as_unit(deterministic_step).checkpointer is None


def test_the_body_terminates_instead_of_routing_onward(agent):
    """A standalone unit has no successor, so every exit edge reaches ``END``."""
    compiled = compile_as_unit(agent)

    assert {"review_one#agent", "review_one#tools", "review_one#final_response"} <= set(
        compiled.nodes
    )
    assert END in compiled.get_graph().nodes


def test_a_compiled_unit_runs_on_state_it_is_given(deterministic_step):
    """Building the graph is not enough -- the unit has to execute end to end."""
    compiled = compile_as_unit(deterministic_step)
    state: FlowState = {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {"item": "a.py"},
        "agent_context_limits": {},
    }

    result = asyncio.run(compiled.ainvoke(state, config={"recursion_limit": 10}))

    responses = result["context"]["review_one"]["tool_responses"]
    assert "processed:a.py" in str(responses)
