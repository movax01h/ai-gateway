# These tests cover a YAML flow config, so there is no module to name the file after.
# pylint: disable=file-naming-for-tests
"""A failed closing step of ``bl_security/1.0.0.yml`` must fail the run.

``DeterministicStepComponent`` records a tool failure in
``context:<step>.execution_result`` instead of raising, so only the routers can
stop the run. These tests drive the shipped router table through the real
``Flow._build_routers`` over stub steps and check where the graph ends.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, StateGraph

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.flows.base import Flow
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState
from duo_workflow_service.entities.state import WorkflowStatusEnum

CLOSING_STEPS = ("triage_collect", "adjudicate_post", "write_report")


def _config() -> FlowConfig:
    return FlowConfig.from_yaml_config("bl_security", "1.0.0")


def _router(step: str) -> dict:
    (router,) = [r for r in _config().routers if r["from"] == step]
    return router


def _stub(name: str) -> MagicMock:
    component = MagicMock(spec=BaseComponent)
    component.configure_mock(name=name)
    component.__entry_hook__ = MagicMock(return_value=name)
    component.inputs = ()
    return component


def _stub_step(name: str, failing: str | None) -> MagicMock:
    """A step that publishes only its ``execution_result``, then asks its router."""
    result = "failed" if name == failing else "success"
    step = _stub(name)

    async def run(_state):
        return {"context": {name: {"execution_result": result}}}

    def attach(graph, router):
        graph.add_node(name, run)
        graph.add_conditional_edges(name, router.route)

    step.attach = attach
    return step


def _run(failing: str | None = None) -> dict:
    config = _config()
    graph = StateGraph(FlowState)
    components = {"end": _stub("end")}
    graph.add_node("end", lambda _state: {"status": WorkflowStatusEnum.COMPLETED})
    graph.add_edge("end", END)
    for component in config.components:
        components[component["name"]] = _stub_step(component["name"], failing)

    flow = SimpleNamespace(
        _config=config,
        _workflow_id="1",
        _workflow_type=None,
        _internal_event_client=None,
    )
    # A stand-in flow and stub steps: only what the routers read.
    Flow._build_routers(flow, components, graph)  # type: ignore[arg-type]
    assert config.flow.entry_point
    graph.set_entry_point(config.flow.entry_point)

    initial: FlowState = {
        "status": WorkflowStatusEnum.NOT_STARTED,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {},
        "agent_context_limits": {},
    }
    return asyncio.run(graph.compile().ainvoke(initial))


@pytest.mark.parametrize("step", CLOSING_STEPS)
def test_a_closing_step_routes_only_on_success(step):
    router = _router(step)
    assert "to" not in router
    assert router["condition"]["input"] == f"context:{step}.execution_result"
    assert list(router["condition"]["routes"]) == ["success"]


def test_a_run_whose_steps_all_succeed_completes():
    assert _run()["status"] == WorkflowStatusEnum.COMPLETED


@pytest.mark.parametrize("step", CLOSING_STEPS)
def test_a_failed_closing_step_raises_out_of_the_graph(step):
    with pytest.raises(KeyError, match=f"Route key .*'{step}', 'execution_result'"):
        _run(failing=step)
