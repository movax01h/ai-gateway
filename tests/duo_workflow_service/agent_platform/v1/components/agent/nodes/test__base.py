import pytest

from duo_workflow_service.agent_platform.v1.components.agent.nodes._base import (
    BaseAgentComponentNode,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys, RuntimeIOKey
from duo_workflow_service.agent_platform.v1.state.base import IOKey
from duo_workflow_service.entities import WorkflowStatusEnum


class _Node(BaseAgentComponentNode):
    """A node that says which of its two paths ran."""

    def __init__(self, *, ends_recovery: bool = False):
        super().__init__(
            name="n",
            conversation_history_key=RuntimeIOKey(
                alias="history",
                factory=lambda _: IOKey(target="conversation_history", subkeys=["n"]),
            ),
        )
        self._ends = ends_recovery

    @property
    def _ends_recovery(self) -> bool:
        return self._ends

    async def _run(self, state, runtime=None):
        return "worked"

    def _skip_for_recovery(self, state):
        return "stepped aside"


def _state(status):
    return {FlowStateKeys.STATUS: status, FlowStateKeys.CONVERSATION_HISTORY: {}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "ends_recovery", "expected"),
    [
        (WorkflowStatusEnum.EXECUTION, False, "worked"),
        (WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED, False, "worked"),
        (WorkflowStatusEnum.RECOVERY, False, "stepped aside"),
        # The node that claims the message that recovers the turn is the one that ends the recovery.
        (WorkflowStatusEnum.RECOVERY, True, "worked"),
    ],
    ids=[
        "running",
        "waiting_on_an_approval",
        "recovering",
        "recovering_but_ends_the_recovery",
    ],
)
async def test_a_node_steps_aside_while_the_turn_is_being_recovered(
    status, ends_recovery, expected
):
    assert await _Node(ends_recovery=ends_recovery).run(_state(status)) == expected


@pytest.mark.asyncio
async def test_a_state_with_no_status_is_not_recovering():
    assert await _Node().run({FlowStateKeys.CONVERSATION_HISTORY: {}}) == "worked"


def test_a_node_is_not_a_subagent_until_its_component_says_so():
    node = _Node()

    assert node.is_subagent is False
    node.is_subagent = True
    assert node.is_subagent is True


@pytest.mark.asyncio
async def test_run_hands_the_node_what_it_was_given():
    class _Records(_Node):
        async def _run(self, state, runtime=None):
            self.handed = (state, runtime)
            return "worked"

    node, state, runtime = _Records(), _state(WorkflowStatusEnum.EXECUTION), object()

    await node.run(state, runtime)

    assert node.handed == (state, runtime)
