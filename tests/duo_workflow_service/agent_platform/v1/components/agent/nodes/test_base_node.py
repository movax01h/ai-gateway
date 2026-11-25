# pylint: disable=file-naming-for-tests
import pytest

from duo_workflow_service.agent_platform.v1.components.agent.nodes._base import (
    BaseAgentComponentNode,
)
from duo_workflow_service.agent_platform.v1.state import RuntimeIOKey
from duo_workflow_service.agent_platform.v1.state.base import IOKey


class _Node(BaseAgentComponentNode):
    """A node that records what its work was handed."""

    def __init__(self, **kwargs):
        super().__init__(
            name="n",
            conversation_history_key=RuntimeIOKey(
                alias="history",
                factory=lambda _: IOKey(target="conversation_history", subkeys=["n"]),
            ),
            **kwargs,
        )
        self.handed = None

    async def _run(self, state, runtime=None):
        self.handed = (state, runtime)
        return "worked"


class TestBaseAgentComponentNode:
    @pytest.mark.asyncio
    async def test_run_does_the_nodes_work_with_what_it_was_given(self):
        node, state, runtime = _Node(), {"status": "execution"}, object()

        result = await node.run(state, runtime)

        assert result == "worked"
        assert node.handed == (state, runtime)

    @pytest.mark.asyncio
    async def test_run_needs_no_runtime(self):
        node = _Node()

        await node.run({})

        assert node.handed == ({}, None)

    def test_the_node_reads_its_components_history_key(self):
        state = {"conversation_history": {"n": ["hello"]}}

        history = _Node()._history_iokey(state).value_from_state(state)

        assert history == ["hello"]

    def test_a_node_outside_a_subagent_has_no_session(self):
        assert _Node()._session_id({}) is None
