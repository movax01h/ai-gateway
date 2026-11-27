# `compile_as_tool_approval_reviewer` is a method of `AgentComponent` (agent/component.py), not its own module, so the
# file-naming-for-tests 1-source-to-1-test heuristic doesn't apply here.
# pylint: disable=file-naming-for-tests
"""Tests for AgentComponent.compile_as_tool_approval_reviewer (the auto mode classifier's LLM reviewer)."""

from unittest.mock import Mock

import pytest
from langgraph.graph.state import CompiledStateGraph

from ai_gateway.response_schemas.registry import ResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    MAX_TOOL_APPROVAL_REVIEWER_CYCLES,
    AgentComponent,
    MaxCyclesConfig,
)


@pytest.fixture(name="make_reviewer")
def make_reviewer_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    def _make(tool_names, **overrides):
        mock_toolset.__iter__ = Mock(side_effect=lambda: iter(tool_names))
        mock_toolset.__contains__ = Mock(side_effect=lambda name: name in tool_names)
        kwargs = {
            "name": component_name,
            "flow_id": flow_id,
            "flow_type": flow_type,
            "user": user,
            "prompt_id": "tool_approval_review",
            "toolset": mock_toolset,
            "prompt_registry": mock_prompt_registry,
            "internal_event_client": mock_internal_event_client,
            "max_cycles": 5,
            "response_schema_id": "tool_approval_review",
            "response_schema_version": "^1.0.0",
            "schema_registry": ResponseSchemaRegistry(),
        }
        kwargs.update(overrides)
        # An override of None leaves the field out, so it takes the AgentComponent default.
        return AgentComponent(**{k: v for k, v in kwargs.items() if v is not None})

    return _make


class TestCompileAsToolApprovalReviewer:
    def test_compiles_without_a_checkpointer_or_description(self, make_reviewer):
        reviewer = make_reviewer(["read_file", "grep", "list_dir"])

        compiled = reviewer.compile_as_tool_approval_reviewer()

        assert isinstance(compiled, CompiledStateGraph)
        # False, not None: None would inherit the parent flow's checkpointer.
        assert compiled.checkpointer is False
        assert reviewer.__entry_hook__() in compiled.nodes
        assert "terminate_flow" in compiled.nodes

    def test_never_takes_an_unaddressed_inbox_message(self, make_reviewer):
        reviewer = make_reviewer(["read_file"])
        assert reviewer._claimant({}).accepts_unaddressed is True

        reviewer.compile_as_tool_approval_reviewer()

        # It runs inside the calling agent's turn, so a steering message sent meanwhile is that agent's.
        assert reviewer._claimant({}).accepts_unaddressed is False

    def test_compiles_without_tools(self, make_reviewer):
        assert make_reviewer([]).compile_as_tool_approval_reviewer() is not None

    @pytest.mark.parametrize(
        "tool_names",
        [
            ["read_file", "run_command"],
            ["edit_file"],
            ["create_file_with_contents", "grep"],
        ],
    )
    def test_rejects_a_tool_that_changes_or_runs_anything(
        self, make_reviewer, tool_names
    ):
        with pytest.raises(ValueError, match="may only hold read-only tools"):
            make_reviewer(tool_names).compile_as_tool_approval_reviewer()

    @pytest.mark.parametrize(
        "max_cycles",
        [
            1,
            MAX_TOOL_APPROVAL_REVIEWER_CYCLES,
            MaxCyclesConfig(threshold=MAX_TOOL_APPROVAL_REVIEWER_CYCLES),
        ],
    )
    def test_accepts_a_max_cycles_up_to_the_cap(self, make_reviewer, max_cycles):
        reviewer = make_reviewer(["read_file"], max_cycles=max_cycles)

        assert reviewer.compile_as_tool_approval_reviewer() is not None

    @pytest.mark.parametrize(
        "max_cycles",
        [
            # Omitted: the AgentComponent default, far above the cap.
            None,
            MAX_TOOL_APPROVAL_REVIEWER_CYCLES + 1,
            MaxCyclesConfig(threshold=MAX_TOOL_APPROVAL_REVIEWER_CYCLES + 1),
        ],
    )
    def test_rejects_a_max_cycles_above_the_cap(self, make_reviewer, max_cycles):
        reviewer = make_reviewer(["read_file"], max_cycles=max_cycles)

        with pytest.raises(ValueError, match="max_cycles must be at most"):
            reviewer.compile_as_tool_approval_reviewer()

    @pytest.mark.parametrize(
        "schema_overrides",
        [
            pytest.param(
                {"response_schema_id": None, "response_schema_version": None},
                id="text_answer",
            ),
            pytest.param(
                {
                    "response_schema_id": "secret_fp_detection",
                    "response_schema_version": "^1.0.0",
                },
                id="another_schema",
            ),
        ],
    )
    def test_rejects_any_answer_but_the_review_schema(
        self, make_reviewer, schema_overrides
    ):
        """The LLM backend reads the verdict from that schema's answer, so any other answer would never be one."""
        reviewer = make_reviewer(["read_file"], **schema_overrides)

        with pytest.raises(ValueError, match="response_schema_id must be"):
            reviewer.compile_as_tool_approval_reviewer()

    def test_compile_as_subagent_still_inherits_the_parent_checkpointer(
        self, make_reviewer
    ):
        subagent = make_reviewer(["read_file"], description="A subagent.")

        assert subagent.compile_as_subagent(address=Mock()).checkpointer is None
