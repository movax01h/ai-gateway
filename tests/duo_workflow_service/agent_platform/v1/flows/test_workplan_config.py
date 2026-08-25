# pylint: disable=file-naming-for-tests
"""Guards for the shipped workplan flow config.

These tests assert on the content of ``workplan/1.0.0.yml`` (pinned tool
options) rather than on ``FlowConfig`` machinery — generic ``FlowConfig``
behavior is covered in ``test_flow_config.py``.
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.tools.work_item import CreateWorkItemNoteInput


class TestWorkplanToolOptions:
    """Guard the workplan flow's pinned create_work_item_note tool options.

    internal is pinned at the flow level (rather than left to the LLM) so
    every note either stage posts is structurally a plain, visible-to-humans
    comment, not an internal-only one. start_discussion isn't pinned because
    it isn't exposed on the tool at all (createDiscussion lacks the
    ai_workflows token scope; see !6551) - a human replying to the comment
    is what makes it a resolvable discussion instead.
    """

    EXPECTED_OPTIONS = {"internal": False}
    AGENT_COMPONENT_NAMES = ["research", "planner"]

    @staticmethod
    def _create_work_item_note_options(config: FlowConfig, component_name: str) -> dict:
        component = next(
            c for c in config.components if c.get("name") == component_name
        )
        for entry in component["toolset"]:
            if isinstance(entry, dict) and "create_work_item_note" in entry:
                return entry["create_work_item_note"]
        raise AssertionError(
            "create_work_item_note is not declared with pinned tool options "
            f"in {component_name}"
        )

    @pytest.mark.parametrize("component_name", AGENT_COMPONENT_NAMES)
    def test_create_work_item_note_args_are_pinned(self, component_name):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        assert (
            self._create_work_item_note_options(config, component_name)
            == self.EXPECTED_OPTIONS
        )

    @pytest.mark.parametrize("component_name", AGENT_COMPONENT_NAMES)
    def test_pinned_option_keys_are_valid_tool_parameters(self, component_name):
        # Mirrors Toolset._validate_tool_options: every pinned key must be a real
        # parameter on the tool's input schema, so a typo/rename fails fast here.
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        options = self._create_work_item_note_options(config, component_name)
        valid_fields = set(CreateWorkItemNoteInput.model_fields.keys())
        assert set(options).issubset(valid_fields)


class TestWorkplanRouterWiring:
    """Guard the workplan flow's router table.

    A typo'd route key (e.g. ``readyy`` instead of ``ready``) wouldn't be
    caught by schema validation - it would just silently fall through to
    default_route, degrading every research/planner turn to "always ask a
    human" with nothing in CI to catch it. These tests pin down the exact
    routing table instead of only spot-checking it by hand.
    """

    @staticmethod
    def _router_for(config: FlowConfig, from_component: str) -> dict:
        return next(r for r in config.routers if r["from"] == from_component)

    def test_research_router(self):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        router = self._router_for(config, "research")

        assert router["condition"]["input"] == "context:research.final_answer.decision"
        assert router["condition"]["routes"] == {
            "ready": "planner",
            "needs_input": "research_gate",
            "default_route": "research_gate",
        }

    def test_planner_router(self):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        router = self._router_for(config, "planner")

        assert router["condition"]["input"] == "context:planner.final_answer.decision"
        assert router["condition"]["routes"] == {
            "ask_question": "plan_gate",
            "plan_ready": "end",
            "default_route": "plan_gate",
        }

    @pytest.mark.parametrize(
        "gate_name,target",
        [("research_gate", "research"), ("plan_gate", "planner")],
    )
    def test_gate_resumes_into_its_own_agent(self, gate_name, target):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        router = self._router_for(config, gate_name)

        assert router["to"] == target

    def test_entry_point_is_research(self):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")

        assert config.flow.entry_point == "research"
