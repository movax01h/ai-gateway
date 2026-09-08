# pylint: disable=file-naming-for-tests
"""Guards for the shipped workplan flow config.

These tests assert on the content of ``workplan/1.0.0.yml`` (pinned tool
options) rather than on ``FlowConfig`` machinery — generic ``FlowConfig``
behavior is covered in ``test_flow_config.py``.
"""

import json
from pathlib import Path

import pytest

from ai_gateway import response_schemas
from ai_gateway.prompts.base import jinja2_formatter
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.tools.work_item import CreateWorkItemNoteInput
from lib.version import resolve_version

# The flow's two AgentComponents. Shared by the classes below so a rename or a
# third stage can't leave one list behind.
AGENT_COMPONENT_NAMES = ["research", "planner"]


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


class TestWorkplanQuestionToolingIsAvailable:
    """Guard the tools both stages' prompts depend on to ask a question well.

    The prompts require two things of every question comment: that it isn't a
    duplicate of one already on the work item (checked by reading the notes,
    since a run can resume days later or restart with no memory), and that it
    @mentions someone who can answer it (looked up at runtime, not templated).
    Both are prompt-only behaviors - dropping either tool from a toolset
    wouldn't fail schema validation, it would just turn the instruction into a
    hallucinated no-op, so the coupling is pinned here.
    """

    REQUIRED_TOOLS = {"get_work_item_notes", "get_current_user"}

    @pytest.mark.parametrize("component_name", AGENT_COMPONENT_NAMES)
    def test_component_can_read_notes_and_identify_the_current_user(
        self, component_name
    ):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        component = next(
            c for c in config.components if c.get("name") == component_name
        )

        # Entries are either plain strings or single-key {tool: options} maps,
        # matching FlowGraphBuilder._parse_toolset.
        declared = {
            next(iter(entry)) if isinstance(entry, dict) else entry
            for entry in component["toolset"]
        }

        assert self.REQUIRED_TOOLS <= declared


class TestDuplicateQuestionStopDecision:
    """Guard the stage-conditional decision literal in the duplicate-question stop.

    ``asking_a_question`` is one partial rendered twice, and the keyword it
    tells the agent to stop with differs per stage. Nothing else fails if the
    two branches are swapped or a keyword is renamed: the agent would emit a
    value its response schema doesn't accept, the router would fall through to
    ``default_route``, and the flow would keep working - just always gated on a
    human. So both halves are pinned here, against the schema rather than a
    hardcoded string.
    """

    SCHEMA_DEFINITIONS = Path(response_schemas.__file__).parent / "definitions"

    EXPECTED_DECISION = {"research": "needs_input", "planner": "ask_question"}

    @staticmethod
    def _component(config: FlowConfig, component_name: str) -> dict:
        return next(c for c in config.components if c.get("name") == component_name)

    def _declared_decisions(self, component: dict) -> list[str]:
        """Read the ``decision`` enum straight out of the component's schema."""
        schema_dir = self.SCHEMA_DEFINITIONS / component["response_schema_id"] / "base"
        version = resolve_version(
            [f.stem for f in schema_dir.glob("*.json")],
            component["response_schema_version"],
        )
        schema = json.loads((schema_dir / f"{version}.json").read_text())

        return schema["properties"]["decision"]["enum"]

    @pytest.mark.parametrize("component_name", AGENT_COMPONENT_NAMES)
    def test_stop_names_this_stage_decision_and_the_schema_accepts_it(
        self, component_name
    ):
        config = FlowConfig.from_yaml_config("workplan", "1.0.0")
        component = self._component(config, component_name)
        expected = self.EXPECTED_DECISION[component_name]

        # The stage literal the flow actually passes, not the component name -
        # a config declaring the wrong one renders the other stage's branch.
        stage = next(
            inp["from"]
            for inp in component["inputs"]
            if inp.get("as") == "stage" and inp.get("literal")
        )
        assert config.prompts is not None
        prompt = next(
            p for p in config.prompts if p.prompt_id == component["prompt_id"]
        )
        rendered = jinja2_formatter(
            prompt.prompt_template["system"],
            stage=stage,
            goal="",
            research_findings="",
        )

        assert f"`decision: {expected}`" in rendered
        assert expected in self._declared_decisions(component)

        # And the other stage's keyword must not leak in: both branches of the
        # conditional rendering would look fine in isolation.
        (other,) = set(self.EXPECTED_DECISION.values()) - {expected}
        assert other not in rendered
