# pylint: disable=file-naming-for-tests
"""Guards for the workplan flow's 1.2.0 config: 1.1.0 plus workspace agents and skills.

1.2.0 is a copy of 1.1.0 with two additions:

-   Workspace agent templates, bound through an ``include`` block
    (``source: workspace``, ``item_type: agent_template``, ``item_id: "*"``).
    Only ``research`` claims them as subagents, and only its prompt carries the
    ``has_workspace_agents`` delegation guidance. ``planner`` never delegates.
-   Workspace agent skills. ``research`` and ``planner`` each take an optional
    ``workspace_agent_skills`` input, render the shared skills partial, and get
    ``read_files`` so they can open skill bodies at the manifest's absolute paths.

These invariants matter enough to pin in CI. The generic sweep in
``test_configs.py`` only proves the config binds and compiles with synthetic
items. It would not notice ``read_files`` or the skills input going missing, or
the claim moving from ``research`` to ``planner``. Everything outside the two
additions must match 1.1.0 exactly, so the versions stay comparable.
"""

import copy
from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.v1.catalog import (
    CatalogItemRef,
    CatalogItems,
    WorkspaceAgent,
    bind_catalog_items,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.components.tools_registry import ToolsRegistry

FLOW_NAME = "workplan"
FLOW_VERSION = "1.2.0"
PREVIOUS_VERSION = "1.1.0"

# The components 1.2.0 changes; every other component matches 1.1.0.
SKILL_COMPONENTS = ["research", "planner"]
SKILL_PROMPT_IDS = ["workplan_research_prompt", "workplan_planner_prompt"]

WORKSPACE_AGENTS_REF = {
    "source": "workspace",
    "item_type": "agent_template",
    "item_id": "*",
}

SKILLS_INPUT = {
    "from": "context:inputs.workspace_agent_skills",
    "as": "workspace_agent_skills",
    "optional": True,
}

SKILLS_PARTIAL = "{% include 'workspace_agent_skills/base/1.0.0.jinja' %}"

# The operating constraint each skill-aware prompt rewrites to allow read_files.
# Built line by line: the prompts indent continuation lines by two spaces.
OLD_FILES_CONSTRAINT = (
    "- You have GitLab API tools only - no working tree exists for this flow,\n"
    "  so nothing here can read local files, run commands, or execute git.\n"
    "  Repository research (below) goes through the GitLab API instead.\n"
)
NEW_FILES_CONSTRAINT = (
    "- Repository research goes through the GitLab API - there is no\n"
    "  working tree here, and you cannot run commands or execute git. The\n"
    "  one exception is `read_files`, which opens agent skill files at the\n"
    "  absolute paths listed under <workspace_agent_skills>. Do not point\n"
    "  it at repository paths.\n"
)

DELEGATION_GUIDANCE = (
    "{%- if has_workspace_agents %}\n"
    "- Workspace agents are available to you as subagents. Delegate a\n"
    "  self-contained research question to one with `delegate_task`, as the\n"
    "  only tool call in that turn, and fold its answer into `findings`.\n"
    "  Delegation spends the same tool-call budget as any other call.\n"
    "{%- endif %}\n"
)


def _config(version: str = FLOW_VERSION) -> FlowConfig:
    return FlowConfig.from_yaml_config(FLOW_NAME, version)


def _component(config: FlowConfig, name: str) -> dict:
    return next(c for c in config.components if c["name"] == name)


def _router_for(config: FlowConfig, from_component: str) -> dict:
    return next(r for r in config.routers if r["from"] == from_component)


def _prompt(config: FlowConfig, prompt_id: str):
    return next(p for p in config.prompts or [] if p.prompt_id == prompt_id)


def _system(config: FlowConfig, prompt_id: str) -> str:
    system = _prompt(config, prompt_id).prompt_template["system"]
    assert isinstance(system, str)
    return system


def _without_skill_additions(component: dict) -> dict:
    """Undo 1.2.0's component additions, leaving what 1.1.0 declares."""
    stripped = copy.deepcopy(component)
    stripped["inputs"] = [i for i in stripped["inputs"] if i != SKILLS_INPUT]
    stripped["toolset"] = [t for t in stripped["toolset"] if t != "read_files"]
    stripped.pop("subagents", None)
    return stripped


def _without_skill_prompt_additions(system: str) -> str:
    """Undo 1.2.0's system prompt additions, leaving 1.1.0's text."""
    return (
        system.replace(NEW_FILES_CONSTRAINT, OLD_FILES_CONSTRAINT)
        .replace(DELEGATION_GUIDANCE, "")
        .replace(f"\n\n{SKILLS_PARTIAL}", "")
    )


class TestWorkplanV120WorkspaceAgents:
    """Only research claims workspace agent templates."""

    def test_include_declares_exactly_the_workspace_agent_templates(self):
        assert _config().include == [CatalogItemRef(**WORKSPACE_AGENTS_REF)]

    def test_research_claims_the_include_entry(self):
        assert _component(_config(), "research")["subagents"] == [WORKSPACE_AGENTS_REF]

    def test_research_is_the_only_component_with_subagents_besides_the_supervisor(
        self,
    ):
        # readiness_supervisor keeps its static evaluator subagents from 1.1.0.
        with_subagents = {c["name"] for c in _config().components if c.get("subagents")}
        assert with_subagents == {"research", "readiness_supervisor"}

    def test_supervisor_does_not_claim_workspace_agents(self):
        subagents = _component(_config(), "readiness_supervisor")["subagents"]
        assert WORKSPACE_AGENTS_REF not in subagents

    @pytest.mark.parametrize(
        "prompt_id,has_guidance",
        [
            ("workplan_research_prompt", True),
            ("workplan_planner_prompt", False),
        ],
    )
    def test_only_research_prompt_reads_has_workspace_agents(
        self, prompt_id, has_guidance
    ):
        system = _system(_config(), prompt_id)
        assert ("has_workspace_agents" in system) is has_guidance
        assert (DELEGATION_GUIDANCE in system) is has_guidance

    def test_no_other_prompt_reads_has_workspace_agents(self):
        for prompt in _config().prompts or []:
            if prompt.prompt_id == "workplan_research_prompt":
                continue
            for template in prompt.prompt_template.values():
                assert "has_workspace_agents" not in str(template), prompt.prompt_id


class TestWorkplanV120BindsWorkspaceAgents:
    """Items bind to research, and only to research."""

    @staticmethod
    def _bind(agents: list[WorkspaceAgent]) -> dict[str, dict]:
        config = _config()
        bound = bind_catalog_items(
            config.components,
            config.include,
            CatalogItems(workspace_agents=agents),
            Mock(spec=ToolsRegistry),
        )
        return {c["name"]: c for c in bound}

    @staticmethod
    def _has_agents_flag(component: dict) -> list[dict]:
        return [i for i in component["inputs"] if i["as"] == "has_workspace_agents"]

    @pytest.mark.usefixtures("workspace_agents_flag")
    def test_items_become_subagents_of_research(self):
        bound = self._bind(
            [
                WorkspaceAgent(
                    name="tester", description="Runs tests.", prompt="Be terse."
                ),
                WorkspaceAgent(
                    name="reviewer", description="Reviews changes.", prompt="Be picky."
                ),
            ]
        )

        authored = {c["name"] for c in _config().components}
        synthesized = [name for name in bound if name not in authored]

        assert len(synthesized) == 2
        assert bound["research"]["subagents"] == [
            {"name": name} for name in synthesized
        ]
        assert [f["from"] for f in self._has_agents_flag(bound["research"])] == ["true"]

    @pytest.mark.usefixtures("workspace_agents_flag")
    def test_planner_never_gets_subagents(self):
        bound = self._bind(
            [WorkspaceAgent(name="tester", description="Runs tests.", prompt="x")]
        )
        assert "subagents" not in bound["planner"]
        assert not self._has_agents_flag(bound["planner"])

    @pytest.mark.usefixtures("workspace_agents_flag")
    def test_research_stays_a_plain_agent_without_items(self):
        bound = self._bind([])

        assert not bound["research"].get("subagents")
        assert [f["from"] for f in self._has_agents_flag(bound["research"])] == [""]
        assert bound.keys() == {c["name"] for c in _config().components}


class TestWorkplanV120WorkspaceAgentSkills:
    """Research and planner read the skills manifest and can open skill bodies."""

    @pytest.mark.parametrize("component_name", SKILL_COMPONENTS)
    def test_skills_input_is_optional(self, component_name):
        inputs = _component(_config(), component_name)["inputs"]
        assert SKILLS_INPUT in inputs

    @pytest.mark.parametrize("component_name", SKILL_COMPONENTS)
    def test_toolset_includes_read_files(self, component_name):
        assert "read_files" in _component(_config(), component_name)["toolset"]

    def test_only_skill_components_get_read_files_or_the_skills_input(self):
        for component in _config().components:
            if component["name"] in SKILL_COMPONENTS:
                continue
            assert "read_files" not in (component.get("toolset") or [])
            assert SKILLS_INPUT not in (component.get("inputs") or [])

    @pytest.mark.parametrize("prompt_id", SKILL_PROMPT_IDS)
    def test_prompt_renders_the_skills_partial(self, prompt_id):
        assert SKILLS_PARTIAL in _system(_config(), prompt_id)

    @pytest.mark.parametrize("prompt_id", SKILL_PROMPT_IDS)
    def test_prompt_confines_read_files_to_skill_paths(self, prompt_id):
        system = _system(_config(), prompt_id)
        assert NEW_FILES_CONSTRAINT in system
        assert OLD_FILES_CONSTRAINT not in system


class TestWorkplanV120MatchesV110OutsideAdditions:
    """Outside workspace agents and skills, 1.2.0 must match 1.1.0."""

    @staticmethod
    def _v110() -> FlowConfig:
        return _config(PREVIOUS_VERSION)

    def test_component_names_and_order_match(self):
        assert [c["name"] for c in _config().components] == [
            c["name"] for c in self._v110().components
        ]

    def test_unchanged_components_match(self):
        v110 = {c["name"]: c for c in self._v110().components}
        for component in _config().components:
            if component["name"] in SKILL_COMPONENTS:
                continue
            assert component == v110[component["name"]], component["name"]

    @pytest.mark.parametrize("component_name", SKILL_COMPONENTS)
    def test_skill_components_differ_only_by_the_additions(self, component_name):
        assert _without_skill_additions(
            _component(_config(), component_name)
        ) == _component(self._v110(), component_name)

    def test_routers_match(self):
        assert _config().routers == self._v110().routers

    def test_flow_section_matches(self):
        assert _config().flow == self._v110().flow

    def test_remaining_top_level_fields_match(self):
        changed = {"components", "prompts", "include", "resolved_version"}
        assert _config().model_dump(exclude=changed) == self._v110().model_dump(
            exclude=changed
        )

    def test_v110_has_no_include(self):
        assert self._v110().include is None

    def test_response_schemas_match(self):
        assert _config().response_schemas == self._v110().response_schemas

    def test_prompt_ids_and_order_match(self):
        assert [p.prompt_id for p in _config().prompts or []] == [
            p.prompt_id for p in self._v110().prompts or []
        ]

    def test_unchanged_prompts_match(self):
        v110 = {p.prompt_id: p for p in self._v110().prompts or []}
        for prompt in _config().prompts or []:
            if prompt.prompt_id in SKILL_PROMPT_IDS:
                continue
            assert prompt == v110[prompt.prompt_id], prompt.prompt_id

    @pytest.mark.parametrize("prompt_id", SKILL_PROMPT_IDS)
    def test_skill_prompts_differ_only_by_the_additions(self, prompt_id):
        v120 = _prompt(_config(), prompt_id)
        v110 = _prompt(self._v110(), prompt_id)

        assert _without_skill_prompt_additions(_system(_config(), prompt_id)) == (
            _system(self._v110(), prompt_id)
        )
        # Everything but the system prompt is untouched.
        assert v120.model_dump(exclude={"prompt_template"}) == v110.model_dump(
            exclude={"prompt_template"}
        )
        assert {k: v for k, v in v120.prompt_template.items() if k != "system"} == {
            k: v for k, v in v110.prompt_template.items() if k != "system"
        }
