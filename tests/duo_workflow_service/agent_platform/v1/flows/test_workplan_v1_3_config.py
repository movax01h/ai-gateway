# pylint: disable=file-naming-for-tests
"""Guards for the workplan flow's 1.3.0 config: 1.2.0 plus child work items per track.

1.3.0 is a copy of 1.2.0 with these changes:

-   The planner answers with the inline ``workplan_decision_tracks`` schema,
    which adds ``tracks`` to ``workplan_decision`` 1.0.0.
-   The planner prompt has a ``## Tracks`` section (which also tells the
    planner not to create the child work items itself) and one more
    announcement bullet, and a longer call timeout.
-   A ``publish_tracks`` deterministic step runs ``workplan_publish_tracks``
    between the planner and the scoring tail.
-   The planner has the read-only ``gitlab_graphql`` and never has
    ``create_work_item``: a child that the planner creates has no track marker,
    so ``publish_tracks`` would create that track again. The comparison with
    1.2.0 leaves out both tools, so it holds whether or not the 1.2.0 planner
    has them.

Everything else must match 1.2.0 exactly, so the versions stay comparable.
"""

import copy

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW_NAME = "workplan"
FLOW_VERSION = "1.3.0"
PREVIOUS_VERSION = "1.2.0"

PLANNER_PROMPT_ID = "workplan_planner_prompt"
TRACKS_SCHEMA_ID = "workplan_decision_tracks"

PLANNER_READ_TOOL = "gitlab_graphql"
CREATE_TOOL = "create_work_item"


def _config(version: str = FLOW_VERSION) -> FlowConfig:
    return FlowConfig.from_yaml_config(FLOW_NAME, version)


def _component(config: FlowConfig, name: str) -> dict:
    return next(c for c in config.components if c["name"] == name)


def _entry_names(entry) -> list[str]:
    # A toolset entry is a name, or a mapping of names to pinned args.
    return [entry] if isinstance(entry, str) else list(entry)


def _tool_names(component: dict) -> list[str]:
    return [
        name for entry in component.get("toolset") or [] for name in _entry_names(entry)
    ]


def _without_tools(component: dict, names: set[str]) -> dict:
    # Strips by name, so a mapping keeps the other tools that it pins.
    component = copy.deepcopy(component)
    kept: list = []
    for entry in component["toolset"]:
        if isinstance(entry, str):
            if entry not in names:
                kept.append(entry)
        elif rest := {k: v for k, v in entry.items() if k not in names}:
            kept.append(rest)
    component["toolset"] = kept
    return component


def _prompt(config: FlowConfig, prompt_id: str):
    return next(p for p in config.prompts or [] if p.prompt_id == prompt_id)


def _system(config: FlowConfig, prompt_id: str) -> str:
    system = _prompt(config, prompt_id).prompt_template["system"]
    assert isinstance(system, str)
    return system


def _tracks_section(system: str) -> str:
    start = system.index("## Tracks\n")
    end = system.index("## Notifying the author")
    return system[start:end]


ANNOUNCEMENT_BULLET = (
    "- When you return two or more tracks, name them in one line, and say\n"
    "  that each one becomes a child work item when the tracks pass the\n"
    "  checks. When this work item already has child items from an earlier\n"
    "  plan, say that they keep their Workplan.\n"
)


class TestWorkplanV130PublishTracks:
    def test_publish_tracks_runs_the_deterministic_tool(self):
        component = _component(_config(), "publish_tracks")

        assert component["type"] == "DeterministicStepComponent"
        assert component["tool_name"] == "workplan_publish_tracks"
        assert component["inputs"] == [
            {
                "from": "context:inputs.agent_platform_resource_context.work_item_web_url",
                "as": "url",
                "optional": True,
            },
            {
                "from": "context:planner.final_answer.tracks",
                "as": "tracks",
                "optional": True,
            },
        ]

    def test_plan_ready_goes_through_publish_tracks_to_scoring(self):
        routers = _config().routers
        planner = next(r for r in routers if r["from"] == "planner")
        publish = next(r for r in routers if r["from"] == "publish_tracks")

        assert planner["condition"]["routes"]["plan_ready"] == "publish_tracks"
        assert publish == {"from": "publish_tracks", "to": "score_fetch"}

    def test_no_component_but_publish_tracks_can_create_work_items(self):
        for component in _config().components:
            names = _tool_names(component)
            assert CREATE_TOOL not in names, component["name"]
            assert "workplan_publish_tracks" not in names, component["name"]

    def test_only_the_planner_has_the_graphql_read_tool(self):
        for component in _config().components:
            has_tool = PLANNER_READ_TOOL in _tool_names(component)
            assert has_tool == (component["name"] == "planner"), component["name"]

    def test_planner_uses_the_inline_tracks_schema(self):
        planner = _component(_config(), "planner")

        assert planner["response_schema_id"] == TRACKS_SCHEMA_ID
        assert "response_schema_version" not in planner
        assert [s.schema_id for s in _config().response_schemas or []] == [
            TRACKS_SCHEMA_ID
        ]

    def test_tracks_schema_extends_workplan_decision(self):
        schema = (_config().response_schemas or [])[0].to_schema_dict()
        properties = schema["properties"]

        assert properties["decision"]["enum"] == [
            "ask_question",
            "needs_context",
            "plan_ready",
        ]
        assert schema["required"] == ["decision"]
        # The tool checks the count and the keys and refuses with a note. A
        # schema rejection would fail the session after the plan is posted.
        assert "maxItems" not in properties["tracks"]
        assert "pattern" not in properties["tracks"]["items"]["properties"]["key"]
        assert properties["tracks"]["items"]["required"] == [
            "key",
            "title",
            "files",
            "depends_on",
            "workplan",
        ]

    def test_planner_prompt_states_the_rules_the_tool_checks(self):
        section = _tracks_section(_system(_config(), PLANNER_PROMPT_ID))

        assert "Two tracks that change the same file are one track" in section
        assert "Every track names its files." in section
        assert "At most 10 tracks." in section
        assert "word for word" in section
        # The planner returns no tracks for any other type: the tool skips
        # that type, and its note would contradict the announcement.
        assert "Return an empty list unless this work item is an Issue" in " ".join(
            section.split()
        )

    def test_planner_prompt_leaves_child_creation_to_publish_tracks(self):
        section = _tracks_section(_system(_config(), PLANNER_PROMPT_ID))

        assert "Do not create the child work items yourself" in " ".join(
            section.split()
        )


class TestWorkplanV130MatchesV120OutsideChanges:
    @staticmethod
    def _v120() -> FlowConfig:
        return _config(PREVIOUS_VERSION)

    def test_components_match_except_planner_and_publish_tracks(self):
        v120 = {c["name"]: c for c in self._v120().components}
        names = [c["name"] for c in _config().components]

        assert [n for n in names if n != "publish_tracks"] == list(v120)
        for component in _config().components:
            if component["name"] in ("planner", "publish_tracks"):
                continue
            assert component == v120[component["name"]], component["name"]

    def test_planner_differs_only_by_the_schema_and_its_tool_rule(self):
        planner = _without_tools(
            _component(_config(), "planner"), {PLANNER_READ_TOOL, CREATE_TOOL}
        )
        planner["response_schema_id"] = "workplan_decision"
        planner["response_schema_version"] = "^1.0.0"

        assert planner == _without_tools(
            _component(self._v120(), "planner"), {PLANNER_READ_TOOL, CREATE_TOOL}
        )

    def test_routers_differ_only_by_publish_tracks(self):
        routers = copy.deepcopy(_config().routers)
        routers = [r for r in routers if r["from"] != "publish_tracks"]
        planner = next(r for r in routers if r["from"] == "planner")
        planner["condition"]["routes"]["plan_ready"] = "score_fetch"

        assert routers == self._v120().routers

    def test_flow_section_and_include_match(self):
        assert _config().flow == self._v120().flow
        assert _config().include == self._v120().include

    def test_unchanged_prompts_match(self):
        v120 = {p.prompt_id: p for p in self._v120().prompts or []}

        assert [p.prompt_id for p in _config().prompts or []] == list(v120)
        for prompt in _config().prompts or []:
            if prompt.prompt_id == PLANNER_PROMPT_ID:
                continue
            assert prompt == v120[prompt.prompt_id], prompt.prompt_id

    def test_planner_prompt_differs_only_by_the_additions(self):
        system = _system(_config(), PLANNER_PROMPT_ID)
        without = system.replace(_tracks_section(system), "").replace(
            ANNOUNCEMENT_BULLET, ""
        )

        assert without == _system(self._v120(), PLANNER_PROMPT_ID)

        v130 = _prompt(_config(), PLANNER_PROMPT_ID)
        v120 = _prompt(self._v120(), PLANNER_PROMPT_ID)
        assert v130.params.timeout == 300
        assert v130.model_dump(
            exclude={"prompt_template", "params"}
        ) == v120.model_dump(exclude={"prompt_template", "params"})
        assert {k: v for k, v in v130.prompt_template.items() if k != "system"} == {
            k: v for k, v in v120.prompt_template.items() if k != "system"
        }
