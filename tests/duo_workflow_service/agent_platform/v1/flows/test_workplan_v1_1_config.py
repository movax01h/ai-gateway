# pylint: disable=file-naming-for-tests
"""Guards for the workplan flow's 1.1.0 config: planning plus the readiness-score tail.

1.1.0 keeps 1.0.0's five planning components unchanged and merges the standalone
``readiness_score`` flow in behind the planner: ``plan_ready`` now routes to a
scoring tail (``score_fetch`` -> ``score_fetch_notes`` ->
``readiness_supervisor``) instead of ``end``. The supervisor is declared as a
plain ``AgentComponent`` with a ``subagents`` list so the factory
(``dap_parallel_subagents`` flag) promotes it to ``SupervisorAgentComponentV2``
and runs the two evaluator rounds in parallel.

These invariants matter enough to pin in CI:

-   Scoring failures must route to ``end``, never ``abort``. Unlike the
    standalone readiness flow, the plan has already been written and announced
    by the time scoring starts, so a failed fetch must not mark a successful
    planning run failed.
-   The supervisor's ``update_work_item`` is unpinned and passes only
    ``readiness_score`` and ``readiness_score_feedback``. The scoring stage did
    not author the plan.
-   Everything outside the scoring tail matches 1.0.0 exactly, so the two
    versions differ only by scoring and can be compared against each other.
-   The scoring tail reads the work item URL from the
    ``agent_platform_resource_context`` envelope, never ``context:goal``,
    whose prose ``GitLabUrlParser`` rejects.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ai_gateway.prompts import InMemoryPromptRegistry
from ai_gateway.prompts.base import Prompt
from ai_gateway.response_schemas.inline_registry import InlineResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import (
    FlowGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.state import RuntimeIOKey
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.work_item import GetWorkItemNotesInput
from lib.events import GLReportingEventContext
from lib.feature_flags import current_feature_flag_context
from lib.feature_flags.context import FeatureFlag

FLOW_NAME = "workplan"
FLOW_VERSION = "1.1.0"

PLANNING_COMPONENTS = {
    "research",
    "research_gate",
    "planner",
    "plan_gate",
    "context_gate",
}

SCORING_COMPONENTS = {
    "score_fetch",
    "score_fetch_notes",
    "rubric",
    "coverage",
    "falsifier",
    "xexam_rubric",
    "xexam_coverage",
    "xexam_falsifier",
    "readiness_supervisor",
}

EXPECTED_COMPONENTS = PLANNING_COMPONENTS | SCORING_COMPONENTS

SCORING_PROMPT_IDS = {
    "rs_rubric_prompt",
    "rs_coverage_prompt",
    "rs_falsifier_prompt",
    "rs_xexam_scorer_prompt",
    "rs_xexam_falsifier_prompt",
    "workplan_readiness_supervisor_prompt",
}

WORK_ITEM_URL_INPUT = "context:inputs.agent_platform_resource_context.work_item_web_url"

ROUND_ONE_EVALUATORS = ["rubric", "coverage", "falsifier"]
CROSS_EXAMINERS = ["xexam_rubric", "xexam_coverage", "xexam_falsifier"]


def _config() -> FlowConfig:
    return FlowConfig.from_yaml_config(FLOW_NAME, FLOW_VERSION)


def _component(config: FlowConfig, name: str) -> dict:
    return next(c for c in config.components if c["name"] == name)


def _router_for(config: FlowConfig, from_component: str) -> dict:
    return next(r for r in config.routers if r["from"] == from_component)


class TestWorkplanV110ComponentSet:
    """The new version adds exactly the nine scoring-tail components."""

    def test_component_set_is_planning_five_plus_scoring_nine(self):
        config = _config()
        assert {c["name"] for c in config.components} == EXPECTED_COMPONENTS

    def test_config_loads_with_ambient_environment(self):
        config = _config()
        assert config.flow.entry_point == "research"
        assert config.environment == "ambient"

    def test_planner_router_routes_plan_ready_into_the_scoring_tail(self):
        config = _config()
        routes = _router_for(config, "planner")["condition"]["routes"]
        assert routes["plan_ready"] == "score_fetch"
        assert routes["ask_question"] == "plan_gate"
        assert routes["needs_context"] == "context_gate"
        assert routes["default_route"] == "plan_gate"

    @pytest.mark.parametrize(
        "component_name,tool_name,expected_inputs",
        [
            ("score_fetch", "get_work_item", {"url": WORK_ITEM_URL_INPUT}),
            (
                "score_fetch_notes",
                "get_work_item_notes",
                {"url": WORK_ITEM_URL_INPUT, "page_size": "100"},
            ),
        ],
    )
    def test_score_fetch_steps_are_deterministic(
        self, component_name, tool_name, expected_inputs
    ):
        component = _component(_config(), component_name)
        assert component["type"] == "DeterministicStepComponent"
        assert component["tool_name"] == tool_name
        inputs = {i["as"]: i["from"] for i in component["inputs"]}
        assert inputs == expected_inputs

    def test_notes_page_size_is_the_tools_maximum(self):
        component = _component(_config(), "score_fetch_notes")
        page_size = next(i for i in component["inputs"] if i["as"] == "page_size")
        assert page_size["literal"] is True
        # Literals are strings; the schema coerces it to the max page.
        assert GetWorkItemNotesInput(page_size=page_size["from"]).page_size == 100
        with pytest.raises(ValidationError):
            GetWorkItemNotesInput(page_size=101)


class TestWorkplanV110MatchesV100OutsideScoring:
    """Outside the scoring tail, 1.1.0 must match 1.0.0."""

    @staticmethod
    def _v100() -> FlowConfig:
        return FlowConfig.from_yaml_config(FLOW_NAME, "1.0.0")

    @pytest.mark.parametrize("component_name", sorted(PLANNING_COMPONENTS))
    def test_planning_components_match(self, component_name):
        assert _component(_config(), component_name) == _component(
            self._v100(), component_name
        )

    def test_v100_has_exactly_the_planning_components(self):
        assert {c["name"] for c in self._v100().components} == PLANNING_COMPONENTS

    @pytest.mark.parametrize(
        "from_component", sorted(PLANNING_COMPONENTS - {"planner"})
    )
    def test_planning_routers_match(self, from_component):
        assert _router_for(_config(), from_component) == _router_for(
            self._v100(), from_component
        )

    def test_planner_router_differs_only_by_plan_ready(self):
        v110 = _router_for(_config(), "planner")["condition"]
        v100 = _router_for(self._v100(), "planner")["condition"]
        assert v110["input"] == v100["input"]
        assert v100["routes"]["plan_ready"] == "end"
        assert {**v110["routes"], "plan_ready": "end"} == v100["routes"]

    def test_planning_prompts_match(self):
        v110 = {
            p.prompt_id: p
            for p in _config().prompts or []
            if p.prompt_id not in SCORING_PROMPT_IDS
        }
        v100 = {p.prompt_id: p for p in self._v100().prompts or []}
        assert v110 == v100

    def test_entry_point_matches(self):
        assert _config().flow.entry_point == self._v100().flow.entry_point


class TestWorkplanV110WorkItemUrl:
    """The scoring tail addresses the work item by a bare, parseable URL."""

    # Rails' goal template for this flow.
    GOAL = (
        "Generate a workplan for the work item at "
        "http://gdk.test:3000/gitlab-duo/test/-/work_items/17."
    )
    URL = "http://gdk.test:3000/gitlab-duo/test/-/work_items/17"
    HOST = "gdk.test:3000"

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(GOAL, id="goal-sentence"),
            pytest.param(f"{URL}.", id="url-with-trailing-period"),
        ],
    )
    def test_unparseable_url_shapes_the_flow_must_not_bind(self, value):
        """Why the fetches cannot be bound to ``context:goal``."""
        with pytest.raises(GitLabUrlParseError):
            GitLabUrlParser.parse_work_item_url(value, self.HOST)

    def test_bare_url_parses(self):
        parsed = GitLabUrlParser.parse_work_item_url(self.URL, self.HOST)
        assert parsed.full_path == "gitlab-duo/test"
        assert parsed.work_item_iid == 17

    @pytest.mark.parametrize(
        "component_name",
        ["score_fetch", "score_fetch_notes", "readiness_supervisor"],
    )
    def test_scoring_tail_never_binds_the_raw_goal(self, component_name):
        component = _component(_config(), component_name)
        assert "context:goal" not in {i["from"] for i in component["inputs"]}

    @pytest.mark.parametrize(
        "component_name",
        ["score_fetch", "score_fetch_notes", "readiness_supervisor"],
    )
    def test_the_url_input_is_optional(self, component_name):
        """A session with no work item resource must end the run, not raise."""
        component = _component(_config(), component_name)
        url_input = next(
            i for i in component["inputs"] if i["from"] == WORK_ITEM_URL_INPUT
        )
        assert url_input["optional"] is True

    def test_flow_declares_the_resource_context_category(self):
        """The envelope is dropped at ingestion unless the flow declares it."""
        config = _config()
        categories = {i.category: i for i in config.flow.inputs}
        assert "agent_platform_resource_context" in categories

        declared = categories["agent_platform_resource_context"]
        assert "work_item_web_url" in declared.input_schema
        assert declared.input_schema["work_item_web_url"].optional is True
        assert (
            config.input_json_schemas_by_category()["agent_platform_resource_context"][
                "required"
            ]
            == []
        )
        assert declared.version_constraint == "^1.0.0"


class TestWorkplanV110ScoringFailuresEndTheRun:
    """A scoring failure must end the run, not abort it.

    The standalone readiness_score flow routes fetch/persist failures to
    ``abort``, which is correct when scoring is the whole run. Here the plan is
    already written and announced before scoring starts, so aborting would mark
    a successful planning run failed.
    """

    @pytest.mark.parametrize(
        "from_component,expected_input,success_target",
        [
            (
                "score_fetch",
                "context:score_fetch.execution_result",
                "score_fetch_notes",
            ),
            (
                "score_fetch_notes",
                "context:score_fetch_notes.execution_result",
                "readiness_supervisor",
            ),
        ],
    )
    def test_fetch_failure_routes_to_end_not_abort(
        self, from_component, expected_input, success_target
    ):
        config = _config()
        router = _router_for(config, from_component)

        assert "to" not in router, (
            f"{from_component} must route on its execution result, not unconditionally"
        )
        assert router["condition"]["input"] == expected_input
        assert router["condition"]["routes"]["success"] == success_target
        assert router["condition"]["routes"]["default_route"] == "end"

    def test_supervisor_routes_to_end(self):
        assert _router_for(_config(), "readiness_supervisor")["to"] == "end"

    def test_no_router_ever_targets_abort(self):
        config = _config()
        for router in config.routers:
            assert router.get("to") != "abort"
            for route in (router.get("condition") or {}).get("routes", {}).values():
                assert route != "abort"


class TestWorkplanV110SupervisorWiring:
    """The supervisor delegates to the six evaluators and writes the score and feedback."""

    def test_supervisor_declares_all_six_subagents(self):
        supervisor = _component(_config(), "readiness_supervisor")
        subagent_names = [entry["name"] for entry in supervisor["subagents"]]
        assert subagent_names == ROUND_ONE_EVALUATORS + CROSS_EXAMINERS

    def test_supervisor_caps_delegations_at_one_per_subagent(self):
        # Three evaluators plus three cross-examiners is exactly one delegation
        # per subagent; a higher cap would only ever let the supervisor re-run a
        # round instead of writing the score.
        supervisor = _component(_config(), "readiness_supervisor")
        assert supervisor["max_delegations"] == 6

    def test_supervisor_toolset_is_only_an_unpinned_update_work_item(self):
        supervisor = _component(_config(), "readiness_supervisor")
        assert supervisor["toolset"] == ["update_work_item"]

    def test_supervisor_binds_the_work_item_url_for_the_update_call(self):
        supervisor = _component(_config(), "readiness_supervisor")
        inputs = {i["as"]: i["from"] for i in supervisor["inputs"]}
        assert inputs == {"work_item_url": WORK_ITEM_URL_INPUT}

    def test_supervisor_prompt_prohibits_agent_plan_and_partial_ensembles(self):
        config = _config()
        prompt = next(
            p
            for p in config.prompts
            if p.prompt_id == "workplan_readiness_supervisor_prompt"
        )
        system = prompt.prompt_template["system"]
        # The agent_plan prohibition must be stated forcefully in the prompt.
        assert "agent_plan" in system
        # The partial-ensemble guard: no score may be written unless all three
        # round-1 evaluators returned.
        assert "STOP" in system

    def test_supervisor_prompt_persists_feedback_alongside_the_score(self):
        # The supervisor writes the 3 recommendations it already produces for
        # its final answer into readiness_score_feedback, so the widget shows
        # *why* the plan scored what it did — not just the number.
        config = _config()
        prompt = next(
            p
            for p in config.prompts
            if p.prompt_id == "workplan_readiness_supervisor_prompt"
        )
        system = prompt.prompt_template["system"]
        assert "readiness_score_feedback" in system
        # The Turn 3 call must pass both arguments, not the score alone.
        assert "`readiness_score` and `readiness_score_feedback`" in system
        # And the feedback must be a bare markdown list, with no heading.
        assert "markdown list with no heading" in system
        assert "## Recommendation" not in system


class TestWorkplanV110GraphBuilds:
    """The config must compile into a real graph, subagent wiring included.

    Parsing alone leaves the supervisor half-verified: a subagent missing its
    ``description`` (``compile_as_subagent`` raises), an unknown tool in the
    supervisor's toolset, or a subagent name absent from ``subagents`` all pass
    ``FlowConfig.from_yaml_config`` and only fail at graph-build time — which,
    in production, is flow start.
    """

    # The six subagents as real graph nodes, plus the supervisor's own nodes.
    EXPECTED_SUPERVISOR_NODES = {
        "readiness_supervisor#agent",
        "readiness_supervisor#tools",
        "readiness_supervisor#final_response",
        "readiness_supervisor#delegation_prepare",
        "readiness_supervisor#delegation_collect",
    }

    @pytest.fixture(name="feature_flag_context")
    def feature_flag_context_fixture(self):
        token = current_feature_flag_context.set(
            {FeatureFlag.DAP_PARALLEL_SUBAGENTS.value}
        )
        yield
        current_feature_flag_context.reset(token)

    @pytest.fixture(name="mock_container")
    def mock_container_fixture(self):
        """Real per-run dependencies for everything the graph build touches."""
        from ai_gateway.config import Config  # pylint: disable=import-outside-toplevel
        from ai_gateway.container import ContainerApplication

        container = ContainerApplication()
        container.config.from_dict(
            Config(_env_file=None, _env_prefix="AIGW_TEST").model_dump()
        )
        return container

    @pytest.fixture(name="builder")
    def builder_fixture(self, mock_container, user, tool_metadata):
        # Flow-defined prompts are in-memory (registered from the config
        # itself), so the builder needs the same wrapping Flow uses:
        # InMemoryPromptRegistry over the local registry, with this config's
        # prompts registered into it. Prompt building is stubbed out: it needs
        # model metadata (an LLM definition) that only a real request context
        # carries, and this test asserts graph structure, not prompt content.
        config = _config()
        flow_prompt_registry = InMemoryPromptRegistry(
            mock_container.pkg_prompts.prompt_registry()
        )
        for prompt_config in config.prompts or []:
            flow_prompt_registry.register_prompt(
                prompt_id=prompt_config.prompt_id,
                prompt_data=prompt_config.to_prompt_data(),
            )
        flow_prompt_registry.get_on_behalf = MagicMock(return_value=Mock(spec=Prompt))
        return FlowGraphBuilder(
            tools_registry=ToolsRegistry(
                enabled_tools=["read_write_gitlab"],
                preapproved_tools=[],
                tool_metadata=tool_metadata,
            ),
            prompt_registry=flow_prompt_registry,
            schema_registry=InlineResponseSchemaRegistry(
                mock_container.pkg_schemas.schema_registry()
            ),
            workflow_id="test-workflow-workplan-1-1-0",
            workflow_type=GLReportingEventContext.from_workflow_definition("workplan"),
            user=user,
            internal_event_client=mock_container.internal_event.client(),
            catalog_items=None,
        )

    @pytest.mark.usefixtures("feature_flag_context")
    def test_graph_compiles_with_supervisor_and_all_subagent_nodes(self, builder):
        config = _config()

        graph = builder.build(config)
        compiled = graph.compile()
        assert isinstance(compiled, CompiledStateGraph)

        node_names = set(compiled.nodes.keys())
        assert "research#agent" in node_names
        assert "planner#agent" in node_names
        assert "score_fetch#deterministic_step" in node_names
        assert "score_fetch_notes#deterministic_step" in node_names
        assert self.EXPECTED_SUPERVISOR_NODES <= node_names
        # Every subagent is attached as a real dispatchable node of the
        # supervisor's graph (native Send targets), keyed by its own name.
        for subagent in ROUND_ONE_EVALUATORS + CROSS_EXAMINERS:
            assert subagent in node_names

    @pytest.fixture(name="sequential_feature_flag_context")
    def sequential_feature_flag_context_fixture(self):
        token = current_feature_flag_context.set(set())
        yield
        current_feature_flag_context.reset(token)

    @pytest.mark.usefixtures("sequential_feature_flag_context")
    @pytest.mark.parametrize("component_name", CROSS_EXAMINERS)
    def test_sequential_cross_examiners_read_the_delegation_prompt(
        self, builder, component_name
    ):
        """With ``dap_parallel_subagents`` off, cross-examiners still read the delegation prompt."""
        # Full build: subagents are bound to the supervisor at attach time.
        built: list[dict] = []
        build_components = builder._build_components

        def capture(*args):
            built.append(build_components(*args))
            return built[-1]

        with patch.object(builder, "_build_components", side_effect=capture):
            builder.build(_config())
        supervisor = built[0]["readiness_supervisor"]
        # Compared by name: @inject wraps the class in a function.
        assert type(supervisor).__name__ == "SupervisorAgentComponent"

        inputs = supervisor.subagent_components[component_name].inputs
        assert [i.template_variable_name for i in inputs] == ["goal"]
        assert isinstance(inputs[0], RuntimeIOKey)

    def test_supervisor_prompt_handles_parallel_delegation_rejection(self):
        """A rejected parallel turn must not read as an evaluator failure.

        With ``dap_parallel_subagents`` off, the sequential supervisor rejects
        a multi-``delegate_task`` turn with a recoverable error. Unless the
        prompt says so, the Turn 1 STOP rule (no score when any round-1
        delegation errored) can swallow that rejection and end the run with no
        score.
        """
        prompt = next(
            p
            for p in _config().prompts or []
            if p.prompt_id == "workplan_readiness_supervisor_prompt"
        )
        system = prompt.prompt_template["system"]
        assert "one subagent at a time" in system
        assert "not an evaluator failure" in system


class TestWorkplanV110Subagents:
    """Every evaluator is a description-carrying, tool-less AgentComponent.

    ``AgentComponent.compile_as_subagent`` raises without a description, and
    the evaluators reason only over what they are given — none may hold tools.
    """

    @pytest.mark.parametrize(
        "component_name,prompt_id,schema_id",
        [
            ("rubric", "rs_rubric_prompt", "rs_evaluator_result"),
            ("coverage", "rs_coverage_prompt", "rs_evaluator_result"),
            ("falsifier", "rs_falsifier_prompt", "rs_falsifier_result"),
            ("xexam_rubric", "rs_xexam_scorer_prompt", "rs_crossexam_result"),
            ("xexam_coverage", "rs_xexam_scorer_prompt", "rs_crossexam_result"),
            (
                "xexam_falsifier",
                "rs_xexam_falsifier_prompt",
                "rs_crossexam_falsifier_result",
            ),
        ],
    )
    def test_subagent_prompt_and_schema_bindings(
        self, component_name, prompt_id, schema_id
    ):
        component = _component(_config(), component_name)
        assert component["type"] == "AgentComponent"
        assert component["prompt_id"] == prompt_id
        assert component["response_schema_id"] == schema_id
        assert component["response_schema_version"] == "^1.0.0"
        assert component["toolset"] == []
        assert component.get("description")

    @pytest.mark.parametrize("component_name", ROUND_ONE_EVALUATORS)
    def test_round_one_evaluators_read_the_fetch_results(self, component_name):
        # Subagent inputs resolve inside the isolated dispatch state, which
        # inherits the parent context — so the evaluators bind the fetch
        # results directly and must not bind context:goal (the delegation
        # prompt overwrites it).
        component = _component(_config(), component_name)
        inputs = {i["as"]: i["from"] for i in component["inputs"]}
        assert inputs == {
            "work_item": "context:score_fetch.tool_responses",
            "notes": "context:score_fetch_notes.tool_responses",
        }

    @pytest.mark.parametrize("component_name", CROSS_EXAMINERS)
    def test_cross_examiners_take_their_inputs_from_the_delegation_prompt(
        self, component_name
    ):
        # The supervisor copies the round-1 results into the delegation prompt.
        component = _component(_config(), component_name)
        assert component["inputs"] == [{"from": "context:goal"}]

    @pytest.mark.parametrize(
        "prompt_id", ["rs_xexam_scorer_prompt", "rs_xexam_falsifier_prompt"]
    )
    def test_cross_exam_prompts_render_the_delegation_prompt(self, prompt_id):
        prompt = next(p for p in _config().prompts or [] if p.prompt_id == prompt_id)
        assert prompt.prompt_template["user"].strip() == "{{goal}}"
