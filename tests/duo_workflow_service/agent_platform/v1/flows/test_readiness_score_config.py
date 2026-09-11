"""Guards for the shipped readiness_score flow config.

Both fetch steps (``fetch`` → get_work_item, ``fetch_notes`` → get_work_item_notes)
must abort the flow on failure rather than silently continuing into the evaluators
with missing context and producing a misleading readiness score. The ``persist``
step must likewise abort on failure rather than reaching end and reporting the
flow completed while the score was never written.

The scorer/cross-exam chain must also stay wired through the verdict judge and
the persist step (``xexam_falsifier`` → ``verdict`` → ``persist``, then end on
success or abort on failure): verdict applies the gating/cap logic and persist
is the only node that writes the score back to the work item, so a regression
in those edges must fail CI here rather than only in manual GDK validation.

See the review comment on !6678 and the Flow Registry contribution guidelines
(docs/flow_registry/contribution_guidelines.md).
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


class TestReadinessScoreAbortsOnFailedStep:
    """A failed fetch or persist must route to abort, not silently continue.

    A failed work-item or notes fetch must not reach the evaluators with missing context, and a failed persist must not
    reach end — persisting is the whole point of the step, so its failure must surface instead of reporting the flow
    completed with the score silently dropped.
    """

    @staticmethod
    def _router_for(config: FlowConfig, from_component: str) -> dict:
        return next(r for r in config.routers if r["from"] == from_component)

    @pytest.mark.parametrize(
        "from_component,expected_input",
        [
            ("fetch", "context:fetch.execution_result"),
            ("fetch_notes", "context:fetch_notes.execution_result"),
            ("persist", "context:persist.execution_result"),
        ],
    )
    def test_step_routes_on_execution_result_not_unconditionally(
        self, from_component, expected_input
    ):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        router = self._router_for(config, from_component)
        assert "to" not in router, (
            f"{from_component} must not route unconditionally — a failure "
            "would otherwise go unsurfaced"
        )
        assert router["condition"]["input"] == expected_input

    @pytest.mark.parametrize(
        "from_component,success_target",
        [
            ("fetch", "fetch_notes"),
            ("fetch_notes", "rubric"),
            ("persist", "end"),
        ],
    )
    def test_success_continues_and_failure_aborts(self, from_component, success_target):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        routes = self._router_for(config, from_component)["condition"]["routes"]
        assert routes["success"] == success_target
        assert routes["default_route"] == "abort"

    def test_unconditional_routers_wire_scorers_through_verdict_and_persist(self):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")

        assert self._router_for(config, "rubric").get("to") == "coverage"
        assert self._router_for(config, "coverage").get("to") == "falsifier"
        assert self._router_for(config, "falsifier").get("to") == "xexam_rubric"
        assert self._router_for(config, "xexam_rubric").get("to") == "xexam_coverage"
        assert self._router_for(config, "xexam_coverage").get("to") == "xexam_falsifier"
        assert self._router_for(config, "xexam_falsifier").get("to") == "verdict"
        assert self._router_for(config, "verdict").get("to") == "persist"

    @staticmethod
    def _inputs_for(config: FlowConfig, component_name: str) -> dict:
        component = next(c for c in config.components if c["name"] == component_name)
        return {i["as"]: i["from"] for i in component["inputs"]}

    @pytest.mark.parametrize(
        "component_name,expected_inputs",
        [
            (
                "xexam_rubric",
                {
                    "own_result": "context:rubric.final_answer",
                    "other_result_1": "context:coverage.final_answer",
                    "other_result_2": "context:falsifier.final_answer",
                },
            ),
            (
                "xexam_coverage",
                {
                    "own_result": "context:coverage.final_answer",
                    "other_result_1": "context:rubric.final_answer",
                    "other_result_2": "context:falsifier.final_answer",
                },
            ),
            (
                "xexam_falsifier",
                {
                    "own_result": "context:falsifier.final_answer",
                    "other_result_1": "context:rubric.final_answer",
                    "other_result_2": "context:coverage.final_answer",
                },
            ),
        ],
    )
    def test_xexam_components_bind_own_and_other_results_correctly(
        self, component_name, expected_inputs
    ):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        assert self._inputs_for(config, component_name) == expected_inputs

    def test_config_loads_successfully(self):
        """Smoke-test: the YAML is valid and parses without error."""
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        assert config.flow.entry_point == "fetch"
        assert config.environment == "ambient"


class TestReadinessScoreVerdictOutput:
    """Verdict must emit a structured score rather than prose.

    The aggregate score is written back to the work item's agent plan, so a
    plain-text summary is not enough: dropping the schema would leave the flow
    with no machine-readable number to persist.
    """

    def test_verdict_declares_the_verdict_result_schema(self):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        verdict = next(c for c in config.components if c["name"] == "verdict")

        assert verdict["response_schema_id"] == "rs_verdict_result"
        assert verdict["response_schema_version"] == "^1.0.0"


class TestReadinessScorePersistsVerdict:
    """The score must reach the work item, not only the session log.

    ``persist`` binds ``readiness_score`` and nothing else: the flow scores a plan
    it did not author, so binding ``agent_plan`` as well would overwrite the
    drafted plan.
    """

    @staticmethod
    def _persist(config: FlowConfig) -> dict:
        return next(c for c in config.components if c["name"] == "persist")

    def test_persist_runs_update_work_item_deterministically(self):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        persist = self._persist(config)

        assert persist["type"] == "DeterministicStepComponent"
        assert persist["tool_name"] == "update_work_item"

    def test_persist_binds_the_verdict_score_and_nothing_else(self):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        inputs = {i["as"]: i["from"] for i in self._persist(config)["inputs"]}

        assert inputs == {
            "url": "context:goal",
            "readiness_score": "context:verdict.final_answer.score",
        }
