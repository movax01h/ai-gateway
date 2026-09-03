"""Guards for the shipped readiness_score flow config.

Both fetch steps (``fetch`` → get_work_item, ``fetch_notes`` → get_work_item_notes)
must abort the flow on failure rather than silently continuing into the evaluators
with missing context and producing a misleading readiness score.

See the review comment on !6678 and the Flow Registry contribution guidelines
(docs/flow_registry/contribution_guidelines.md).
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


class TestReadinessScoreAbortsOnFailedFetch:
    """A failed work-item or notes fetch must route to abort, not into the evaluators."""

    @staticmethod
    def _router_for(config: FlowConfig, from_component: str) -> dict:
        return next(r for r in config.routers if r["from"] == from_component)

    @pytest.mark.parametrize(
        "from_component,expected_input",
        [
            ("fetch", "context:fetch.execution_result"),
            ("fetch_notes", "context:fetch_notes.execution_result"),
        ],
    )
    def test_fetch_routes_on_execution_result_not_unconditionally(
        self, from_component, expected_input
    ):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        router = self._router_for(config, from_component)
        assert "to" not in router, (
            f"{from_component} must not route unconditionally — a failed fetch "
            "would otherwise reach the evaluators with missing context and emit "
            "a misleading readiness score"
        )
        assert router["condition"]["input"] == expected_input

    @pytest.mark.parametrize(
        "from_component,success_target",
        [
            ("fetch", "fetch_notes"),
            ("fetch_notes", "rubric"),
        ],
    )
    def test_success_continues_and_failure_aborts(self, from_component, success_target):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        routes = self._router_for(config, from_component)["condition"]["routes"]
        assert routes["success"] == success_target
        assert routes["default_route"] == "abort"

    def test_unconditional_routers_wire_rubric_to_coverage_to_end(self):
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        rubric_router = self._router_for(config, "rubric")
        coverage_router = self._router_for(config, "coverage")

        assert rubric_router.get("to") == "coverage"
        assert coverage_router.get("to") == "end"

    def test_config_loads_successfully(self):
        """Smoke-test: the YAML is valid and parses without error."""
        config = FlowConfig.from_yaml_config("readiness_score", "1.0.0")
        assert config.flow.entry_point == "fetch"
        assert config.environment == "ambient"
