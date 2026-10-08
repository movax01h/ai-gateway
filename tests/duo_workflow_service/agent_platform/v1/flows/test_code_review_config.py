"""Guards for the shipped code_review flow configs (RFH #5233 regression).

See https://gitlab.com/gitlab-com/request-for-help/-/work_items/5233

If build_review_context fails to resolve a real merge_request_iid, the flow must abort instead of silently continuing to
a forced placeholder publish.

Parametrizes over every shipped config rather than a hardcoded list, so promoting a new version cannot skip the guards.
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW = "code_review"


def shipped_versions() -> list[str]:
    """Every version shipped for the flow, taken from the config filenames."""
    config_dir = FlowConfig.DIRECTORY_PATH / FLOW
    return sorted(path.stem for path in config_dir.glob("*.yml"))


def test_shipped_versions_are_discovered():
    """A bad glob would silently parametrize the guards below into nothing."""
    assert shipped_versions(), f"no shipped configs found for {FLOW}"


@pytest.mark.parametrize("version", shipped_versions())
class TestCodeReviewAbortsOnUnresolvedMergeRequest:
    def _build_review_context_router(self, config: FlowConfig) -> dict:
        return next(
            r for r in config.routers if r.get("from") == "build_review_context"
        )

    def test_routes_on_execution_result_not_unconditionally(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        router = self._build_review_context_router(config)
        assert "to" not in router, (
            "build_review_context must not route unconditionally - a failed "
            "MR-context lookup would otherwise reach post_duo_code_review "
            "with no real merge_request_iid (RFH #5233)"
        )
        assert (
            router["condition"]["input"]
            == "context:build_review_context.execution_result"
        )

    def test_success_continues_and_failure_aborts(self, version):
        config = FlowConfig.from_yaml_config("code_review", version)
        routes = self._build_review_context_router(config)["condition"]["routes"]
        assert routes["success"] == "fetch_mr_diffs"
        assert routes["default_route"] == "abort"


@pytest.mark.parametrize("version", shipped_versions())
def test_explore_step_reads_the_checkout(version):
    """list_repository_tree without a ref reads the default branch, not the merge request."""
    config = FlowConfig.from_yaml_config(FLOW, version)
    explore = next(
        c for c in config.components if c.get("name") == "explore_relevant_directories"
    )

    assert explore["toolset"] == ["find_files"]


def test_3_0_0_reads_the_merge_request_from_the_resource_context():
    config = FlowConfig.from_yaml_config(FLOW, "3.0.0")
    sources = {
        i["from"]
        for c in config.components
        for i in c.get("inputs", [])
        if i["as"] == "merge_request_iid"
    }
    [category] = [
        i for i in config.flow.inputs if i.category == "agent_platform_resource_context"
    ]

    assert sources == {
        "context:inputs.agent_platform_resource_context.merge_request_id"
    }
    assert not category.input_schema["merge_request_id"].optional
