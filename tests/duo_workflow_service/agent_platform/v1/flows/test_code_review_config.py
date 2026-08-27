"""Guards for the shipped code_review flow configs (RFH #5233 regression).

See https://gitlab.com/gitlab-com/request-for-help/-/work_items/5233

If build_review_context fails to resolve a real merge_request_iid, the flow must abort instead of silently continuing to
a forced placeholder publish.
"""

import pytest

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0-dev"])
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
