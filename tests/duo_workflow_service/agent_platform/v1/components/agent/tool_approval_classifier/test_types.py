import pytest

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier import (
    MAX_REASON_CHARS,
    ToolApprovalReview,
    ToolApprovalVerdict,
)


class TestToolApprovalReview:
    def test_reason_is_optional(self):
        assert ToolApprovalReview(verdict=ToolApprovalVerdict.ALLOW).reason is None

    def test_converts_a_string_verdict(self):
        review = ToolApprovalReview(verdict="deny")

        assert review.verdict is ToolApprovalVerdict.DENY

    @pytest.mark.parametrize("verdict", ["maybe", None, {"verdict": "allow"}])
    def test_rejects_an_unknown_verdict(self, verdict):
        with pytest.raises(ValueError):
            ToolApprovalReview(verdict=verdict)

    @pytest.mark.parametrize("reason", ["", "   \n"])
    def test_turns_an_empty_reason_into_none(self, reason):
        assert ToolApprovalReview(verdict="ask", reason=reason).reason is None

    def test_strips_and_bounds_the_reason(self):
        review = ToolApprovalReview(verdict="deny", reason=f"  {'r' * 50_000}  ")

        assert review.reason == "r" * MAX_REASON_CHARS
