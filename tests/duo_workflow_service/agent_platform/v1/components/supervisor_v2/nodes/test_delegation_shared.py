"""Test suite for delegation_shared helpers (format_delegation_result)."""

import pytest

from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DelegationStatus,
    format_delegation_result,
)


class TestFormatDelegationResult:
    """Tests for format_delegation_result."""

    def test_completed_status_wraps_content_in_result_tag(self):
        xml = format_delegation_result(
            subagent_name="developer",
            status=DelegationStatus.COMPLETED,
            content="Implementation complete.",
        )
        assert "<subagent_name>developer</subagent_name>" in xml
        assert "<status>completed</status>" in xml
        assert "<result>" in xml
        assert "Implementation complete." in xml
        assert "<error>" not in xml

    def test_error_status_wraps_content_in_error_tag(self):
        xml = format_delegation_result(
            subagent_name="developer",
            status=DelegationStatus.ERROR,
            content="Something went wrong.",
        )
        assert "<status>error</status>" in xml
        assert "<error>" in xml
        assert "Something went wrong." in xml
        assert "<result>" not in xml

    @pytest.mark.parametrize(
        "status", [DelegationStatus.COMPLETED, DelegationStatus.ERROR]
    )
    def test_always_includes_subagent_name(self, status):
        xml = format_delegation_result(
            subagent_name="tester",
            status=status,
            content="content",
        )
        assert "<subagent_name>tester</subagent_name>" in xml

    @pytest.mark.parametrize(
        "status", [DelegationStatus.COMPLETED, DelegationStatus.ERROR]
    )
    def test_never_reports_a_subsession_id(self, status):
        """Nothing can be resumed, so an ID the LLM cannot act on must not be advertised to it."""
        xml = format_delegation_result(
            subagent_name="tester",
            status=status,
            content="content",
        )
        assert "subsession_id" not in xml
