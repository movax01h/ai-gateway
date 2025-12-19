"""Tests for AI prompt scanning integration.

These tests verify that the security scanning integration works correctly
based on the use_ai_prompt_scanning feature flag and tool trust level:

- Feature flag enabled + untrusted tool: PromptSecurity runs, then HiddenLayer scan
- Feature flag enabled + trusted tool: PromptSecurity runs, HiddenLayer skipped
- Feature flag disabled: PromptSecurity runs, HiddenLayer skipped

PromptSecurity sanitization always runs regardless of feature flag state.
"""

import asyncio
from unittest.mock import MagicMock, patch

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tracking import MonitoringContext


class TestApplySecurityScanning:
    """Test the apply_security_scanning helper function."""

    def test_feature_flag_enabled_untrusted_tool_runs_security_and_scan(self):
        """When feature flag is enabled and tool is untrusted, both security layers run."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory.scan_prompt"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            content = "test content"
            result = apply_security_scanning(
                response=content,
                tool_name="test_tool",
                trust_level=None,  # Untrusted (defaults to UNTRUSTED_USER_CONTENT)
            )

            mock_security.apply_security_to_tool_response.assert_called_once_with(
                response=content, tool_name="test_tool"
            )
            mock_scan.assert_called_once_with("sanitized content")
            assert result == "sanitized content"

    def test_feature_flag_enabled_trusted_tool_skips_scan(self):
        """When feature flag is enabled but tool is trusted, HiddenLayer scan is skipped."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory.scan_prompt"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            content = "test content"
            result = apply_security_scanning(
                response=content,
                tool_name="test_tool",
                trust_level=ToolTrustLevel.TRUSTED_INTERNAL,
            )

            mock_security.apply_security_to_tool_response.assert_called_once_with(
                response=content, tool_name="test_tool"
            )
            mock_scan.assert_not_called()
            assert result == "sanitized content"

    def test_feature_flag_disabled_skips_scan(self):
        """When feature flag is disabled, HiddenLayer scan is skipped regardless of trust level."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory.scan_prompt"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=False
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            content = "test content"
            result = apply_security_scanning(
                response=content,
                tool_name="test_tool",
                trust_level=None,
            )

            mock_security.apply_security_to_tool_response.assert_called_once_with(
                response=content, tool_name="test_tool"
            )
            mock_scan.assert_not_called()
            assert result == "sanitized content"


class TestScanPrompt:
    """Test the scan_prompt function behavior."""

    def test_scan_prompt_schedules_async_task(self):
        """Verify scan_prompt schedules an async task without blocking."""
        from duo_workflow_service.security.scanner_factory import scan_prompt

        with patch(
            "duo_workflow_service.security.scanner_factory._get_hiddenlayer_scanner"
        ) as mock_get_scanner:
            mock_scanner = MagicMock()
            mock_scanner.enabled = True
            mock_get_scanner.return_value = mock_scanner

            async def run_test():
                result = scan_prompt("test content")
                # scan_prompt should return immediately (fire-and-forget)
                assert result == "test content"
                # Give async task a chance to be scheduled
                await asyncio.sleep(0.01)

            asyncio.run(run_test())

    def test_scan_prompt_returns_original_content(self):
        """Verify scan_prompt returns the original content unchanged."""
        from duo_workflow_service.security.scanner_factory import scan_prompt

        with patch(
            "duo_workflow_service.security.scanner_factory._get_hiddenlayer_scanner"
        ) as mock_get_scanner:
            mock_scanner = MagicMock()
            mock_scanner.enabled = False
            mock_get_scanner.return_value = mock_scanner

            content = {"key": "value", "nested": {"data": "test"}}
            result = scan_prompt(content)

            assert result == content

    def test_scan_prompt_handles_nested_structures(self):
        """Verify scan_prompt processes nested data structures."""
        from duo_workflow_service.security.scanner_factory import scan_prompt

        with patch(
            "duo_workflow_service.security.scanner_factory._get_hiddenlayer_scanner"
        ) as mock_get_scanner:
            mock_scanner = MagicMock()
            mock_scanner.enabled = False
            mock_get_scanner.return_value = mock_scanner

            content = {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ],
                "metadata": {"source": "test"},
            }
            result = scan_prompt(content)

            assert result == content

    def test_scan_prompt_handles_no_event_loop(self):
        """Verify scan_prompt handles case when no event loop is running."""
        from duo_workflow_service.security.scanner_factory import scan_prompt

        with patch(
            "duo_workflow_service.security.scanner_factory._get_hiddenlayer_scanner"
        ) as mock_get_scanner:
            mock_scanner = MagicMock()
            mock_scanner.enabled = True
            mock_get_scanner.return_value = mock_scanner

            # Call without an event loop - should not raise
            content = "test content"
            result = scan_prompt(content)

            assert result == content
