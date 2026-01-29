"""Tests for AI prompt scanning integration.

Tests verify security scanning behavior based on prompt_injection_protection_level:
- NO_CHECKS: PromptSecurity only, no HiddenLayer scanning
- LOG_ONLY: PromptSecurity + non-blocking HiddenLayer scan (threats logged)
- INTERRUPT: PromptSecurity + blocking HiddenLayer scan (raises on detection)

PromptSecurity sanitization always runs regardless of protection level.
"""

from unittest.mock import MagicMock, patch

import pytest

from duo_workflow_service.gitlab.schema import PromptInjectionProtectionLevel
from duo_workflow_service.security.prompt_scanner import DetectionType, ScanResult
from duo_workflow_service.security.scanner_factory import PromptInjectionDetectedError
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tracking import MonitoringContext


class TestApplySecurityScanning:
    """Test apply_security_scanning with different protection levels."""

    def test_no_checks_skips_hiddenlayer_scan(self):
        """NO_CHECKS mode: PromptSecurity runs, HiddenLayer skipped."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                prompt_injection_protection_level=PromptInjectionProtectionLevel.NO_CHECKS
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response="test content",
                tool_name="test_tool",
                trust_level=None,
            )

            mock_security.apply_security_to_tool_response.assert_called_once()
            mock_scan.assert_not_called()
            assert result == "sanitized content"

    def test_log_only_runs_fire_and_forget_scan(self):
        """LOG_ONLY mode: PromptSecurity runs, HiddenLayer scan is non-blocking."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=PromptInjectionProtectionLevel.LOG_ONLY,
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response="test content",
                tool_name="test_tool",
                trust_level=None,
            )

            mock_security.apply_security_to_tool_response.assert_called_once()
            mock_scan.assert_called_once_with("sanitized content")
            assert result == "sanitized content"

    def test_interrupt_runs_blocking_scan(self):
        """INTERRUPT mode: PromptSecurity runs, HiddenLayer scan blocks."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._run_blocking_scan"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=PromptInjectionProtectionLevel.INTERRUPT,
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response="test content",
                tool_name="test_tool",
                trust_level=None,
            )

            mock_security.apply_security_to_tool_response.assert_called_once()
            mock_scan.assert_called_once()
            assert result == "sanitized content"

    def test_trusted_tool_skips_scan_regardless_of_level(self):
        """Trusted tools skip HiddenLayer scan in all modes."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan"
            ) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                prompt_injection_protection_level=PromptInjectionProtectionLevel.LOG_ONLY
            )
            mock_security.apply_security_to_tool_response.return_value = (
                "sanitized content"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response="test content",
                tool_name="test_tool",
                trust_level=ToolTrustLevel.TRUSTED_INTERNAL,
            )

            mock_security.apply_security_to_tool_response.assert_called_once()
            mock_scan.assert_not_called()
            assert result == "sanitized content"


class TestInterruptModeBlocking:
    """Test INTERRUPT mode blocking behavior."""

    def test_interrupt_raises_on_threat_detection(self):
        """INTERRUPT mode raises PromptInjectionDetectedError when threat detected."""
        threat_result = ScanResult(
            detected=True,
            blocked=True,
            detection_type=DetectionType.PROMPT_INJECTION,
            confidence=0.95,
            details="Malicious prompt injection detected",
        )

        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._run_blocking_scan"
            ) as mock_blocking_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=PromptInjectionProtectionLevel.INTERRUPT,
            )
            mock_security.apply_security_to_tool_response.return_value = "content"
            # _run_blocking_scan raises exception when threat detected
            mock_blocking_scan.side_effect = PromptInjectionDetectedError(
                threat_result, "dangerous_tool"
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            with pytest.raises(PromptInjectionDetectedError) as exc_info:
                apply_security_scanning(
                    response="malicious content",
                    tool_name="dangerous_tool",
                    trust_level=None,
                )

            assert exc_info.value.tool_name == "dangerous_tool"
            assert exc_info.value.scan_result == threat_result


class TestHiddenLayerConfig:
    """Test HiddenLayerConfig configuration handling."""

    def test_from_environment_includes_project_id(self):
        """Verify from_environment loads HL_PROJECT_ID."""
        from duo_workflow_service.security.hidden_layer_scanner import HiddenLayerConfig

        with patch.dict(
            "os.environ",
            {
                "HL_CLIENT_ID": "test-client-id",
                "HL_CLIENT_SECRET": "test-client-secret",
                "HL_PROJECT_ID": "internal-search-chatbot",
            },
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()

            assert config.client_id == "test-client-id"
            assert config.client_secret == "test-client-secret"
            assert config.project_id == "internal-search-chatbot"

    def test_from_environment_project_id_optional(self):
        """Verify from_environment handles missing HL_PROJECT_ID."""
        from duo_workflow_service.security.hidden_layer_scanner import HiddenLayerConfig

        # Mock os.getenv to return None for HL_PROJECT_ID
        def mock_getenv(key, default=None):
            env_values = {
                "HL_CLIENT_ID": "test-client-id",
                "HL_CLIENT_SECRET": "test-client-secret",
                "HIDDENLAYER_ENVIRONMENT": "prod-us",
                "HIDDENLAYER_BASE_URL": None,
                "HL_PROJECT_ID": None,
            }
            return env_values.get(key, default)

        with patch("os.getenv", side_effect=mock_getenv):
            config = HiddenLayerConfig.from_environment()

            assert config.project_id is None

    def test_config_default_project_id_is_none(self):
        """Verify HiddenLayerConfig defaults project_id to None."""
        from duo_workflow_service.security.hidden_layer_scanner import HiddenLayerConfig

        config = HiddenLayerConfig()
        assert config.project_id is None

    def test_scanner_passes_project_id_header_to_client(self):
        """Verify HiddenLayerScanner passes HL-Project-Id header when configured."""
        from duo_workflow_service.security.hidden_layer_scanner import (
            HiddenLayerConfig,
            HiddenLayerScanner,
        )

        config = HiddenLayerConfig(
            client_id="test-client-id",
            client_secret="test-client-secret",
            project_id="my-project",
        )

        with patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class:
            _ = HiddenLayerScanner(config=config)

            mock_client_class.assert_called_once_with(
                client_id="test-client-id",
                client_secret="test-client-secret",
                default_headers={"HL-Project-Id": "my-project"},
                environment="prod-us",
            )

    def test_scanner_no_headers_when_project_id_not_set(self):
        """Verify HiddenLayerScanner omits default_headers when project_id not set."""
        from duo_workflow_service.security.hidden_layer_scanner import (
            HiddenLayerConfig,
            HiddenLayerScanner,
        )

        config = HiddenLayerConfig(
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

        with patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class:
            _ = HiddenLayerScanner(config=config)

            # default_headers should NOT be passed when project_id is not set
            mock_client_class.assert_called_once_with(
                client_id="test-client-id",
                client_secret="test-client-secret",
                environment="prod-us",
            )


class TestUseAiPromptScanningFlag:
    """Test that use_ai_prompt_scanning flag controls HiddenLayer scanning."""

    @pytest.mark.parametrize(
        "use_ai_prompt_scanning,protection_level,should_skip_scan",
        [
            # use_ai_prompt_scanning=False should skip scanning regardless of protection level
            (False, PromptInjectionProtectionLevel.LOG_ONLY, True),
            (False, PromptInjectionProtectionLevel.INTERRUPT, True),
            (False, PromptInjectionProtectionLevel.NO_CHECKS, True),
            # use_ai_prompt_scanning=True with NO_CHECKS should skip scanning
            (True, PromptInjectionProtectionLevel.NO_CHECKS, True),
            # use_ai_prompt_scanning=True with LOG_ONLY should scan
            (True, PromptInjectionProtectionLevel.LOG_ONLY, False),
            # use_ai_prompt_scanning=True with INTERRUPT should scan
            (True, PromptInjectionProtectionLevel.INTERRUPT, False),
        ],
    )
    def test_use_ai_prompt_scanning_controls_hiddenlayer_scanning(
        self,
        use_ai_prompt_scanning,
        protection_level,
        should_skip_scan,
    ):
        """Test that use_ai_prompt_scanning flag properly controls HiddenLayer scanning."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan"
            ) as mock_fire_and_forget,
            patch(
                "duo_workflow_service.security.scanner_factory._run_blocking_scan"
            ) as mock_blocking_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            # Setup monitoring context
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=use_ai_prompt_scanning,
                prompt_injection_protection_level=protection_level,
            )

            # Setup sanitization to return the input unchanged
            mock_security.apply_security_to_tool_response.return_value = "test response"

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            # Call apply_security_scanning
            result = apply_security_scanning(
                response="test response",
                tool_name="test_tool",
                trust_level=ToolTrustLevel.UNTRUSTED_USER_CONTENT,
            )

            # Verify result
            assert result == "test response"

            # Verify HiddenLayer scanning behavior
            if should_skip_scan:
                # Neither fire-and-forget nor blocking scan should be called
                mock_fire_and_forget.assert_not_called()
                mock_blocking_scan.assert_not_called()
            else:
                # One of the scan methods should be called based on protection level
                if protection_level == PromptInjectionProtectionLevel.LOG_ONLY:
                    mock_fire_and_forget.assert_called_once()
                    mock_blocking_scan.assert_not_called()
                elif protection_level == PromptInjectionProtectionLevel.INTERRUPT:
                    mock_fire_and_forget.assert_not_called()
                    mock_blocking_scan.assert_called_once()

    def test_trusted_tools_skip_scanning_regardless_of_flag(self):
        """Test that TRUSTED_INTERNAL tools skip HiddenLayer scanning even when flag is enabled."""
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan"
            ) as mock_fire_and_forget,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            # Setup monitoring context with scanning enabled
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=PromptInjectionProtectionLevel.LOG_ONLY,
            )

            # Setup sanitization to return the input unchanged
            mock_security.apply_security_to_tool_response.return_value = "test response"

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            # Call apply_security_scanning with TRUSTED_INTERNAL tool
            result = apply_security_scanning(
                response="test response",
                tool_name="test_tool",
                trust_level=ToolTrustLevel.TRUSTED_INTERNAL,
            )

            # Verify result
            assert result == "test response"

            # Verify no scanning was performed
            mock_fire_and_forget.assert_not_called()
