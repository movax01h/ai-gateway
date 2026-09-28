# pylint: disable=file-naming-for-tests,import-outside-toplevel
"""Tests for AI prompt scanning integration.

Tests verify security scanning behavior based on prompt_injection_protection_level:
- NO_CHECKS: PromptSecurity only, no HiddenLayer scanning
- LOG_ONLY: PromptSecurity + non-blocking HiddenLayer scan (threats logged)
- INTERRUPT: PromptSecurity + blocking HiddenLayer scan (raises on detection)

PromptSecurity sanitization always runs regardless of protection level.
"""

from unittest.mock import patch

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
            # One of the scan methods should be called based on protection level
            elif protection_level == PromptInjectionProtectionLevel.LOG_ONLY:
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


class TestImageBlocksAreNotScanned:
    """Image blocks never reach the scanner.

    A text scanner cannot read an injection out of base64, so sending the
    payload finds nothing and adds megabytes to every scan: a blocking
    HiddenLayer round trip in INTERRUPT mode, a megabyte-scale POST per image
    in LOG_ONLY. The whole block is dropped from the scan copy, since in
    LOG_ONLY every remaining string value (type, id, mime type) would be its
    own call. The text that travels alongside the image must still be scanned,
    and the caller's response must come back with its payload intact. The
    helper itself is unit-tested in ``entities/test_image_blocks.py``.
    """

    PAYLOAD = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" * 64
    RESPONSE = [
        {"type": "text", "text": "Read image file: ./x.png (image/png, 1 KB)."},
        {"type": "image", "base64": PAYLOAD, "mime_type": "image/png"},
    ]

    def _scan_with(self, level, scan_patch_target):
        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(scan_patch_target) as mock_scan,
            patch(
                "duo_workflow_service.security.prompt_security.PromptSecurity"
            ) as mock_security,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=level,
            )
            mock_security.apply_security_to_tool_response.return_value = self.RESPONSE

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response=self.RESPONSE, tool_name="read_file", trust_level=None
            )
            return mock_scan, result

    def test_interrupt_scans_the_text_without_the_payload(self):
        mock_scan, result = self._scan_with(
            PromptInjectionProtectionLevel.INTERRUPT,
            "duo_workflow_service.security.scanner_factory._run_blocking_scan",
        )

        # Exactly the text block, joined the way the scanner joins: its `type`
        # value and its text. Nothing of the image block, payload or otherwise.
        scanned_text = mock_scan.call_args[0][0]
        assert scanned_text == "text Read image file: ./x.png (image/png, 1 KB)."
        # The response handed back to the model keeps its pixels.
        assert result[1]["base64"] == self.PAYLOAD

    def test_log_only_schedules_only_the_text_block(self):
        mock_scan, result = self._scan_with(
            PromptInjectionProtectionLevel.LOG_ONLY,
            "duo_workflow_service.security.scanner_factory._schedule_fire_and_forget_scan",
        )

        # One block left to walk, so one scan for its text (plus the scheduler's
        # pre-existing one for the block's `type` value), none for the image.
        scheduled = mock_scan.call_args[0][0]
        assert scheduled == [self.RESPONSE[0]]
        assert result[1]["base64"] == self.PAYLOAD


class TestImagePayloadExemptionSurvivesSanitization:
    """The exemption keys on shape because the sanitization step erases provenance.

    ``PromptSecurity.apply_security_to_tool_response`` rebuilds every dict for a
    tool that has security functions, so the ``_InternalImageBlock`` marker the
    producer set is a plain dict by the time the scan step sees it. Keying the
    exemption on provenance would therefore exempt nothing for those tools. This
    runs the real sanitizer, no mocks on it, for both kinds of tool.
    """

    PAYLOAD = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" * 64

    def _response(self):
        from duo_workflow_service.entities.image_blocks import image_content_block

        return [
            {"type": "text", "text": "Read image file: ./x.png (image/png, 1 KB)."},
            image_content_block(base64=self.PAYLOAD, mime_type="image/png"),
        ]

    @pytest.mark.parametrize(
        "tool_name",
        [
            # Empty security-function override: the block reaches the scan step untouched.
            "read_file",
            # Default security functions: every dict is rebuilt, the provenance marker is lost.
            "run_mcp_tool",
        ],
    )
    def test_interrupt_never_scans_the_payload_whatever_the_tool(self, tool_name):
        from duo_workflow_service.entities.image_blocks import is_internal_image_block

        with (
            patch(
                "duo_workflow_service.tracking.current_monitoring_context"
            ) as mock_context,
            patch(
                "duo_workflow_service.security.scanner_factory._run_blocking_scan"
            ) as mock_scan,
        ):
            mock_context.get.return_value = MonitoringContext(
                use_ai_prompt_scanning=True,
                prompt_injection_protection_level=PromptInjectionProtectionLevel.INTERRUPT,
            )

            from duo_workflow_service.security.scanner_factory import (
                apply_security_scanning,
            )

            result = apply_security_scanning(
                response=self._response(), tool_name=tool_name, trust_level=None
            )

        # Exactly the text block; the image block, marker or no marker, is gone.
        scanned_text = mock_scan.call_args[0][0]
        assert scanned_text == "text Read image file: ./x.png (image/png, 1 KB)."
        # The model still gets its pixels, whether or not the marker survived.
        assert result[1]["base64"] == self.PAYLOAD
        # Documents the fact the design rests on: only the override tool keeps provenance.
        assert is_internal_image_block(result[1]) is (tool_name == "read_file")
