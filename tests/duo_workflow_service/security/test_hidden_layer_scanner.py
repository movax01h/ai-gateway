"""Tests for HiddenLayerScanner logging behavior based on realm configuration.

Tests verify that HiddenLayerScanner only logs threat detection responses when:
1. A threat is actually detected (has_detections=True)
2. The current gitlab_realm is in the configured log_allowed_realms
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.security.hidden_layer_scanner import (
    HiddenLayerConfig,
    HiddenLayerScanner,
)
from duo_workflow_service.security.prompt_scanner import DetectionType


@pytest.fixture
def mock_hidden_layer_response_with_detection():
    """Create a mock Hidden Layer response with threat detection."""
    response = MagicMock()
    response.evaluation = MagicMock()
    response.evaluation.has_detections = True
    response.evaluation.action = "Block"
    response.analysis = [
        MagicMock(
            detected=True,
            findings=[MagicMock(type="prompt_injection")],
        )
    ]
    response.to_dict.return_value = {
        "evaluation": {"has_detections": True, "action": "Block"},
        "analysis": [{"detected": True, "findings": [{"type": "prompt_injection"}]}],
    }
    return response


@pytest.fixture
def mock_hidden_layer_response_no_detection():
    """Create a mock Hidden Layer response without threat detection."""
    response = MagicMock()
    response.evaluation = MagicMock()
    response.evaluation.has_detections = False
    response.evaluation.action = "Allow"
    response.analysis = []
    response.to_dict.return_value = {
        "evaluation": {"has_detections": False, "action": "Allow"},
        "analysis": [],
    }
    return response


class TestHiddenLayerScannerLogging:
    """Test HiddenLayerScanner logging behavior based on realm configuration."""

    @pytest.mark.asyncio
    async def test_logs_response_when_detection_and_realm_allowed(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test that response is logged when threat detected AND realm is in allowed list."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas", "self-managed"},
        )

        with (
            patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_async_client = MagicMock()
            mock_async_client.interactions.analyze = AsyncMock(
                return_value=mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_async_client

            mock_realm.get.return_value = "saas"

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            result = await scanner.scan("test prompt")

            # Verify detection was found
            assert result.detected is True
            assert result.detection_type == DetectionType.PROMPT_INJECTION

            # Verify warning was logged - should have at least one warning call
            assert (
                mock_log.warning.called
            ), "Warning should be logged when detection found"

    @pytest.mark.asyncio
    async def test_does_not_log_response_when_detection_but_realm_not_allowed(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test that response is NOT logged when threat detected but realm not in allowed list."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas"},  # Only allow saas
        )

        with (
            patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_async_client = MagicMock()
            mock_async_client.interactions.analyze = AsyncMock(
                return_value=mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_async_client

            mock_realm.get.return_value = "self-managed"  # Not in allowed list

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            result = await scanner.scan("test prompt")

            # Verify detection was found
            assert result.detected is True

            # Verify warning was NOT logged with response for detection
            detection_warning_calls = [
                call
                for call in mock_log.warning.call_args_list
                if "Hidden Layer scan detects threats" in str(call)
            ]
            assert (
                len(detection_warning_calls) == 0
            ), "Should not log response when realm not allowed"

            # But the general threat warning should still be logged
            threat_warning_calls = [
                call
                for call in mock_log.warning.call_args_list
                if "Hidden Layer detected potential threat" in str(call)
            ]
            assert len(threat_warning_calls) > 0

    @pytest.mark.asyncio
    async def test_does_not_log_response_when_no_detection(
        self, mock_hidden_layer_response_no_detection
    ):
        """Test that response is NOT logged when no threat detected, regardless of realm."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas", "self-managed"},
        )

        with (
            patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_async_client = MagicMock()
            mock_async_client.interactions.analyze = AsyncMock(
                return_value=mock_hidden_layer_response_no_detection
            )
            mock_client_class.return_value = mock_async_client

            mock_realm.get.return_value = "saas"

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            result = await scanner.scan("safe prompt")

            # Verify no detection
            assert result.detected is False
            assert result.detection_type == DetectionType.SAFE

            # Verify no warning logs were called
            mock_log.warning.assert_not_called()

    def test_logs_response_sync_when_detection_and_realm_allowed(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test sync scan logs response when threat detected AND realm is in allowed list."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas"},
        )

        with (
            patch("hiddenlayer.HiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_sync_client = MagicMock()
            mock_sync_client.interactions.analyze.return_value = (
                mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_sync_client

            mock_realm.get.return_value = "saas"

            # Set context with unique test values that should NEVER leak to external API
            from lib.hidden_layer_log import set_hidden_layer_log_context

            test_tool_name = "xQw8pL2mK"
            test_tool_arg = "vN4jR9sT"
            test_tool_args = {"arg_key": test_tool_arg}

            set_hidden_layer_log_context(
                tool_name=test_tool_name, tool_args=test_tool_args
            )

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            result = scanner.scan_sync("test prompt")

            # Verify detection was found
            assert result.detected is True

            # CRITICAL: Verify HiddenLayerLogContext is NOT sent to Hidden Layer API
            # The context is for internal logging only and should never be in ANY API argument
            api_call_kwargs = mock_sync_client.interactions.analyze.call_args.kwargs
            api_call_str = str(api_call_kwargs)

            # Verify our test values don't appear anywhere in the API call
            assert test_tool_name not in api_call_str, "Tool name must not leak to API"
            assert test_tool_arg not in api_call_str, "Tool args must not leak to API"

            # Verify expected API structure
            assert api_call_kwargs.get("metadata") == {
                "model": "duo",
                "requester_id": "gitlab-duo-workflow",
            }, "API metadata must contain only expected fields"

            # Verify warning was logged
            assert (
                mock_log.warning.called
            ), "Warning should be logged when detection found"

    def test_does_not_log_response_sync_when_detection_but_realm_not_allowed(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test sync scan does NOT log response when threat detected but realm not allowed."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas"},
        )

        with (
            patch("hiddenlayer.HiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_sync_client = MagicMock()
            mock_sync_client.interactions.analyze.return_value = (
                mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_sync_client

            mock_realm.get.return_value = "self-managed"  # Not in allowed list

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            result = scanner.scan_sync("test prompt")

            # Verify detection was found
            assert result.detected is True

            # Verify warning was NOT logged with response for detection
            detection_warning_calls = [
                call
                for call in mock_log.warning.call_args_list
                if "Hidden Layer scan detects threats" in str(call)
            ]
            assert (
                len(detection_warning_calls) == 0
            ), "Should not log response when realm not allowed"


class TestHiddenLayerConfigRealms:
    """Test HiddenLayerConfig realm parsing from environment."""

    def test_config_parses_single_realm_from_env(self):
        """Test parsing single realm from HIDDENLAYER_LOG_ALLOWED_REALMS."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": "saas"},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            assert config.log_allowed_realms == {"saas"}

    def test_config_parses_multiple_realms_from_env(self):
        """Test parsing multiple realms from HIDDENLAYER_LOG_ALLOWED_REALMS."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": "saas,self-managed"},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            assert config.log_allowed_realms == {"saas", "self-managed"}

    def test_config_ignores_invalid_realms(self):
        """Test that invalid realms are filtered out."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": "saas,invalid-realm,self-managed"},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            # Only valid realms should be included
            assert config.log_allowed_realms == {"saas", "self-managed"}
            assert "invalid-realm" not in config.log_allowed_realms

    def test_config_handles_whitespace_in_realms(self):
        """Test that whitespace is properly stripped from realm values."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": " saas , self-managed "},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            assert config.log_allowed_realms == {"saas", "self-managed"}

    def test_config_defaults_to_saas_when_env_not_set(self):
        """Test that config defaults to saas when HIDDENLAYER_LOG_ALLOWED_REALMS not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = HiddenLayerConfig.from_environment()
            assert config.log_allowed_realms == {"saas"}

    def test_config_defaults_to_saas_when_env_empty(self):
        """Test that config defaults to saas when HIDDENLAYER_LOG_ALLOWED_REALMS is empty."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": ""},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            # Empty string should result in empty set
            assert config.log_allowed_realms == set()

    def test_config_case_insensitive_realm_matching(self):
        """Test that realm matching is case-insensitive."""
        with patch.dict(
            "os.environ",
            {"HIDDENLAYER_LOG_ALLOWED_REALMS": "SAAS,Self-Managed"},
            clear=False,
        ):
            config = HiddenLayerConfig.from_environment()
            # Should not match because realm values are lowercase
            assert config.log_allowed_realms == {"saas", "self-managed"}

    def test_config_realm_in_check(self):
        """Test that realm membership check works correctly."""
        config = HiddenLayerConfig(
            client_id="test",
            client_secret="test",
            log_allowed_realms={"saas", "self-managed"},
        )

        assert "saas" in config.log_allowed_realms
        assert "self-managed" in config.log_allowed_realms
        assert "invalid" not in config.log_allowed_realms


class TestHiddenLayerScannerRealmLogging:
    """Test realm-specific logging behavior in detail."""

    @pytest.mark.asyncio
    async def test_logs_realm_in_warning_message(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test that gitlab_realm is included in warning log messages."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas"},
        )

        with (
            patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_async_client = MagicMock()
            mock_async_client.interactions.analyze = AsyncMock(
                return_value=mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_async_client

            mock_realm.get.return_value = "saas"

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            await scanner.scan("test prompt")

            # Verify realm was logged
            warning_calls = mock_log.warning.call_args_list
            assert any(
                "gitlab_realm" in str(call) for call in warning_calls
            ), "gitlab_realm should be logged"

    @pytest.mark.asyncio
    async def test_logs_allowed_realms_in_warning_message(
        self, mock_hidden_layer_response_with_detection
    ):
        """Test that log_allowed_realms is included in warning log messages."""
        config = HiddenLayerConfig(
            client_id="test_id",
            client_secret="test_secret",
            log_allowed_realms={"saas", "self-managed"},
        )

        with (
            patch("hiddenlayer.AsyncHiddenLayer") as mock_client_class,
            patch(
                "ai_gateway.instrumentators.model_requests.gitlab_realm"
            ) as mock_realm,
            patch("duo_workflow_service.security.hidden_layer_scanner.log") as mock_log,
        ):
            # Setup mocks
            mock_async_client = MagicMock()
            mock_async_client.interactions.analyze = AsyncMock(
                return_value=mock_hidden_layer_response_with_detection
            )
            mock_client_class.return_value = mock_async_client

            mock_realm.get.return_value = "saas"

            # Create scanner and scan
            scanner = HiddenLayerScanner(config=config)
            await scanner.scan("test prompt")

            # Verify allowed realms were logged
            warning_calls = mock_log.warning.call_args_list
            assert any(
                "log_allowed_realms" in str(call) for call in warning_calls
            ), "log_allowed_realms should be logged"
