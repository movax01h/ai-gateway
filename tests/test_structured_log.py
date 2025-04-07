from contextlib import ExitStack
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import structlog

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.structured_logging import prevent_logging_if_disabled, sanitize_logs


class TestSanitizeLogs:
    @pytest.fixture(scope="class")
    def inputs_with_model_metadata(self):
        inputs = MagicMock(
            model_metadata=ModelMetadata(
                name="mistral",
                provider="openai",
                api_key="secret-key-456",
                endpoint="https://example.com",
            ),
            other_fied="other_value",
        )

        return inputs

    def test_sanitize_api_key(self):
        # Test when api_key is present
        event_dict = {"api_key": "secret-key-123"}
        result = sanitize_logs(None, None, event_dict)
        assert result["api_key"] == "**********"

    def test_sanitize_missing_api_key(self):
        # Test when api_key is not present
        event_dict = {"other_field": "value"}
        result = sanitize_logs(None, None, event_dict)
        assert result["api_key"] is None

    def test_sanitize_inputs_with_model_metadata(self, inputs_with_model_metadata):
        event_dict = {"inputs": inputs_with_model_metadata}

        result = sanitize_logs(None, None, event_dict)

        assert result["inputs"].model_metadata.api_key == "**********"
        assert str(result["inputs"].model_metadata.endpoint) == "https://example.com/"
        assert result["inputs"].other_fied == "other_value"

    def test_sanitize_inputs_without_model_metadata(self):
        # Test when inputs exist but without model_metadata
        inputs = SimpleNamespace(other_field="test")
        event_dict = {"inputs": inputs}

        result = sanitize_logs(None, None, event_dict)
        assert result["inputs"].other_field == "test"

    def test_sanitize_no_inputs(self):
        # Test when no inputs field exists
        event_dict = {"some_field": "value"}
        result = sanitize_logs(None, None, event_dict)
        assert "inputs" not in result
        assert result["some_field"] == "value"


class TestPreventLoggingIfDisabled:
    class Case(NamedTuple):
        enable_request_logging: bool
        custom_models_enabled: bool
        enabled_instance_verbose_ai_logs: bool
        feature_flag_enabled: bool

    def _setup_logging_patches(self, case):
        """Helper method to set up common patches for logging tests"""
        return [
            patch(
                "ai_gateway.structured_logging.ENABLE_REQUEST_LOGGING",
                case.enable_request_logging,
            ),
            patch(
                "ai_gateway.structured_logging.CUSTOM_MODELS_ENABLED",
                case.custom_models_enabled,
            ),
            patch(
                "ai_gateway.structured_logging.is_feature_enabled",
                return_value=case.feature_flag_enabled,
            ),
            patch(
                "ai_gateway.structured_logging.enabled_instance_verbose_ai_logs",
                return_value=case.enabled_instance_verbose_ai_logs,
            ),
        ]

    CASES_WHERE_LOGS_SHOULD_NOT_BE_DROPPED = [
        # request logging enabled at AIGW level
        Case(
            enable_request_logging=True,
            custom_models_enabled=False,
            enabled_instance_verbose_ai_logs=False,
            feature_flag_enabled=False,
        ),
        # request logging disabled, custom models enabled, enabled_instance_verbose_ai_logs enabled
        Case(
            enable_request_logging=False,
            custom_models_enabled=True,
            enabled_instance_verbose_ai_logs=True,
            feature_flag_enabled=False,
        ),
        # request logging disabled, custom models disabled, feature flag enabled
        Case(
            enable_request_logging=False,
            custom_models_enabled=False,
            enabled_instance_verbose_ai_logs=False,
            feature_flag_enabled=True,
        ),
    ]

    @pytest.mark.parametrize("case", CASES_WHERE_LOGS_SHOULD_NOT_BE_DROPPED)
    def test_events_are_not_dropped(self, case):
        with ExitStack() as stack:
            for patch in self._setup_logging_patches(case):
                stack.enter_context(patch)
            event_dict = {"key": "value"}
            result = prevent_logging_if_disabled(None, None, event_dict)
            assert result == event_dict

    CASES_WHERE_LOGS_SHOULD_BE_DROPPED = [
        # request logging disabled, custom models disabled, enabled_instance_verbose_ai_logs enabled
        Case(
            enable_request_logging=False,
            custom_models_enabled=False,
            enabled_instance_verbose_ai_logs=True,
            feature_flag_enabled=False,
        ),
        # request logging disabled, custom models enabled, feature flag enabled
        Case(
            enable_request_logging=False,
            custom_models_enabled=True,
            enabled_instance_verbose_ai_logs=False,
            feature_flag_enabled=True,
        ),
    ]

    @pytest.mark.parametrize("case", CASES_WHERE_LOGS_SHOULD_BE_DROPPED)
    def test_logging_disabled(self, case):
        with ExitStack() as stack:
            for patch in self._setup_logging_patches(case):
                stack.enter_context(patch)
            with pytest.raises(structlog.DropEvent):
                prevent_logging_if_disabled(None, None, {"key": "value"})
