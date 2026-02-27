import contextvars
from unittest import mock

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from structlog.testing import capture_logs

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from lib.context import (
    get_llm_operations,
    get_token_usage,
    init_llm_operations,
    init_token_usage,
    llm_operations,
    token_usage,
)
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties

DEFAULT_ARGS = {
    "model_engine": "test_engine",
    "model_name": "test_model",
    "error": "no",
    "error_type": "none",
    "streaming": "no",
    "feature_category": "unknown",
    "unit_primitive": "unknown",
    "lsp_version": "unknown",
    "gitlab_version": "unknown",
    "client_type": "unknown",
    "gitlab_realm": "unknown",
    "finish_reason": "unknown",
}


@pytest.fixture(name="unit_primitive")
def unit_primitive_fixture() -> GitLabUnitPrimitive | None:
    return None


@pytest.fixture(name="container")
def container_fixture(
    unit_primitive: GitLabUnitPrimitive | None,
    internal_event_client: InternalEventsClient,
) -> ModelRequestInstrumentator.WatchContainer:
    return ModelRequestInstrumentator.WatchContainer(
        model_provider="test_provider",
        labels={"model_engine": "test_engine", "model_name": "test_model"},
        streaming=False,
        limits=None,
        unit_primitive=unit_primitive,
        internal_event_client=internal_event_client,
    )


def test_get_token_usage():
    usage = {"test_model": {"input_tokens": 10, "output_tokens": 20}}
    token_usage.set(usage)
    assert get_token_usage() == usage
    assert token_usage.get() is None  # Ensure the usage is reset after being retrieved


def test_get_llm_operations():
    operations = [
        {
            "token_count": 12,
            "model_id": "test",
            "model_engine": "test_provider",
            "model_provider": "test_provider",
            "prompt_tokens": 2,
            "completion_tokens": 10,
        },
    ]
    llm_operations.set(operations)
    assert get_llm_operations() == operations
    assert (
        llm_operations.get() is None
    )  # Ensure the operations are reset after being retrieved


class TestWatchContainer:
    @mock.patch("prometheus_client.Counter.labels")
    def test_register_token_usage(self, mock_counters, container):
        init_token_usage()
        init_llm_operations()

        container.register_token_usage(
            "test_model", {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
        )

        assert mock_counters.mock_calls == [
            mock.call(**DEFAULT_ARGS),
            mock.call().inc(10),
            mock.call(**DEFAULT_ARGS),
            mock.call().inc(15),
        ]
        assert token_usage.get() == {
            "test_model": {"input_tokens": 10, "output_tokens": 15}
        }

        container.register_token_usage(
            "test_model", {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
        )

        # It accumulates across multiple calls
        assert token_usage.get() == {
            "test_model": {"input_tokens": 15, "output_tokens": 25}
        }

        container.register_token_usage(
            "mymodel", {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        )

        # It tracks multiple models
        assert token_usage.get() == {
            "test_model": {"input_tokens": 15, "output_tokens": 25},
            "mymodel": {"input_tokens": 1, "output_tokens": 2},
        }

        # It keeps track of all individual operations
        assert llm_operations.get() == [
            {
                "token_count": 25,
                "model_id": "test_model",
                "model_engine": "test_provider",
                "model_provider": "test_provider",
                "prompt_tokens": 10,
                "completion_tokens": 15,
            },
            {
                "token_count": 15,
                "model_id": "test_model",
                "model_engine": "test_provider",
                "model_provider": "test_provider",
                "prompt_tokens": 5,
                "completion_tokens": 10,
            },
            {
                "token_count": 3,
                "model_id": "mymodel",
                "model_engine": "test_provider",
                "model_provider": "test_provider",
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
        ]

    @pytest.mark.parametrize(
        "unit_primitive",
        [GitLabUnitPrimitive.DUO_CHAT],
    )
    def test_register_token_usage_track_usage(
        self, container, unit_primitive, internal_event_client
    ):
        with capture_logs() as cap_logs:
            container.register_token_usage(
                "test_model",
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        cache_read=4,
                        cache_creation=5,
                        ephemeral_5m_input_tokens=6,
                        ephemeral_1h_input_tokens=7,
                    ),
                ),
                {"extra_key": "val"},
            )

        assert cap_logs[0] == {
            **container.labels,
            "event": "LLM call finished with token usage",
            "log_level": "info",
            "model_provider": container.model_provider,
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "cache_read": 4,
            "cache_creation": 5,
            "ephemeral_5m_input_tokens": 6,
            "ephemeral_1h_input_tokens": 7,
        }

        internal_event_client.track_event.assert_called_once_with(
            f"token_usage_{unit_primitive}",
            category="ai_gateway.instrumentators.model_requests",
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            **container.labels,
            model_provider=container.model_provider,
            additional_properties=InternalEventAdditionalProperties(
                label="cache_details",
                property=None,
                value=None,
                cache_read=4,
                cache_creation=5,
                ephemeral_5m_input_tokens=6,
                ephemeral_1h_input_tokens=7,
                extra_key="val",
            ),
        )

    def test_register_token_usage_without_init(self, container):
        container.register_token_usage(
            "test_model", {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
        )

        assert token_usage.get() is None
        assert llm_operations.get() is None

    def test_register_token_usage_in_different_contexts(self, container):
        def run():
            # Init counters
            init_token_usage()
            init_llm_operations()

            # Register a single call
            container.register_token_usage(
                "test_model",
                {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
            )

            # Assert that only a single call (ours) is registered
            assert token_usage.get() == {
                "test_model": {"input_tokens": 10, "output_tokens": 15}
            }
            assert llm_operations.get() == [
                {
                    "token_count": 25,
                    "model_id": "test_model",
                    "model_engine": "test_provider",
                    "model_provider": "test_provider",
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                }
            ]

            # Intentionally not calling `.set(None)` in the end to test that the value doesn't leak

        contextvars.copy_context().run(run)
        contextvars.copy_context().run(run)

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    @pytest.mark.parametrize(
        "response_metadata_list,expected_stop_reason",
        [
            ([], "unknown"),  # Default value when `register_message` is never called
            (
                [{}],
                "unknown",
            ),  # Default value when `register_message` is called without model metadata
            ([{"finish_reason": "length"}], "length"),
            ([{"stop_reason": "tool_calls"}], "tool_calls"),
            ([{"finish_reason": "unexpected"}], "other"),
            # Check that the finish reason is retained through multiple `register_message` calls, even if the message
            # with `finish_reason` is not necessarily the last one (which can happen in real world usage)
            ([{}, {"finish_reason": "length"}, {}], "length"),
        ],
    )
    def test_finish(
        self,
        time_counter,
        mock_histograms,
        mock_counters,
        mock_gauges,
        response_metadata_list,
        expected_stop_reason,
        container,
    ):
        time_counter.side_effect = [1, 2]

        container.start()
        mock_gauges.reset_mock()  # So we only have the calls from `stop` below

        for response_metadata in response_metadata_list:
            container.register_message(
                AIMessage(content="", response_metadata=response_metadata)
            )

        with capture_logs() as cap_logs:
            container.finish()

        assert len(cap_logs) == 1
        assert cap_logs[0]["event"] == "Request to LLM complete"
        assert cap_logs[0]["duration"] == 1

        assert mock_gauges.mock_calls == [
            mock.call(model_engine="test_engine", model_name="test_model"),
            mock.call().dec(),
        ]

        expected_call = mock.call(
            **{**DEFAULT_ARGS, "finish_reason": expected_stop_reason}
        )

        assert mock_counters.mock_calls == [expected_call, mock.call().inc()]
        assert mock_histograms.mock_calls == [expected_call, mock.call().observe(1)]


class TestModelRequestInstrumentator:
    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_sync(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]

        instrumentator = ModelRequestInstrumentator(
            model_engine="test_engine",
            model_name="test_model",
            limits=None,
        )
        with instrumentator.watch():
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="test_engine", model_name="test_model"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

        assert mock_counters.mock_calls == [
            mock.call(**DEFAULT_ARGS),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(**DEFAULT_ARGS),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_sync_with_error(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]

        instrumentator = ModelRequestInstrumentator(
            model_engine="test_engine",
            model_name="test_model",
            limits=None,
        )

        with pytest.raises(ValueError):
            with instrumentator.watch():
                assert mock_gauges.mock_calls == [
                    mock.call(model_engine="test_engine", model_name="test_model"),
                    mock.call().inc(),
                ]

                mock_gauges.reset_mock()

                raise ValueError("broken")

        assert mock_gauges.mock_calls == [
            mock.call(model_engine="test_engine", model_name="test_model"),
            mock.call().dec(),
        ]
        assert mock_counters.mock_calls == [
            mock.call(**{**DEFAULT_ARGS, "error": "yes", "error_type": "other"}),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(**{**DEFAULT_ARGS, "error": "yes", "error_type": "other"}),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_limits(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="test_engine",
            model_name="test_model",
            limits={"input_tokens": 5, "output_tokens": 10, "concurrency": 15},
        )

        with instrumentator.watch():
            mock_gauges.assert_has_calls(
                [
                    mock.call(model_engine="test_engine", model_name="test_model"),
                    mock.call().set(15),
                    mock.call(model_engine="test_engine", model_name="test_model"),
                    mock.call().set(5),
                    mock.call(model_engine="test_engine", model_name="test_model"),
                    mock.call().set(10),
                ]
            )

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_partial_limits(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="test_engine",
            model_name="test_model",
            limits={"concurrency": 15},
        )

        with instrumentator.watch():
            mock_gauges.assert_has_calls(
                [
                    mock.call(model_engine="test_engine", model_name="test_model"),
                    mock.call().set(15),
                ]
            )

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_watch_async(
        self, time_counter, mock_histograms, mock_counters, mock_gauges
    ):
        time_counter.side_effect = [1, 2]
        instrumentator = ModelRequestInstrumentator(
            model_engine="test_engine",
            model_name="test_model",
            limits=None,
        )

        with instrumentator.watch(stream=True) as watcher:
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="test_engine", model_name="test_model"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

            watcher.finish()

            assert mock_gauges.mock_calls == [
                mock.call(model_engine="test_engine", model_name="test_model"),
                mock.call().dec(),
            ]
            assert mock_counters.mock_calls == [
                mock.call(**{**DEFAULT_ARGS, "streaming": "yes"}),
                mock.call().inc(),
            ]
            assert mock_histograms.mock_calls == [
                mock.call(**{**DEFAULT_ARGS, "streaming": "yes"}),
                mock.call().observe(1),
            ]


@pytest.fixture(name="instrumentator")
def instrumentator_fixture():

    return ModelRequestInstrumentator(
        model_engine="test_engine",
        model_name="test_model",
        limits=None,
    )


class TestDetailLabels:
    @pytest.mark.parametrize(
        "unit_primitive,expected_unit_primitive,expected_feature_category",
        [
            (GitLabUnitPrimitive.SUMMARIZE_COMMENTS, "summarize_comments", "unknown"),
            (None, "unknown", "unknown"),
        ],
    )
    def test_detail_labels(
        self,
        instrumentator,
        unit_primitive,
        expected_unit_primitive,
        expected_feature_category,
    ):
        with instrumentator.watch(unit_primitive=unit_primitive) as watcher:
            labels = watcher._detail_labels()
            assert labels["unit_primitive"] == expected_unit_primitive
            assert labels["feature_category"] == expected_feature_category

    def test_detail_labels_without_unit_primitive(self, instrumentator):
        with instrumentator.watch(unit_primitive=None) as watcher:
            labels = watcher._detail_labels()
            assert labels["unit_primitive"] == "unknown"


class TestRegisterError:
    def test_register_error_no_exception(self, container):
        container.register_error(None)
        assert container.error is True
        assert container.error_type == "other"

    def test_register_error_prompt_too_long_lowercase(self, container):
        class CustomError(Exception):
            pass

        exception = CustomError("prompt is too long: 189294 tokens > 180000 maximum")
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "prompt_too_long"

    def test_register_error_prompt_too_long_capitalized(self, container):
        class CustomError(Exception):
            pass

        exception = CustomError("Prompt is too long")
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "prompt_too_long"

    def test_register_error_status_code_400(self, container):
        class BadRequestError(Exception):
            def __init__(self):
                self.status_code = 400

        exception = BadRequestError()
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "http_400"

    def test_register_error_status_code_403(self, container):
        class PermissionError(Exception):
            def __init__(self):
                self.status_code = 403

        exception = PermissionError()
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "permission_error"

    def test_register_error_code_attribute_403(self, container):
        """Test that code attribute is checked (used by LiteLLM)"""

        class LiteLLMError(Exception):
            def __init__(self):
                self.code = 403

        exception = LiteLLMError()
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "permission_error"

    def test_register_error_status_code_unknown(self, container):
        class UnknownError(Exception):
            def __init__(self):
                self.status_code = 418  # I'm a teapot

        exception = UnknownError()
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "other"

    def test_register_error_no_status_code(self, container):
        class GenericError(Exception):
            pass

        exception = GenericError("Some error")
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "other"

    def test_register_error_prompt_too_long_takes_precedence(self, container):
        """Test that prompt_too_long check takes precedence over status_code."""

        class BadRequestError(Exception):
            def __init__(self):
                self.status_code = 400

        exception = BadRequestError()
        exception.args = ("prompt is too long: 189294 tokens > 180000 maximum",)
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "prompt_too_long"

    def test_register_error_overloaded_message(self, container):
        """Test that 'Overloaded' in message is caught."""

        class LiteLLMError(Exception):
            pass

        exception = LiteLLMError("Overloaded")
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "overloaded"

    def test_register_error_status_code_503(self, container):
        class ServiceUnavailableError(Exception):
            def __init__(self):
                self.status_code = 503

        exception = ServiceUnavailableError()
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "service_unavailable"

    def test_register_error_overloaded_message_takes_precedence(self, container):
        """Test that overloaded message check takes precedence over status_code."""

        class LiteLLMError(Exception):
            def __init__(self):
                self.status_code = 400

        exception = LiteLLMError()
        exception.args = ("Overloaded",)
        container.register_error(exception)
        assert container.error is True
        assert container.error_type == "overloaded"
