from unittest import mock

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from structlog.testing import capture_logs

from ai_gateway.api.feature_category import current_feature_category
from ai_gateway.instrumentators.model_requests import (
    ModelRequestInstrumentator,
    get_token_usage,
    token_usage,
)


def test_get_token_usage():
    usage = {"claude": {"input_tokens": 10, "output_tokens": 20}}
    token_usage.set(usage)
    assert get_token_usage() == usage
    assert token_usage.get() is None  # Ensure the usage is reset after being retrieved


class TestWatchContainer:
    @mock.patch("prometheus_client.Counter.labels")
    def test_register_token_usage(self, mock_counters):
        container = ModelRequestInstrumentator.WatchContainer(
            labels={"model_engine": "anthropic", "model_name": "claude"},
            streaming=False,
            limits=None,
            unit_primitives=None,
        )

        container.register_token_usage(
            "claude", {"input_tokens": 10, "output_tokens": 15}
        )

        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().inc(10),
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().inc(15),
        ]
        assert token_usage.get() == {
            "claude": {"input_tokens": 10, "output_tokens": 15}
        }

        container.register_token_usage(
            "claude", {"input_tokens": 5, "output_tokens": 10}
        )

        # It accumulates across multiple calls
        assert token_usage.get() == {
            "claude": {"input_tokens": 15, "output_tokens": 25}
        }

        container.register_token_usage(
            "mymodel", {"input_tokens": 1, "output_tokens": 2}
        )

        # It tracks multiple models
        assert token_usage.get() == {
            "claude": {"input_tokens": 15, "output_tokens": 25},
            "mymodel": {"input_tokens": 1, "output_tokens": 2},
        }

    @mock.patch("prometheus_client.Gauge.labels")
    @mock.patch("prometheus_client.Counter.labels")
    @mock.patch("prometheus_client.Histogram.labels")
    @mock.patch("time.perf_counter")
    def test_finish(
        self,
        time_counter,
        mock_histograms,
        mock_counters,
        mock_gauges,
    ):
        container = ModelRequestInstrumentator.WatchContainer(
            labels={"model_engine": "anthropic", "model_name": "claude"},
            streaming=False,
            limits=None,
            unit_primitives=None,
        )
        time_counter.side_effect = [1, 2]

        container.start()
        mock_gauges.reset_mock()  # So we only have the calls from `stop` below

        with capture_logs() as cap_logs:
            container.finish()

        assert len(cap_logs) == 1
        assert cap_logs[0]["event"] == "Request to LLM complete"
        assert cap_logs[0]["duration"] == 1

        assert mock_gauges.mock_calls == [
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ]

        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().observe(1),
        ]


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
            model_engine="anthropic", model_name="claude", limits=None
        )
        with instrumentator.watch():
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
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
            model_engine="anthropic", model_name="claude", limits=None
        )

        with pytest.raises(ValueError):
            with instrumentator.watch():
                assert mock_gauges.mock_calls == [
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().inc(),
                ]

                mock_gauges.reset_mock()

                raise ValueError("broken")

        assert mock_gauges.mock_calls == [
            mock.call(model_engine="anthropic", model_name="claude"),
            mock.call().dec(),
        ]
        assert mock_counters.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="yes",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().inc(),
        ]
        assert mock_histograms.mock_calls == [
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="yes",
                streaming="no",
                feature_category="unknown",
                unit_primitive="unknown",
            ),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_limits(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic",
            model_name="claude",
            limits={"input_tokens": 5, "output_tokens": 10, "concurrency": 15},
        )

        with instrumentator.watch():
            mock_gauges.assert_has_calls(
                [
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().set(15),
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().set(5),
                    mock.call(model_engine="anthropic", model_name="claude"),
                    mock.call().set(10),
                ]
            )

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_partial_limits(self, mock_gauges):
        instrumentator = ModelRequestInstrumentator(
            model_engine="anthropic",
            model_name="claude",
            limits={"concurrency": 15},
        )

        with instrumentator.watch():
            mock_gauges.assert_has_calls(
                [
                    mock.call(model_engine="anthropic", model_name="claude"),
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
            model_engine="anthropic", model_name="claude", limits=None
        )

        with instrumentator.watch(stream=True) as watcher:
            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().inc(),
            ]

            mock_gauges.reset_mock()

            watcher.finish()

            assert mock_gauges.mock_calls == [
                mock.call(model_engine="anthropic", model_name="claude"),
                mock.call().dec(),
            ]
            assert mock_counters.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="yes",
                    feature_category="unknown",
                    unit_primitive="unknown",
                ),
                mock.call().inc(),
            ]
            assert mock_histograms.mock_calls == [
                mock.call(
                    model_engine="anthropic",
                    model_name="claude",
                    error="no",
                    streaming="yes",
                    feature_category="unknown",
                    unit_primitive="unknown",
                ),
                mock.call().observe(1),
            ]


@pytest.fixture
def instrumentator():

    return ModelRequestInstrumentator(
        model_engine="test_engine", model_name="test_model", limits=None
    )


class TestDetailLabels:
    @pytest.mark.parametrize(
        "unit_primitives,expected_unit_primitive,expected_feature_category",
        [
            ([GitLabUnitPrimitive.SUMMARIZE_COMMENTS], "summarize_comments", "unknown"),
            (
                [GitLabUnitPrimitive.DUO_CHAT, GitLabUnitPrimitive.CODE_SUGGESTIONS],
                "duo_chat",
                "unknown",
            ),
            (None, "unknown", "unknown"),
        ],
    )
    def test_detail_labels(
        self,
        instrumentator,
        unit_primitives,
        expected_unit_primitive,
        expected_feature_category,
    ):
        with instrumentator.watch(unit_primitives=unit_primitives) as watcher:
            labels = watcher._detail_labels()
            assert labels["unit_primitive"] == expected_unit_primitive
            assert labels["feature_category"] == expected_feature_category
            GitLabUnitPrimitive.CODE_SUGGESTIONS,

    def test_detail_labels_without_unit_primitive(self, instrumentator):

        with instrumentator.watch(unit_primitives=None) as watcher:
            labels = watcher._detail_labels()
            assert labels["unit_primitive"] == "unknown"
