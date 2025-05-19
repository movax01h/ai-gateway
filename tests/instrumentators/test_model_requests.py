from unittest import mock

import pytest
from structlog.testing import capture_logs

from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator


class TestWatchContainer:
    @mock.patch("prometheus_client.Counter.labels")
    def test_register_token_usage(self, mock_counters):
        container = ModelRequestInstrumentator.WatchContainer(
            labels={"model_engine": "anthropic", "model_name": "claude"},
            streaming=False,
            limits=None,
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
            ),
            mock.call().inc(10),
            mock.call(
                model_engine="anthropic",
                model_name="claude",
                error="no",
                streaming="no",
                feature_category="unknown",
            ),
            mock.call().inc(15),
        ]

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
            ),
            mock.call().observe(1),
        ]

    @mock.patch("prometheus_client.Gauge.labels")
    def test_watch_with_limit(self, mock_gauges):
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
                ),
                mock.call().observe(1),
            ]
