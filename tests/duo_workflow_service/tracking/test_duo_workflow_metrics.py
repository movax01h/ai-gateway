import time
import unittest
from typing import Any, cast
from unittest.mock import MagicMock

from duo_workflow_service.tracking.duo_workflow_metrics import DuoWorkflowMetrics


class TestDuoWorkflowMetrics(unittest.TestCase):
    def setUp(self):
        self.mock_registry = MagicMock()
        self.metrics = DuoWorkflowMetrics(registry=self.mock_registry)

        self._setup_histogram_mocks()

    def _setup_histogram_mocks(self):
        """Set up properly typed mocks for all histograms."""
        for metric_name in [
            "workflow_duration",
            "llm_request_duration",
            "tool_call_duration",
            "compute_duration",
            "gitlab_response_duration",
            "network_latency",
            "llm_response_counter",
        ]:
            mock_histogram = MagicMock()
            setattr(self.metrics, metric_name, mock_histogram)
            mock_histogram.labels = cast(Any, MagicMock())

    def test_timer_callback_with_zero(self):
        mock_callback = MagicMock()
        timer = self.metrics._timer(mock_callback)
        timer.start_time = None
        timer.__exit__(None, None, None)
        mock_callback.assert_called_once_with(0.0)

    def test_timer_without_start(self):
        mock_callback = MagicMock()
        timer = self.metrics._timer(mock_callback)
        timer.start_time = None
        timer.__exit__(None, None, None)
        mock_callback.assert_called_once_with(0.0)

    def test_timer_measures_duration(self):
        mock_callback = MagicMock()
        timer = self.metrics._timer(mock_callback)

        with timer:
            time.sleep(0.01)

        self.assertEqual(mock_callback.call_count, 1)
        args, _ = mock_callback.call_args
        self.assertGreater(args[0], 0)

    def test_time_llm_request(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.llm_request_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_llm_request(
            model="test_model", request_type="test_request"
        ):
            pass

        cast(
            MagicMock, self.metrics.llm_request_duration.labels
        ).assert_called_once_with(model="test_model", request_type="test_request")
        observe_mock.assert_called_once()

    def test_time_tool_call(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.tool_call_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_tool_call(tool_name="test_tool"):
            pass

        cast(MagicMock, self.metrics.tool_call_duration.labels).assert_called_once_with(
            tool_name="test_tool"
        )
        observe_mock.assert_called_once()

    def test_time_compute(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.compute_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_compute(operation_type="test_operation"):
            pass

        cast(MagicMock, self.metrics.compute_duration.labels).assert_called_once_with(
            operation_type="test_operation"
        )
        observe_mock.assert_called_once()

    def test_time_gitlab_response(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.gitlab_response_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_gitlab_response(endpoint="test_endpoint", method="GET"):
            pass

        cast(
            MagicMock, self.metrics.gitlab_response_duration.labels
        ).assert_called_once_with(endpoint="test_endpoint", method="GET")
        observe_mock.assert_called_once()

    def test_time_network_latency(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.network_latency.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_network_latency(source="service", destination="gitlab"):
            pass

        cast(MagicMock, self.metrics.network_latency.labels).assert_called_once_with(
            source="service", destination="gitlab"
        )
        observe_mock.assert_called_once()

    def test_time_workflow(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.workflow_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_workflow(workflow_type="test_workflow"):
            pass

        cast(MagicMock, self.metrics.workflow_duration.labels).assert_called_once_with(
            workflow_type="test_workflow"
        )
        observe_mock.assert_called_once()

    def test_llm_response_counter(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.inc = observe_mock

        cast(MagicMock, self.metrics.llm_response_counter.labels).return_value = (
            labels_result_mock
        )

        self.metrics.count_llm_response(
            model="test_model",
            request_type="test_request",
            stop_reason="test_reason",
        )

        cast(
            MagicMock, self.metrics.llm_response_counter.labels
        ).assert_called_once_with(
            model="test_model", request_type="test_request", stop_reason="other"
        )


if __name__ == "__main__":
    unittest.main()
