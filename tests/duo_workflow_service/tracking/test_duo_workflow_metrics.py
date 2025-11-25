import time
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

from ai_gateway.code_suggestions import LanguageServerVersion
from duo_workflow_service.tracking.duo_workflow_metrics import (
    DuoWorkflowMetrics,
    SessionTypeEnum,
)


class TestDuoWorkflowMetrics(unittest.TestCase):
    def setUp(self):
        self.mock_registry = MagicMock()
        self.metrics = DuoWorkflowMetrics(registry=self.mock_registry)

        self._setup_metric_mocks()

    def _setup_metric_mocks(self):
        """Set up properly typed mocks for all histograms."""
        for metric_name in [
            "workflow_duration",
            "tool_call_duration",
            "compute_duration",
            "gitlab_response_duration",
            "network_latency",
            "checkpoint_counter",
            "agent_platform_session_start_counter",
            "agent_platform_session_success_counter",
            "agent_platform_session_failure_counter",
            "agent_platform_session_abort_counter",
            "agent_platform_tool_failure_counter",
            "agent_platform_receive_start_counter",
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

    def test_time_tool_call(self):
        observe_mock = MagicMock()
        labels_result_mock = MagicMock()
        labels_result_mock.observe = observe_mock

        cast(MagicMock, self.metrics.tool_call_duration.labels).return_value = (
            labels_result_mock
        )

        with self.metrics.time_tool_call(tool_name="test_tool", flow_type="test_flow"):
            pass

        cast(MagicMock, self.metrics.tool_call_duration.labels).assert_called_once_with(
            tool_name="test_tool",
            flow_type="test_flow",
            lsp_version="unknown",
            gitlab_version="unknown",
            client_type="unknown",
            gitlab_realm="unknown",
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

    def _assert_counter_called(
        self,
        counter_name: str,
        method_name: str,
        expected_labels: dict,
        *args,
        **kwargs,
    ):
        """Asserts counter is called with labels."""
        counter = getattr(self.metrics, counter_name)
        method = getattr(self.metrics, method_name)

        labels_result_mock = MagicMock()
        cast(MagicMock, counter.labels).return_value = labels_result_mock

        method(*args, **kwargs)

        cast(MagicMock, counter.labels).assert_called_once_with(**expected_labels)

    def test_checkpoint_counter(self):
        self._assert_counter_called(
            "checkpoint_counter",
            "count_checkpoints",
            {
                "endpoint": "test_endpoint",
                "status_code": "test_status",
                "method": "POST",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            endpoint="test_endpoint",
            status_code="test_status",
            method="POST",
        )

    def test_agent_platform_session_start_counter(self):
        self._assert_counter_called(
            "agent_platform_session_start_counter",
            "count_agent_platform_session_start",
            {
                "flow_type": "test_flow_type",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
        )

    def test_agent_platform_session_success_counter(self):
        self._assert_counter_called(
            "agent_platform_session_success_counter",
            "count_agent_platform_session_success",
            {
                "flow_type": "test_flow_type",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
        )

    @patch("duo_workflow_service.tracking.duo_workflow_metrics.session_type_context")
    def test_agent_platform_session_failure_counter(self, mock_session_context):
        mock_session_context.get.return_value = SessionTypeEnum.START.value
        self._assert_counter_called(
            "agent_platform_session_failure_counter",
            "count_agent_platform_session_failure",
            {
                "flow_type": "test_flow_type",
                "failure_reason": "model_error",
                "session_type": "start",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
            failure_reason="model_error",
        )

    @patch("duo_workflow_service.tracking.duo_workflow_metrics.session_type_context")
    def test_agent_platform_session_abort_counter(self, mock_session_context):
        mock_session_context.get.return_value = SessionTypeEnum.START.value
        self._assert_counter_called(
            "agent_platform_session_abort_counter",
            "count_agent_platform_session_abort",
            {
                "flow_type": "test_flow_type",
                "session_type": "start",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
        )

    def test_agent_platform_tool_failure_counter(self):
        self._assert_counter_called(
            "agent_platform_tool_failure_counter",
            "count_agent_platform_tool_failure",
            {
                "flow_type": "test_flow_type",
                "tool_name": "test_tool",
                "failure_reason": "test_error",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
            tool_name="test_tool",
            failure_reason="test_error",
        )

    @patch("ai_gateway.instrumentators.model_requests.language_server_version")
    def test_agent_platform_tool_failure_counter_with_lsp_version(
        self, mock_language_server_version
    ):
        mock_language_server_version.get.return_value = (
            LanguageServerVersion.from_string("8.22.0")
        )
        self._assert_counter_called(
            "agent_platform_tool_failure_counter",
            "count_agent_platform_tool_failure",
            {
                "flow_type": "test_flow_type",
                "tool_name": "test_tool",
                "failure_reason": "test_error",
                "lsp_version": "8.22.0",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
            tool_name="test_tool",
            failure_reason="test_error",
        )

    @patch("ai_gateway.instrumentators.model_requests.gitlab_version")
    def test_agent_platform_tool_failure_counter_with_gitlab_version(
        self, mock_gitlab_version
    ):
        mock_gitlab_version.get.return_value = "18.3.0"

        self._assert_counter_called(
            "agent_platform_tool_failure_counter",
            "count_agent_platform_tool_failure",
            {
                "flow_type": "test_flow_type",
                "tool_name": "test_tool",
                "failure_reason": "test_error",
                "lsp_version": "unknown",
                "gitlab_version": "18.3.0",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
            tool_name="test_tool",
            failure_reason="test_error",
        )

    @patch("ai_gateway.instrumentators.model_requests.client_type")
    def test_agent_platform_tool_failure_counter_with_client_type(
        self, mock_client_type
    ):
        mock_client_type.get.return_value = "node-grpc"
        self._assert_counter_called(
            "agent_platform_tool_failure_counter",
            "count_agent_platform_tool_failure",
            {
                "flow_type": "test_flow_type",
                "tool_name": "test_tool",
                "failure_reason": "test_error",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "node-grpc",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
            tool_name="test_tool",
            failure_reason="test_error",
        )

    @patch("ai_gateway.instrumentators.model_requests.gitlab_realm")
    def test_agent_platform_tool_failure_counter_with_gitlab_realm(
        self, mock_gitlab_realm
    ):
        mock_gitlab_realm.get.return_value = "saas"

        self._assert_counter_called(
            "agent_platform_tool_failure_counter",
            "count_agent_platform_tool_failure",
            {
                "flow_type": "test_flow_type",
                "tool_name": "test_tool",
                "failure_reason": "test_error",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "saas",
            },
            flow_type="test_flow_type",
            tool_name="test_tool",
            failure_reason="test_error",
        )

    def test_agent_platform_receive_start_counter(self):
        self._assert_counter_called(
            "agent_platform_receive_start_counter",
            "count_agent_platform_receive_start_counter",
            {
                "flow_type": "test_flow_type",
                "lsp_version": "unknown",
                "gitlab_version": "unknown",
                "client_type": "unknown",
                "gitlab_realm": "unknown",
            },
            flow_type="test_flow_type",
        )


if __name__ == "__main__":
    unittest.main()
