import json
import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from duo_workflow_service.audit_events.client import AuditEventClient
from duo_workflow_service.audit_events.event_types import ToolInvokedEvent
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse


def _parse_payload(http_client):
    return json.loads(http_client.apost.call_args.kwargs["body"])


@pytest.fixture(name="http_client")
def http_client_fixture():
    return AsyncMock(spec=GitlabHttpClient)


@pytest.fixture(name="events")
def events_fixture():
    return [
        ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file"),
        ToolInvokedEvent(workflow_id="wf-1", tool_name="write_file"),
    ]


def _make_client(http_client, max_retries=3):
    return AuditEventClient(
        http_client=http_client,
        workflow_id="wf-1",
        max_retries=max_retries,
        base_delay=0.01,
    )


@patch.object(
    AuditEventClient, "_is_supported", new_callable=PropertyMock, return_value=True
)
class TestSendBatch:
    @pytest.mark.asyncio
    async def test_success(self, _mock_supported, http_client, events):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        result = await client.send_batch(events)
        assert result is True
        http_client.apost.assert_called_once()
        call_kwargs = http_client.apost.call_args
        assert (
            call_kwargs.kwargs["path"]
            == "/api/v4/ai/duo_workflows/workflows/wf-1/audit_events"
        )

    @pytest.mark.asyncio
    async def test_empty_events_returns_true(self, _mock_supported, http_client):
        client = _make_client(http_client)
        result = await client.send_batch([])
        assert result is True
        http_client.apost.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_raw_json_payload(self, _mock_supported, http_client, events):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events)
        call_kwargs = http_client.apost.call_args
        assert call_kwargs.kwargs["parse_json"] is False

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, _mock_supported, http_client, events):
        http_client.apost.return_value = GitLabHttpResponse(
            status_code=400, body="Bad Request"
        )
        client = _make_client(http_client)
        result = await client.send_batch(events)
        assert result is False
        assert http_client.apost.call_count == 1


class TestSendBatchUnsupported:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "gitlab_version_str",
        ["17.0.0", None],
        ids=["version_too_old", "version_unparseable"],
    )
    async def test_unsupported_skips_send(
        self, http_client, events, gitlab_version_str
    ):
        with patch(
            "duo_workflow_service.audit_events.client.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = gitlab_version_str
            client = _make_client(http_client)
            result = await client.send_batch(events)

        assert result is True
        http_client.apost.assert_not_called()


@patch.object(
    AuditEventClient, "_is_supported", new_callable=PropertyMock, return_value=True
)
class TestRetryBehavior:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [429, 500, 503, 529])
    async def test_retries_on_retryable_status(
        self, _mock_supported, http_client, events, status_code
    ):
        http_client.apost.side_effect = [
            GitLabHttpResponse(status_code=status_code, body=""),
            GitLabHttpResponse(status_code=200, body=""),
        ]
        client = _make_client(http_client, max_retries=3)
        result = await client.send_batch(events)
        assert result is True
        assert http_client.apost.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, _mock_supported, http_client, events):
        http_client.apost.return_value = GitLabHttpResponse(
            status_code=500, body="Server Error"
        )
        client = _make_client(http_client, max_retries=4)
        result = await client.send_batch(events)
        assert result is False
        assert http_client.apost.call_count == 4

    @pytest.mark.asyncio
    async def test_retries_on_exception(self, _mock_supported, http_client, events):
        http_client.apost.side_effect = [
            ConnectionError("connection refused"),
            GitLabHttpResponse(status_code=200, body=""),
        ]
        client = _make_client(http_client, max_retries=3)
        result = await client.send_batch(events)
        assert result is True
        assert http_client.apost.call_count == 2

    @pytest.mark.asyncio
    async def test_exception_retries_exhausted(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.side_effect = ConnectionError("connection refused")
        client = _make_client(http_client, max_retries=2)
        result = await client.send_batch(events)
        assert result is False
        assert http_client.apost.call_count == 2


@patch.object(
    AuditEventClient, "_is_supported", new_callable=PropertyMock, return_value=True
)
class TestBatchIdentity:
    @pytest.mark.asyncio
    async def test_batch_id_present_in_payload(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events)
        payload = _parse_payload(http_client)
        assert "batch_id" in payload

    @pytest.mark.asyncio
    async def test_batch_id_is_uuid(self, _mock_supported, http_client, events):

        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events)
        payload = _parse_payload(http_client)
        uuid_mod.UUID(payload["batch_id"])  # raises if not a valid UUID

    @pytest.mark.asyncio
    async def test_same_batch_id_across_retries(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.side_effect = [
            GitLabHttpResponse(status_code=500, body=""),
            GitLabHttpResponse(status_code=200, body=""),
        ]
        client = _make_client(http_client, max_retries=3)
        await client.send_batch(events)
        calls = http_client.apost.call_args_list
        batch_ids = [json.loads(c.kwargs["body"])["batch_id"] for c in calls]
        assert batch_ids[0] == batch_ids[1]

    @pytest.mark.asyncio
    async def test_different_batch_id_per_send_batch_call(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events)
        first_id = _parse_payload(http_client)["batch_id"]
        await client.send_batch(events)
        second_id = json.loads(http_client.apost.call_args_list[1].kwargs["body"])[
            "batch_id"
        ]
        assert first_id != second_id


@patch.object(
    AuditEventClient, "_is_supported", new_callable=PropertyMock, return_value=True
)
class TestFinalBatch:
    @pytest.mark.asyncio
    async def test_final_fields_present_when_is_final(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events, is_final=True, total_events_sent=5)
        payload = _parse_payload(http_client)
        assert payload["final"] is True
        assert payload["total_events_sent"] == 5

    @pytest.mark.asyncio
    async def test_final_fields_absent_by_default(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        await client.send_batch(events)
        payload = _parse_payload(http_client)
        assert "final" not in payload
        assert "total_events_sent" not in payload

    @pytest.mark.asyncio
    async def test_empty_events_with_is_final_sends_request(
        self, _mock_supported, http_client
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        client = _make_client(http_client)
        result = await client.send_batch([], is_final=True, total_events_sent=10)
        assert result is True
        http_client.apost.assert_called_once()
        payload = _parse_payload(http_client)
        assert payload["events"] == []
        assert payload["final"] is True
        assert payload["total_events_sent"] == 10

    @pytest.mark.asyncio
    async def test_empty_events_without_is_final_skips_request(
        self, _mock_supported, http_client
    ):
        client = _make_client(http_client)
        result = await client.send_batch([])
        assert result is True
        http_client.apost.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_send_failure_is_non_recoverable(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=500, body="")
        client = _make_client(http_client, max_retries=2)
        result = await client.send_batch(events, is_final=True, total_events_sent=2)
        assert result is False
        assert http_client.apost.call_count == 2

    @pytest.mark.asyncio
    async def test_unsupported_skips_final_send(
        self, _mock_supported, http_client, events
    ):
        _mock_supported.return_value = False
        client = _make_client(http_client)
        result = await client.send_batch(events, is_final=True, total_events_sent=2)
        assert result is True
        http_client.apost.assert_not_called()


@patch.object(
    AuditEventClient, "_is_supported", new_callable=PropertyMock, return_value=True
)
class TestSendBatchMetrics:
    @pytest.mark.asyncio
    async def test_success_increments_sent_counter_per_event(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client)
            await client.send_batch(events)

        mock_metrics.count_audit_events_sent.assert_called_once_with(
            result="success", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_http_error_increments_sent_and_dropped_per_event(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(
            status_code=400, body="Bad Request"
        )
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client)
            await client.send_batch(events)

        mock_metrics.count_audit_events_sent.assert_called_once_with(
            result="http_error", amount=len(events)
        )
        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="http_error", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_http_retries_exhausted_drops_with_retries_exhausted_reason(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(
            status_code=500, body="Server Error"
        )
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client, max_retries=2)
            await client.send_batch(events)

        mock_metrics.count_audit_events_sent.assert_called_once_with(
            result="http_error", amount=len(events)
        )
        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="retries_exhausted", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_exception_exhausted_increments_sent_counter_per_event(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.side_effect = ConnectionError("boom")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client, max_retries=1)
            await client.send_batch(events)

        mock_metrics.count_audit_events_sent.assert_called_once_with(
            result="exception", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_retries_exhausted_increments_dropped_counter_per_event(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.side_effect = ConnectionError("boom")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client, max_retries=2)
            await client.send_batch(events)

        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="retries_exhausted", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_observes_batch_size(self, _mock_supported, http_client, events):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client)
            await client.send_batch(events)

        mock_metrics.observe_audit_events_batch_size.assert_called_once_with(
            len(events)
        )

    @pytest.mark.asyncio
    async def test_observes_payload_bytes_utf8(
        self, _mock_supported, http_client, events
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client)
            await client.send_batch(events)

        call_args = mock_metrics.observe_audit_events_payload_bytes.call_args
        assert call_args is not None
        observed_bytes = call_args.args[0]
        assert observed_bytes > 0

    @pytest.mark.asyncio
    async def test_empty_final_batch_skips_size_and_bytes_observation(
        self, _mock_supported, http_client
    ):
        http_client.apost.return_value = GitLabHttpResponse(status_code=200, body="")
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.client.duo_workflow_metrics",
            mock_metrics,
        ):
            client = _make_client(http_client)
            await client.send_batch([], is_final=True, total_events_sent=0)

        mock_metrics.observe_audit_events_batch_size.assert_not_called()
        mock_metrics.observe_audit_events_payload_bytes.assert_not_called()


class TestSendBatchMetricsUnsupported:
    @pytest.mark.asyncio
    async def test_version_unsupported_increments_dropped_counter_per_event(
        self, http_client, events
    ):
        mock_metrics = MagicMock()
        with (
            patch(
                "duo_workflow_service.audit_events.client.duo_workflow_metrics",
                mock_metrics,
            ),
            patch(
                "duo_workflow_service.audit_events.client.gitlab_version"
            ) as mock_version,
        ):
            mock_version.get.return_value = "17.0.0"
            client = _make_client(http_client)
            await client.send_batch(events)

        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="version_unsupported", amount=len(events)
        )

    @pytest.mark.asyncio
    async def test_version_unsupported_no_drop_for_empty_events(self, http_client):
        mock_metrics = MagicMock()
        with (
            patch(
                "duo_workflow_service.audit_events.client.duo_workflow_metrics",
                mock_metrics,
            ),
            patch(
                "duo_workflow_service.audit_events.client.gitlab_version"
            ) as mock_version,
        ):
            mock_version.get.return_value = "17.0.0"
            client = _make_client(http_client)
            await client.send_batch([])

        mock_metrics.count_audit_events_dropped.assert_not_called()
