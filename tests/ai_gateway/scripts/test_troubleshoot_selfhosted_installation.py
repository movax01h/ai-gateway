"""Tests for the self-hosted installation troubleshoot checks."""

import sys
from unittest.mock import MagicMock

import grpc
import pytest
import requests
from grpc_health.v1 import health_pb2

from ai_gateway.scripts import troubleshoot_selfhosted_installation as ts


class _FakeRpcError(grpc.RpcError):
    def __init__(self, details, code=grpc.StatusCode.UNAVAILABLE):
        self._details = details
        self._code = code

    def details(self):
        return self._details

    def code(self):
        return self._code


class TestCheckCustomerPortalReachable:
    def test_reachable(self, monkeypatch, capsys):
        monkeypatch.setattr(ts.requests, "get", lambda *a, **k: MagicMock())

        ts.check_customer_portal_reachable()

        assert "Customer Portal is reachable ✔" in capsys.readouterr().out

    def test_non_2xx_reports_failure(self, monkeypatch, capsys):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("404")
        monkeypatch.setattr(ts.requests, "get", lambda *a, **k: response)

        ts.check_customer_portal_reachable()

        assert "Could not reach the Customer Portal" in capsys.readouterr().out

    def test_connection_error_reports_failure(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(ts.requests, "get", boom)

        ts.check_customer_portal_reachable()

        assert "Could not reach the Customer Portal" in capsys.readouterr().out


class TestDwsChannel:
    def test_insecure_channel(self, monkeypatch):
        monkeypatch.delenv("DUO_WORKFLOW_TLS__CERT_FILE", raising=False)

        channel = ts._dws_channel("localhost:50052", secure=False)

        assert channel is not None
        channel.close()

    def test_secure_channel_default_roots(self, monkeypatch):
        monkeypatch.delenv("DUO_WORKFLOW_TLS__CERT_FILE", raising=False)

        channel = ts._dws_channel("localhost:50052", secure=True)

        assert channel is not None
        channel.close()

    def test_secure_channel_with_cert_file(self, monkeypatch, tmp_path):
        cert = tmp_path / "cert.pem"
        cert.write_bytes(b"dummy-cert")
        monkeypatch.setenv("DUO_WORKFLOW_TLS__CERT_FILE", str(cert))

        channel = ts._dws_channel("localhost:50052", secure=True)

        assert channel is not None
        channel.close()

    def test_unreadable_cert_file_bails(self, monkeypatch, capsys):
        monkeypatch.setenv("DUO_WORKFLOW_TLS__CERT_FILE", "/nope/missing-cert.pem")

        channel = ts._dws_channel("localhost:50052", secure=True)

        assert channel is None
        assert "Could not read DUO_WORKFLOW_TLS__CERT_FILE" in capsys.readouterr().out


class TestCheckDwsHealth:
    @pytest.fixture
    def stub(self, monkeypatch):
        stub = MagicMock()
        monkeypatch.setattr(
            ts, "_dws_channel", lambda *a, **k: MagicMock(name="channel")
        )
        monkeypatch.setattr(ts.health_pb2_grpc, "HealthStub", lambda channel: stub)
        return stub

    def test_serving(self, monkeypatch, capsys, stub):
        stub.Check.return_value = MagicMock(
            status=health_pb2.HealthCheckResponse.SERVING
        )

        ts.check_dws_health("localhost:50052")

        assert "up and running ✔" in capsys.readouterr().out

    def test_not_serving(self, monkeypatch, capsys, stub):
        stub.Check.return_value = MagicMock(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

        ts.check_dws_health("localhost:50052")

        assert "not serving" in capsys.readouterr().out

    def test_tls_error_classified(self, monkeypatch, capsys, stub):
        stub.Check.side_effect = _FakeRpcError("Ssl handshake failed")

        ts.check_dws_health("localhost:50052")

        assert "TLS handshake failed" in capsys.readouterr().out

    def test_connection_error(self, monkeypatch, capsys, stub):
        stub.Check.side_effect = _FakeRpcError("Connection refused")

        ts.check_dws_health("localhost:50052")

        assert "Failed to reach Duo Workflow Service" in capsys.readouterr().out

    def test_bails_when_channel_unavailable(self, monkeypatch, capsys):
        monkeypatch.setattr(ts, "_dws_channel", lambda *a, **k: None)

        ts.check_dws_health("localhost:50052")

        assert "up and running" not in capsys.readouterr().out


class TestTroubleshoot:
    def test_runs_new_checks(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["troubleshoot"])
        for name in (
            "check_general_env_variables",
            "check_aigw_endpoint",
            "check_gitlab_connectivity",
            "check_customer_portal_reachable",
            "check_dws_health",
        ):
            monkeypatch.setattr(ts, name, MagicMock())

        ts.troubleshoot()

        ts.check_customer_portal_reachable.assert_called_once()
        ts.check_dws_health.assert_called_once_with("localhost:50052")
