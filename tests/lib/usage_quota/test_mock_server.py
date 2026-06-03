# pylint: disable=redefined-outer-name
"""Tests for the mock usage quota server.

Tests exercise the server via its public HTTP interface only.
"""

import errno as errno_module
import json
import socket
import time
from http.client import HTTPConnection
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from lib.usage_quota.mock_server import start_mock_server

RESOLVE_PATH = "/api/v1/consumers/resolve"
FLIP_PATH = "/api/dev/flip_credits"


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 2.0) -> bool:
    """Block until *port* is accepting connections or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _head(port: int, path: str) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("HEAD", path)
    resp = conn.getresponse()
    conn.close()
    return resp.status


def _get(port: int, path: str):
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    parsed = json.loads(body) if body else None
    return resp.status, parsed


def _set_credits(port: int, available: bool) -> None:
    """Drive the server's credit state to *available* via the public flip endpoint."""
    for _ in range(2):
        _, body = _get(port, FLIP_PATH)
        if body and body.get("has_credits") is available:
            return
    raise AssertionError(f"Could not set credit state to {available}")


@pytest.fixture(scope="module")
def server_port() -> int:
    """Start the mock server once per module and return its port."""
    port = _find_free_port()
    thread = start_mock_server(port=port)
    assert thread is not None and thread.is_alive()
    assert _wait_for_port(port), f"Server did not start on port {port}"
    return port


@pytest.fixture(autouse=True)
def credits_available(server_port: int):
    """Ensure each test starts with credits available."""
    _set_credits(server_port, True)


class TestHeadConsumersResolve:
    def test_returns_200_when_credits_available(self, server_port):
        assert (
            _head(
                server_port,
                f"{RESOLVE_PATH}?user_id=1&realm=saas&root_namespace_id=42",
            )
            == 200
        )

    def test_returns_402_when_credits_exhausted(self, server_port):
        _set_credits(server_port, False)
        assert (
            _head(
                server_port,
                f"{RESOLVE_PATH}?user_id=1&realm=saas&root_namespace_id=42",
            )
            == 402
        )

    @pytest.mark.parametrize(
        "query",
        [
            # user_id missing
            "realm=saas&root_namespace_id=42",
            # realm missing
            "user_id=1&root_namespace_id=42",
            # both root_namespace_id and unique_instance_id missing
            "user_id=1&realm=saas",
        ],
    )
    def test_returns_400_on_missing_required_params(self, server_port, query):
        assert _head(server_port, f"{RESOLVE_PATH}?{query}") == 400

    def test_accepts_unique_instance_id_instead_of_root_namespace_id(self, server_port):
        assert (
            _head(
                server_port,
                f"{RESOLVE_PATH}?user_id=1&realm=saas&unique_instance_id=abc",
            )
            == 200
        )

    def test_returns_404_for_unknown_path(self, server_port):
        assert _head(server_port, "/unknown") == 404


class TestStartMockServer:
    def test_returns_none_and_logs_warning_when_port_is_busy(self, server_port):
        """A second call on an already-bound port should warn and return None."""
        with capture_logs() as cap_logs:
            result = start_mock_server(port=server_port)

        assert result is None
        assert any(
            entry.get("log_level") == "warning"
            and "is busy" in entry.get("event", "")
            and "already running" in entry.get("event", "")
            for entry in cap_logs
        )

    def test_reraises_non_eaddrinuse_oserror(self):
        """Non-EADDRINUSE OSErrors (e.g. EACCES) must propagate, not be swallowed."""
        eacces_error = OSError(errno_module.EACCES, "Permission denied")
        with patch("lib.usage_quota.mock_server.HTTPServer", side_effect=eacces_error):
            with pytest.raises(OSError) as exc_info:
                start_mock_server(port=80)

        assert exc_info.value.errno == errno_module.EACCES


class TestGetFlipCredits:
    def test_flip_toggles_state(self, server_port):
        status, body = _get(server_port, FLIP_PATH)
        assert status == 200
        assert body == {"has_credits": False}

        status, body = _get(server_port, FLIP_PATH)
        assert status == 200
        assert body == {"has_credits": True}

    def test_flip_affects_resolve_endpoint(self, server_port):
        """End-to-end: flipping credits changes the /resolve response code."""
        query = "user_id=1&realm=saas&root_namespace_id=42"
        assert _head(server_port, f"{RESOLVE_PATH}?{query}") == 200

        _get(server_port, FLIP_PATH)
        assert _head(server_port, f"{RESOLVE_PATH}?{query}") == 402

    def test_returns_404_for_unknown_get_path(self, server_port):
        status, body = _get(server_port, "/unknown")
        assert status == 404
        assert "error" in body
