"""Mock usage quota server for local development.

This module provides a lightweight HTTP server that mimics the CustomersDot
usage quota endpoint (``/api/v1/consumers/resolve``). It is intended for local
development only and is started automatically when ``AIGW_MOCK_USAGE_CREDITS=true``.

The server is a Python re-implementation of the Ruby WEBrick server from
https://gitlab.com/gitlab-org/ai-powered/mock-cred-cd.

Endpoints
---------
HEAD /api/v1/consumers/resolve
    Returns 200 when credits are available, 402 when exhausted, 400 on bad input.

GET /api/dev/flip_credits
    Toggles the credit state and returns the new state as JSON.
"""

import errno
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import structlog

log = structlog.stdlib.get_logger("usage_quota.mock_server")

DEFAULT_PORT = 4567

# Module-level credit state shared across all request handlers.
_has_credits = True  # pylint: disable=invalid-name
_credits_lock = threading.Lock()


def _get_has_credits() -> bool:
    with _credits_lock:
        return _has_credits


def _toggle_credits() -> bool:
    global _has_credits  # pylint: disable=global-statement
    with _credits_lock:
        _has_credits = not _has_credits
        return _has_credits


class MockUsageQuotaHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the mock CustomersDot usage quota server."""

    def log_message(self, fmt: str, *args) -> None:
        """Route access logs through structlog instead of stderr."""
        log.debug("access_log", message=fmt % args if args else fmt)

    def do_HEAD(self) -> None:  # pylint: disable=invalid-name
        """Handle HEAD /api/v1/consumers/resolve."""
        parsed = urlparse(self.path)
        if parsed.path != "/api/v1/consumers/resolve":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        def _get(key: str) -> str:
            values = params.get(key, [])
            return values[0] if values else ""

        user_id = _get("user_id")
        realm = _get("realm")
        root_namespace_id = _get("root_namespace_id")
        unique_instance_id = _get("unique_instance_id")

        log.debug(
            "HEAD /api/v1/consumers/resolve",
            user_id=user_id,
            realm=realm,
            root_namespace_id=root_namespace_id,
            unique_instance_id=unique_instance_id,
        )

        if not user_id or not realm:
            self.send_response(400)
            self.end_headers()
            return

        if not root_namespace_id and not unique_instance_id:
            self.send_response(400)
            self.end_headers()
            return

        status = 200 if _get_has_credits() else 402
        self.send_response(status)
        self.end_headers()

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handle GET /api/dev/flip_credits."""
        parsed = urlparse(self.path)
        if parsed.path != "/api/dev/flip_credits":
            body = json.dumps({"error": "Not Found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        new_state = _toggle_credits()
        body = json.dumps({"has_credits": new_state}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_mock_server(port: int = DEFAULT_PORT) -> threading.Thread | None:
    """Start the mock usage quota server in a daemon background thread.

    Intended for local development only — never enable in production.

    This function is invoked from both AI Gateway and Duo Workflow Service
    startup paths. When both run on the same host (e.g. local development via
    GDK), the second caller will find the port already in use. To avoid
    crashing either service, we log a warning and return ``None`` instead of
    raising — assuming the first caller has already started the server.

    Args:
        port: TCP port to listen on (default: 4567).

    Returns:
        The daemon :class:`threading.Thread` running the server, or ``None``
        if the port was already in use.
    """
    try:
        server = HTTPServer(("0.0.0.0", port), MockUsageQuotaHandler)
    except OSError as e:
        # Only swallow EADDRINUSE: the other service likely started the server first.
        # Re-raise other OSErrors (e.g. EACCES on privileged ports) so they surface.
        if e.errno != errno.EADDRINUSE:
            raise
        log.warning(
            "Mock usage quota server port is busy; "
            "assuming the mock server is already running and skipping start",
            port=port,
            error=str(e),
        )
        return None

    thread = threading.Thread(
        target=server.serve_forever,
        name="mock-usage-quota-server",
        daemon=True,  # Automatically stops when the main process exits.
    )
    thread.start()

    log.info("Mock usage quota server started", url=f"http://localhost:{port}")
    return thread
