import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from ai_gateway.api.middleware import HostHeaderValidationMiddleware


@pytest.fixture(name="client")
def client_fixture():
    fastapi_app = FastAPI()
    fastapi_app.add_middleware(HostHeaderValidationMiddleware)

    @fastapi_app.get("/ping")
    def _ping():
        return JSONResponse({"ok": True})

    @fastapi_app.get("/host")
    def _host(request: Request):
        return JSONResponse({"host": request.headers.get("host")})

    return TestClient(fastapi_app)


@pytest.mark.parametrize(
    "host",
    [
        "testserver",
        "example.com",
        "example.com:8080",
        "sub.domain.example.com",
        "127.0.0.1",
        "127.0.0.1:8080",
        "[::1]",
        "[::1]:8080",
        "xn--bcher-kva.example",
    ],
)
def test_valid_hosts_pass_through(client: TestClient, host: str):
    response = client.get("/ping", headers={"Host": host})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "host",
    [
        "example.com/x",  # path delimiter — the X41-2026-002 vector
        "example.com?",  # query delimiter
        "example.com#frag",  # fragment delimiter
        "example com",  # internal space
        "example\tcom",  # tab
        "example\x00com",  # NUL byte
        "x/monitoring/healthz?",  # the concrete auth-bypass crafted value
        "",  # empty Host
        "host_with_underscore.example",  # underscores not in RFC 952/1123 allowlist
    ],
)
def test_invalid_hosts_are_sanitised(client: TestClient, host: str):
    """Requests with malformed Host headers proceed but the header is stripped."""
    response = client.get("/host", headers={"Host": host})
    assert response.status_code == 200
    assert response.json()["host"] is None


@pytest.mark.asyncio
async def test_no_host_header_passes_through_unchanged():
    """A request with no Host header must not be modified."""
    received_scopes = []

    async def inner(scope, _receive, _send):
        received_scopes.append(scope)

    middleware = HostHeaderValidationMiddleware(inner)
    scope = {"type": "http", "headers": [(b"content-type", b"application/json")]}
    await middleware(scope, None, None)

    assert received_scopes[0] is scope


@pytest.mark.asyncio
async def test_non_http_scope_passes_through_unchanged():
    """Scopes other than http/websocket (e.g. lifespan) bypass header logic entirely."""
    received_scopes = []

    async def inner(scope, _receive, _send):
        received_scopes.append(scope)

    middleware = HostHeaderValidationMiddleware(inner)
    lifespan_scope = {"type": "lifespan", "headers": [(b"host", b"evil/bypass")]}
    await middleware(lifespan_scope, None, None)

    assert received_scopes[0] is lifespan_scope
