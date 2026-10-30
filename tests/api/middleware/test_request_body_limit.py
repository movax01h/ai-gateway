import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware.model_config import ModelConfigMiddleware
from ai_gateway.api.middleware.request_body_limit import RequestBodyLimitMiddleware

MAX_BYTES = 100
CODE_PATHS = ("/v2/completions", "/v2/code/", "/v3/code/", "/v4/code/")


@pytest.fixture(name="handled_paths")
def handled_paths_fixture() -> list[str]:
    return []


@pytest.fixture(name="client", params=["limit only", "limit then model config"])
def client_fixture(request, handled_paths: list[str]) -> TestClient:
    middleware = [
        Middleware(RawContextMiddleware),
        Middleware(
            RequestBodyLimitMiddleware, max_bytes=MAX_BYTES, path_prefixes=CODE_PATHS
        ),
    ]
    if request.param == "limit then model config":
        middleware.append(Middleware(ModelConfigMiddleware))
    app = FastAPI(middleware=middleware)

    @app.post("/v2/code/completions")
    @app.post("/v2/completions")
    @app.post("/v3/code/completions")
    @app.post("/v4/code/suggestions")
    @app.post("/v1/prompts/other")
    async def echo_size(request: Request):
        body = await request.body()
        handled_paths.append(request.url.path)
        return {"size": len(body)}

    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/v2/code/completions",
        "/v2/completions",
        "/v3/code/completions",
        "/v4/code/suggestions",
    ],
)
def test_rejects_declared_body_over_the_limit(client, handled_paths, path):
    response = client.post(path, content=b"x" * (MAX_BYTES + 1))

    assert response.status_code == 413
    assert response.json() == {"detail": f"Request body exceeds {MAX_BYTES} bytes"}
    assert response.headers["connection"] == "close"
    assert handled_paths == []


def test_rejects_undeclared_body_over_the_limit(client, handled_paths):
    def chunks():
        yield b"x" * MAX_BYTES
        yield b"x"

    response = client.post("/v2/code/completions", content=chunks())

    assert response.status_code == 413
    assert response.headers["connection"] == "close"
    assert handled_paths == []


def test_passes_body_at_the_limit(client, handled_paths):
    response = client.post("/v2/code/completions", content=b"x" * MAX_BYTES)

    assert response.status_code == 200
    assert response.json() == {"size": MAX_BYTES}
    assert handled_paths == ["/v2/code/completions"]


def test_ignores_other_paths(client, handled_paths):
    response = client.post("/v1/prompts/other", content=b"x" * (MAX_BYTES * 10))

    assert response.status_code == 200
    assert handled_paths == ["/v1/prompts/other"]


@pytest.mark.asyncio
async def test_stops_a_streamed_body_at_the_chunk_that_crosses_the_limit():
    consumed: list[int] = []
    sent: list[dict] = []
    chunks = [
        {"type": "http.request", "body": b"x" * 60, "more_body": True},
        {"type": "http.request", "body": b"x" * 60, "more_body": True},
        {"type": "http.request", "body": b"x" * 60, "more_body": False},
    ]

    async def app(_scope, receive, _send):
        while True:
            message = await receive()
            consumed.append(len(message["body"]))
            if not message["more_body"]:
                return

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(
        app, max_bytes=MAX_BYTES, path_prefixes=CODE_PATHS
    )
    scope = {"type": "http", "path": "/v2/code/completions", "headers": []}

    await middleware(scope, receive, send)

    assert consumed == [60]
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "streamed"])
def test_logs_the_rejected_size(client, declared):
    chunks = [b"x" * MAX_BYTES, b"x" * 20]
    content = b"".join(chunks) if declared else iter(chunks)

    with structlog.testing.capture_logs() as logs:
        client.post("/v2/code/completions", content=content)

    rejections = [log for log in logs if log["event"] == "Request body exceeds limit"]
    assert rejections == [
        {
            "event": "Request body exceeds limit",
            "log_level": "warning",
            "path": "/v2/code/completions",
            "body_bytes": MAX_BYTES + 20,
            "max_bytes": MAX_BYTES,
        }
    ]
