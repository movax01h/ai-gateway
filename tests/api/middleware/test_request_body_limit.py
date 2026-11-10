import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import request_body_limit
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


@pytest.fixture(name="all_routes_client")
def all_routes_client_fixture(handled_paths: list[str]) -> TestClient:
    app = FastAPI(
        middleware=[
            Middleware(RawContextMiddleware),
            Middleware(
                RequestBodyLimitMiddleware,
                max_bytes=MAX_BYTES,
                path_prefixes=("/",),
                excluded_prefixes=("/v1/proxy/",),
            ),
        ]
    )

    @app.post("/v1/prompts/other")
    @app.post("/v2/chat/agent")
    @app.post("/v1/proxy/anthropic/v1/messages")
    async def echo_size(request: Request):
        body = await request.body()
        handled_paths.append(request.url.path)
        return {"size": len(body)}

    return TestClient(app)


@pytest.mark.parametrize("path", ["/v1/prompts/other", "/v2/chat/agent"])
def test_limits_every_path_except_the_excluded_prefixes(
    all_routes_client, handled_paths, path
):
    response = all_routes_client.post(path, content=b"x" * (MAX_BYTES + 1))

    assert response.status_code == 413
    assert handled_paths == []


def test_passes_a_large_body_on_an_excluded_prefix(all_routes_client, handled_paths):
    response = all_routes_client.post(
        "/v1/proxy/anthropic/v1/messages", content=b"x" * (MAX_BYTES * 10)
    )

    assert response.status_code == 200
    assert response.json() == {"size": MAX_BYTES * 10}
    assert handled_paths == ["/v1/proxy/anthropic/v1/messages"]


@pytest.fixture(name="stacked_client")
def stacked_client_fixture(handled_paths: list[str]) -> TestClient:
    # The order on GitLab.com: the code suggestion limit, the limit on every route, then model config.
    app = FastAPI(
        middleware=[
            Middleware(RawContextMiddleware),
            Middleware(
                RequestBodyLimitMiddleware,
                max_bytes=MAX_BYTES,
                path_prefixes=CODE_PATHS,
            ),
            Middleware(
                RequestBodyLimitMiddleware,
                max_bytes=MAX_BYTES * 10,
                path_prefixes=("/",),
                excluded_prefixes=("/v1/proxy/",),
            ),
            Middleware(ModelConfigMiddleware),
        ]
    )

    @app.post("/v2/code/completions")
    @app.post("/v1/prompts/other")
    async def echo_size(request: Request):
        body = await request.body()
        handled_paths.append(request.url.path)
        return {"size": len(body)}

    return TestClient(app)


@pytest.mark.parametrize(
    ("path", "max_bytes"),
    [("/v2/code/completions", MAX_BYTES), ("/v1/prompts/other", MAX_BYTES * 10)],
    ids=["code suggestion limit", "limit on every route"],
)
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "streamed"])
def test_names_the_limit_that_the_body_crossed(
    stacked_client, handled_paths, path, max_bytes, declared
):
    chunks = [b"x" * max_bytes, b"x"]
    content = b"".join(chunks) if declared else iter(chunks)

    response = stacked_client.post(path, content=content)

    assert response.status_code == 413
    assert response.json() == {"detail": f"Request body exceeds {max_bytes} bytes"}
    assert handled_paths == []


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


def _chunk(size: int, more_body: bool = True) -> dict:
    return {"type": "http.request", "body": b"x" * size, "more_body": more_body}


async def _run_limited(
    messages: list[dict], headers: list, app=None
) -> tuple[list[int], list]:
    """Run the middleware on a code suggestion path; return what the app consumed and what was sent."""
    consumed: list[int] = []
    sent: list[dict] = []

    async def consuming_app(_scope, receive, _send):
        while True:
            message = await receive()
            consumed.append(len(message["body"]))
            if not message["more_body"]:
                return

    async def receive():
        assert messages, "read past the end of the request"
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(
        app or consuming_app, max_bytes=MAX_BYTES, path_prefixes=CODE_PATHS
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/code/completions",
        "raw_path": b"/v2/code/completions",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "http_version": "1.1",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "headers": headers,
    }
    await middleware(scope, receive, send)
    return consumed, sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "app_consumed"),
    [([(b"content-length", b"180")], []), ([], [60])],
    ids=["declared", "streamed"],
)
async def test_reads_the_rest_of_a_rejected_body_before_the_413(headers, app_consumed):
    messages = [_chunk(60), _chunk(60), _chunk(60, more_body=False)]

    consumed, sent = await _run_limited(messages, headers)

    assert consumed == app_consumed
    assert messages == []
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_reads_the_rest_of_the_body_when_the_route_answers_the_413():
    # A route that reads the body itself: FastAPI turns the HTTPException into the 413.
    route_app = FastAPI()

    @route_app.post("/v2/code/completions")
    async def read_body(request: Request):
        await request.body()

    messages = [_chunk(60), _chunk(60), _chunk(60, more_body=False)]

    _consumed, sent = await _run_limited(messages, [], app=route_app)

    assert messages == []
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_does_not_wait_for_more_body_when_the_last_chunk_crosses_the_limit():
    messages = [_chunk(60), _chunk(60, more_body=False)]

    _consumed, sent = await _run_limited(messages, [])

    assert messages == []
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_stops_reading_a_rejected_body_when_the_client_disconnects():
    messages = [_chunk(60), {"type": "http.disconnect"}]

    _consumed, sent = await _run_limited(messages, [(b"content-length", b"180")])

    assert messages == []
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_stops_reading_a_rejected_body_at_the_drain_limit(monkeypatch):
    monkeypatch.setattr(request_body_limit, "_MAX_DRAIN_BYTES", 100)
    messages = [_chunk(60), _chunk(60), _chunk(60), _chunk(60, more_body=False)]

    _consumed, sent = await _run_limited(messages, [(b"content-length", b"240")])

    assert len(messages) == 2
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
            "declared": declared,
            "max_bytes": MAX_BYTES,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "messages", "max_drain_bytes", "drained_bytes", "outcome"),
    [
        pytest.param(
            [(b"content-length", b"180")],
            [_chunk(60), _chunk(60), _chunk(60, more_body=False)],
            1000,
            180,
            "body_complete",
            id="declared",
        ),
        pytest.param(
            [],
            [_chunk(60), _chunk(60), _chunk(60, more_body=False)],
            1000,
            60,
            "body_complete",
            id="streamed",
        ),
        pytest.param(
            [(b"content-length", b"180")],
            [_chunk(60), {"type": "http.disconnect"}],
            1000,
            60,
            "client_disconnected",
            id="client-disconnects",
        ),
        pytest.param(
            [(b"content-length", b"240")],
            [_chunk(60), _chunk(60), _chunk(60), _chunk(60, more_body=False)],
            100,
            120,
            "drain_limit",
            id="drain-limit",
        ),
    ],
)
async def test_logs_how_the_drain_ends(
    monkeypatch, headers, messages, max_drain_bytes, drained_bytes, outcome
):
    monkeypatch.setattr(request_body_limit, "_MAX_DRAIN_BYTES", max_drain_bytes)

    with structlog.testing.capture_logs() as logs:
        await _run_limited(messages, headers)

    drains = [log for log in logs if log["event"] == "Rejected request body drained"]
    assert drains == [
        {
            "event": "Rejected request body drained",
            "log_level": "info",
            "path": "/v2/code/completions",
            "drained_bytes": drained_bytes,
            "outcome": outcome,
            "duration_s": pytest.approx(0, abs=1),
        }
    ]
