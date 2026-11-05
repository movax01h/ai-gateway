import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from prometheus_client import REGISTRY
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.api.middleware.actor_limit import ActorLimitMiddleware
from ai_gateway.config import ActorLimitMode, ConfigActorLimits
from lib.context import StarletteUser

LIMIT = 2
INSTANCE_LIMIT = 3
NAMESPACE_LIMIT = 3
CODE_PATHS = ("/v2/completions", "/v2/code/", "/v3/code/", "/v4/code/")
PROMPT_PATHS = ("/v1/prompts/", "/v2/prompts/")
CONTROL = "code_suggestions_actor_limit"
PROMPTS_CONTROL = "prompts_actor_limit"
GATEWAY_ISSUER = "gitlab-ai-gateway"
REJECTION = {
    "detail": "Too many concurrent requests",
    "reason": "rate_limited",
}


def limit_exceeded_event(
    mode: str,
    outcome: str,
    bucket: str,
    actor: str,
    limit: int,
    author: str | None = "user-a",
) -> dict:
    return {
        "event": "Actor limit exceeded",
        "log_level": "warning",
        "event_type": "security.rate_limit.block",
        "author_id": author,
        "entity_id": actor,
        "bucket": bucket,
        "in_flight": limit,
        "path": "/v2/code/completions",
        "details": {"outcome": outcome, "reason": "rate_limited"},
        "gitlab": {
            "control_name": CONTROL,
            "control_type": "rate_limit",
            "threshold": limit,
            "mode": mode,
        },
    }


def make_limits(mode: ActorLimitMode = "enforce", **overrides) -> ConfigActorLimits:
    values = {
        "actor_limit_mode": mode,
        "user_limit": LIMIT,
        "instance_limit": INSTANCE_LIMIT,
        "instance_limit_per_seat": 0,
        "namespace_limit": NAMESPACE_LIMIT,
    }
    return ConfigActorLimits(**{**values, **overrides})


def make_user(
    user_id: str | None = "user-a", claims: UserClaims | None = None
) -> StarletteUser:
    return StarletteUser(
        CloudConnectorUser(authenticated=True, global_user_id=user_id, claims=claims)
    )


class UserFromHeaders:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = Headers(scope=scope)
        realm = headers.get("x-test-realm", "self-managed")
        claims = UserClaims(
            gitlab_realm=realm,
            gitlab_instance_uid=(
                headers.get("x-test-instance", "inst-1")
                if realm == "self-managed"
                else ""
            ),
            extra=(
                {"gitlab_root_namespace_id": headers["x-test-namespace"]}
                if "x-test-namespace" in headers
                else None
            ),
        )
        if user_id := headers.get("x-test-user"):
            scope["user"] = make_user(user_id, claims)
        elif headers.get("x-test-no-user-id"):
            scope["user"] = make_user(None, claims)
        await self.app(scope, receive, send)


def sample(
    name: str, labels: dict[str, str] | None = None, control: str = CONTROL
) -> float:
    return (
        REGISTRY.get_sample_value(name, {"control": control, **(labels or {})}) or 0.0
    )


@pytest.fixture(name="mode")
def mode_fixture(request: pytest.FixtureRequest) -> ActorLimitMode:
    return getattr(request, "param", "enforce")


@pytest.fixture(name="gate")
def gate_fixture() -> asyncio.Event:
    return asyncio.Event()


@pytest.fixture(name="entered")
def entered_fixture() -> list[str]:
    return []


@pytest.fixture(name="limits")
def limits_fixture(
    request: pytest.FixtureRequest, mode: ActorLimitMode
) -> ConfigActorLimits:
    return make_limits(mode, **getattr(request, "param", {}))


@pytest.fixture(name="held_app")
def held_app_fixture(gate: asyncio.Event, entered: list[str]) -> FastAPI:
    app = FastAPI()

    @app.post("/v2/code/completions")
    @app.post("/v4/code/suggestions")
    @app.post("/v1/prompts/other")
    async def held(request: Request) -> dict[str, str]:
        entered.append(request.url.path)
        await gate.wait()
        return {"path": request.url.path}

    @app.post("/v2/code/generations")
    async def failing() -> None:
        raise RuntimeError("boom")

    return app


@pytest.fixture(name="limiter")
def limiter_fixture(
    held_app: FastAPI, limits: ConfigActorLimits
) -> ActorLimitMiddleware:
    return ActorLimitMiddleware(
        held_app, limits=limits, path_prefixes=CODE_PATHS, control_name=CONTROL
    )


@pytest.fixture(name="asgi_app")
def asgi_app_fixture(limiter: ActorLimitMiddleware) -> ASGIApp:
    return UserFromHeaders(limiter)


@pytest.fixture(name="client")
def client_fixture(asgi_app: ASGIApp) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=asgi_app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def post(
    client: httpx.AsyncClient,
    path: str = "/v2/code/completions",
    user: str | None = "user-a",
    instance: str = "inst-1",
    realm: str | None = None,
    namespace: str | None = None,
) -> httpx.Response:
    headers = {"x-test-instance": instance}
    if user is None:
        headers["x-test-no-user-id"] = "1"
    else:
        headers["x-test-user"] = user
    if realm:
        headers["x-test-realm"] = realm
    if namespace:
        headers["x-test-namespace"] = namespace
    return await client.post(path, headers=headers)


def scope_for(user: StarletteUser) -> Scope:
    return {"type": "http", "path": "/v2/code/completions", "headers": [], "user": user}


async def wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        ("/v1/prompts/other", {"x-test-user": "user-a"}),
        ("/v2/code/completions", {}),
        ("/v2/code/completions", {"x-test-no-user-id": "1", "x-test-realm": "saas"}),
    ],
    ids=["other path", "no user", "no user id and no tenant"],
)
def test_passes_through_without_counting(asgi_app, limiter, gate, path, headers):
    gate.set()

    response = TestClient(asgi_app).post(path, headers=headers)

    assert response.status_code == 200
    assert not limiter._in_flight


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        (
            UserClaims(
                issuer=GATEWAY_ISSUER,
                subject="user-a",
                gitlab_realm="saas",
                gitlab_instance_id="inst-1",
                extra={"gitlab_root_namespace_id": 42},
            ),
            [("user", "user-a", LIMIT), ("namespace", "42", NAMESPACE_LIMIT)],
        ),
        (
            UserClaims(
                issuer=GATEWAY_ISSUER,
                subject="user-a",
                gitlab_realm="saas",
                gitlab_instance_id="inst-1",
            ),
            [("user", "user-a", LIMIT)],
        ),
        (
            UserClaims(
                issuer=GATEWAY_ISSUER,
                subject="user-a",
                gitlab_realm="self-managed",
                gitlab_instance_id="inst-1",
                gitlab_instance_uid="uid-1",
            ),
            [("user", "uid-1:user-a", LIMIT), ("instance", "uid-1", INSTANCE_LIMIT)],
        ),
        (
            UserClaims(
                issuer="https://gitlab.example.com",
                subject="instance-uuid",
                gitlab_realm="self-managed",
                gitlab_instance_id="inst-1",
            ),
            [
                ("user", "instance-uuid:user-a", LIMIT),
                ("instance", "instance-uuid", INSTANCE_LIMIT),
            ],
        ),
        (
            UserClaims(
                issuer="https://customers.gitlab.com",
                subject="inst-1",
                gitlab_realm="self-managed",
                gitlab_instance_id="inst-1",
            ),
            [("user", "self-managed:user-a", LIMIT)],
        ),
        (None, [("user", "user-a", LIMIT)]),
    ],
    ids=[
        "saas with root namespace",
        "saas without root namespace",
        "self-managed with instance uid",
        "instance-signed",
        "customers dot without instance uid",
        "no claims",
    ],
)
def test_buckets(claims, expected):
    limiter = ActorLimitMiddleware(
        FastAPI(), limits=make_limits(), path_prefixes=CODE_PATHS, control_name=CONTROL
    )

    assert limiter._buckets(make_user("user-a", claims)) == expected


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        (
            UserClaims(gitlab_realm="self-managed", gitlab_instance_uid="uid-1"),
            [("instance", "uid-1", INSTANCE_LIMIT)],
        ),
        (
            UserClaims(gitlab_realm="saas", extra={"gitlab_root_namespace_id": 42}),
            [("namespace", "42", NAMESPACE_LIMIT)],
        ),
        (UserClaims(gitlab_realm="saas"), []),
        (None, []),
    ],
    ids=["instance", "namespace", "saas without root namespace", "no claims"],
)
def test_buckets_without_a_user_id(claims, expected):
    limiter = ActorLimitMiddleware(
        FastAPI(), limits=make_limits(), path_prefixes=CODE_PATHS, control_name=CONTROL
    )

    assert limiter._buckets(make_user(None, claims)) == expected


@pytest.mark.parametrize(
    ("seats", "expected"),
    [("", INSTANCE_LIMIT), ("20", INSTANCE_LIMIT), ("55", 6), (1000, 100)],
    ids=["no seat count", "below the floor", "rounded up", "integer claim"],
)
def test_instance_limit_scales_with_the_seat_count(seats, expected):
    limiter = ActorLimitMiddleware(
        FastAPI(),
        limits=make_limits(instance_limit_per_seat=0.1),
        path_prefixes=CODE_PATHS,
        control_name=CONTROL,
    )
    claims = UserClaims(
        gitlab_realm="self-managed", gitlab_instance_uid="uid-1", duo_seat_count=seats
    )

    assert limiter._buckets(make_user("user-a", claims))[1].limit == expected


@pytest.mark.asyncio
async def test_enforce_rejects_the_request_above_the_user_limit(client, limiter, gate):
    labels = {"mode": "enforce", "outcome": "rejected", "bucket": "user"}
    rejected_before = sample("actor_limit_total", labels)
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT)]
        await wait_until(
            lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT
        )

        with structlog.testing.capture_logs() as logs:
            rejected = await post(client)
        gate.set()
        responses = await asyncio.gather(*held)

    assert [log for log in logs if log["log_level"] == "warning"] == [
        limit_exceeded_event("enforce", "failure", "user", "inst-1:user-a", LIMIT)
    ]
    assert rejected.status_code == 429
    assert rejected.json() == REJECTION
    assert rejected.headers["retry-after"] == "1"
    assert rejected.headers["connection"] == "close"
    assert rejected.headers["x-gitlab-rate-limit"] == CONTROL
    assert [r.status_code for r in responses] == [200, 200]
    assert not limiter._in_flight
    assert sample("actor_limit_total", labels) == rejected_before + 1


@pytest.mark.asyncio
async def test_enforce_rejects_rotating_users_on_one_instance(client, limiter, gate):
    labels = {"mode": "enforce", "outcome": "rejected", "bucket": "instance"}
    rejected_before = sample("actor_limit_total", labels)
    async with client:
        held = [
            asyncio.create_task(post(client, user=f"user-{i}"))
            for i in range(INSTANCE_LIMIT)
        ]
        await wait_until(
            lambda: limiter._in_flight.get(("instance", "inst-1")) == INSTANCE_LIMIT
        )

        with structlog.testing.capture_logs() as logs:
            rejected = await post(client, user="user-new")
        other_instance = asyncio.create_task(
            post(client, user="user-new", instance="inst-2")
        )
        await wait_until(lambda: limiter._in_flight.get(("instance", "inst-2")) == 1)
        gate.set()
        responses = await asyncio.gather(*held, other_instance)

    assert rejected.status_code == 429
    assert [log for log in logs if log["log_level"] == "warning"] == [
        limit_exceeded_event(
            "enforce",
            "failure",
            "instance",
            "inst-1",
            INSTANCE_LIMIT,
            author="user-new",
        )
    ]
    assert [r.status_code for r in responses] == [200] * (INSTANCE_LIMIT + 1)
    assert not limiter._in_flight
    assert sample("actor_limit_total", labels) == rejected_before + 1


@pytest.mark.asyncio
async def test_enforce_counts_requests_without_a_user_id_against_the_instance(
    client, limiter, gate
):
    labels = {"mode": "enforce", "outcome": "rejected", "bucket": "instance"}
    rejected_before = sample("actor_limit_total", labels)
    async with client:
        held = [
            asyncio.create_task(post(client, user=None)) for _ in range(INSTANCE_LIMIT)
        ]
        await wait_until(
            lambda: limiter._in_flight.get(("instance", "inst-1")) == INSTANCE_LIMIT
        )

        with structlog.testing.capture_logs() as logs:
            rejected = await post(client, user=None)
        gate.set()
        responses = await asyncio.gather(*held)

    assert rejected.status_code == 429
    assert [log for log in logs if log["log_level"] == "warning"] == [
        limit_exceeded_event(
            "enforce", "failure", "instance", "inst-1", INSTANCE_LIMIT, author=None
        )
    ]
    assert [r.status_code for r in responses] == [200] * INSTANCE_LIMIT
    assert not limiter._in_flight
    assert sample("actor_limit_total", labels) == rejected_before + 1


@pytest.mark.asyncio
async def test_user_ids_are_scoped_to_the_instance(client, limiter, gate):
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT)]
        await wait_until(
            lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT
        )
        same_id_elsewhere = asyncio.create_task(post(client, instance="inst-2"))
        await wait_until(lambda: limiter._in_flight.get(("user", "inst-2:user-a")) == 1)

        gate.set()
        responses = await asyncio.gather(*held, same_id_elsewhere)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert not limiter._in_flight


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [{"instance_limit": LIMIT}], indirect=True)
async def test_enforce_records_every_exceeded_bucket(client, limiter, gate):
    user_labels = {"mode": "enforce", "outcome": "rejected", "bucket": "user"}
    instance_labels = {"mode": "enforce", "outcome": "rejected", "bucket": "instance"}
    before = {
        "user": sample("actor_limit_total", user_labels),
        "instance": sample("actor_limit_total", instance_labels),
    }
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT)]
        await wait_until(
            lambda: limiter._in_flight.get(("instance", "inst-1")) == LIMIT
        )

        with structlog.testing.capture_logs() as logs:
            rejected = await post(client)
        gate.set()
        await asyncio.gather(*held)

    assert rejected.status_code == 429
    assert [log for log in logs if log["log_level"] == "warning"] == [
        limit_exceeded_event("enforce", "failure", "user", "inst-1:user-a", LIMIT),
        limit_exceeded_event("enforce", "failure", "instance", "inst-1", LIMIT),
    ]
    assert sample("actor_limit_total", user_labels) == before["user"] + 1
    assert sample("actor_limit_total", instance_labels) == before["instance"] + 1


@pytest.mark.asyncio
async def test_enforce_rejects_users_of_one_saas_root_namespace(client, limiter, gate):
    labels = {"mode": "enforce", "outcome": "rejected", "bucket": "namespace"}
    rejected_before = sample("actor_limit_total", labels)
    async with client:
        held = [
            asyncio.create_task(
                post(client, user=f"user-{i}", realm="saas", namespace="42")
            )
            for i in range(NAMESPACE_LIMIT)
        ]
        await wait_until(
            lambda: limiter._in_flight.get(("namespace", "42")) == NAMESPACE_LIMIT
        )

        rejected = await post(client, user="user-new", realm="saas", namespace="42")
        other_namespace = asyncio.create_task(
            post(client, user="user-new", realm="saas", namespace="43")
        )
        await wait_until(lambda: limiter._in_flight.get(("namespace", "43")) == 1)
        gate.set()
        responses = await asyncio.gather(*held, other_namespace)

    assert rejected.status_code == 429
    assert [r.status_code for r in responses] == [200] * (NAMESPACE_LIMIT + 1)
    assert not limiter._in_flight
    assert sample("actor_limit_total", labels) == rejected_before + 1


@pytest.mark.asyncio
async def test_each_control_counts_only_its_own_routes(held_app, limiter, gate):
    prompts_limiter = ActorLimitMiddleware(
        held_app,
        limits=make_limits(),
        path_prefixes=PROMPT_PATHS,
        control_name=PROMPTS_CONTROL,
    )
    limiter.app = prompts_limiter
    labels = {"mode": "enforce", "outcome": "rejected", "bucket": "user"}
    rejected_before = sample("actor_limit_total", labels, control=PROMPTS_CONTROL)
    code_rejected_before = sample("actor_limit_total", labels)
    transport = httpx.ASGITransport(
        app=UserFromHeaders(limiter), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        held = [
            asyncio.create_task(post(client, path="/v1/prompts/other"))
            for _ in range(LIMIT)
        ]
        await wait_until(
            lambda: prompts_limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT
        )

        rejected = await post(client, path="/v1/prompts/other")
        completion = asyncio.create_task(post(client))
        await wait_until(lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == 1)
        gate.set()
        responses = await asyncio.gather(*held, completion)

    assert rejected.status_code == 429
    assert rejected.headers["x-gitlab-rate-limit"] == PROMPTS_CONTROL
    assert [r.status_code for r in responses] == [200] * (LIMIT + 1)
    assert not limiter._in_flight
    assert not prompts_limiter._in_flight
    assert (
        sample("actor_limit_total", labels, control=PROMPTS_CONTROL)
        == rejected_before + 1
    )
    assert sample("actor_limit_total", labels) == code_rejected_before


@pytest.mark.asyncio
async def test_users_do_not_share_a_limit(client, limiter, gate):
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT)]
        await wait_until(
            lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT
        )
        other_user = asyncio.create_task(post(client, user="user-b"))
        await wait_until(lambda: limiter._in_flight.get(("user", "inst-1:user-b")) == 1)

        gate.set()
        responses = await asyncio.gather(*held, other_user)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert not limiter._in_flight


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["log"], indirect=True)
async def test_log_mode_serves_and_records_the_excess_request(client, limiter, gate):
    labels = {"mode": "log", "outcome": "would_reject", "bucket": "user"}
    would_reject_before = sample("actor_limit_total", labels)
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT)]
        await wait_until(
            lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT
        )
        with structlog.testing.capture_logs() as logs:
            excess = asyncio.create_task(post(client))
            await wait_until(
                lambda: limiter._in_flight.get(("user", "inst-1:user-a")) == LIMIT + 1
            )
        gate.set()
        responses = await asyncio.gather(*held, excess)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert not limiter._in_flight
    assert [log for log in logs if log["log_level"] == "warning"] == [
        limit_exceeded_event("log", "success", "user", "inst-1:user-a", LIMIT)
    ]
    assert sample("actor_limit_total", labels) == would_reject_before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off"], indirect=True)
async def test_off_mode_does_not_count(client, limiter, gate, entered):
    async with client:
        held = [asyncio.create_task(post(client)) for _ in range(LIMIT + 1)]
        await wait_until(lambda: len(entered) == LIMIT + 1)
        gate.set()
        responses = await asyncio.gather(*held)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert not limiter._in_flight


@pytest.mark.asyncio
async def test_releases_the_slot_when_the_handler_fails(client, limiter):
    async with client:
        response = await post(client, path="/v2/code/generations")

    assert response.status_code == 500
    assert not limiter._in_flight


@pytest.mark.asyncio
async def test_observes_the_in_flight_count_of_every_bucket(client, gate):
    before = {
        bucket: (
            sample("actor_in_flight_count", {"bucket": bucket}),
            sample("actor_in_flight_sum", {"bucket": bucket}),
        )
        for bucket in ("user", "instance")
    }
    gate.set()
    async with client:
        await post(client)

    for bucket, (count, total) in before.items():
        assert sample("actor_in_flight_count", {"bucket": bucket}) == count + 1
        assert sample("actor_in_flight_sum", {"bucket": bucket}) == total + 1


@pytest.mark.asyncio
async def test_releases_the_slot_when_the_request_is_cancelled():
    started = asyncio.Event()

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        started.set()
        await asyncio.Event().wait()

    async def receive() -> Message:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(_message: Message) -> None:
        pass

    limiter = ActorLimitMiddleware(
        app, limits=make_limits(), path_prefixes=CODE_PATHS, control_name=CONTROL
    )
    task = asyncio.create_task(limiter(scope_for(make_user()), receive, send))
    await started.wait()
    assert limiter._in_flight == {("user", "user-a"): 1}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not limiter._in_flight


@pytest.mark.asyncio
async def test_releases_the_slot_when_the_client_disconnects_mid_stream():
    sent: list[Message] = []

    async def chunks() -> AsyncIterator[bytes]:
        while True:
            yield b"x"
            await asyncio.sleep(0)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await StreamingResponse(chunks())(scope, receive, send)

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    limiter = ActorLimitMiddleware(
        app, limits=make_limits(), path_prefixes=CODE_PATHS, control_name=CONTROL
    )

    await limiter(scope_for(make_user()), receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert not limiter._in_flight
