import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from ai_gateway.api import client_disconnect
from ai_gateway.api.client_disconnect import (
    ClientDisconnectedError,
    client_disconnected_exception_handler,
    run_unless_disconnected,
)

ROUTE = "/v2/code/completions"


@pytest.fixture(name="request_")
def request_fixture():
    request = MagicMock()
    request.url.path = ROUTE
    request.is_disconnected = AsyncMock(return_value=False)
    return request


@pytest.fixture(name="fast_poll")
def fast_poll_fixture():
    with patch.object(client_disconnect, "_DISCONNECT_POLL_INTERVAL_SECONDS", 0.001):
        yield


@pytest.fixture(name="cancelled")
def cancelled_fixture():
    return asyncio.Event()


@pytest.fixture(name="hanging")
def hanging_fixture(cancelled):
    async def _hanging():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    return _hanging


async def _value(value):
    return value


def _disconnects(route: str) -> float:
    return (
        REGISTRY.get_sample_value("client_disconnects_total", {"route": route}) or 0.0
    )


@pytest.mark.asyncio
async def test_returns_the_result_while_the_client_stays(request_):
    assert await run_unless_disconnected(request_, _value("ok")) == "ok"


@pytest.mark.asyncio
async def test_reraises_the_awaitable_error(request_):
    async def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_unless_disconnected(request_, failing())


@pytest.mark.asyncio
@pytest.mark.usefixtures("fast_poll")
async def test_cancels_the_awaitable_once_the_client_disconnects(
    request_, hanging, cancelled
):
    request_.is_disconnected.return_value = True
    before = _disconnects(ROUTE)

    with pytest.raises(ClientDisconnectedError):
        await run_unless_disconnected(request_, hanging())

    assert cancelled.is_set()
    assert _disconnects(ROUTE) == before + 1


@pytest.mark.asyncio
async def test_awaits_directly_when_disabled(request_):
    request_.is_disconnected.return_value = True

    assert await run_unless_disconnected(request_, _value("ok"), enabled=False) == "ok"

    request_.is_disconnected.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.usefixtures("fast_poll")
async def test_outer_cancellation_reaches_the_awaitable(request_, hanging, cancelled):
    outer = asyncio.ensure_future(run_unless_disconnected(request_, hanging()))
    await asyncio.sleep(0.01)

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_exception_handler_answers_499(request_):
    response = await client_disconnected_exception_handler(
        request_, ClientDisconnectedError()
    )

    assert response.status_code == 499
