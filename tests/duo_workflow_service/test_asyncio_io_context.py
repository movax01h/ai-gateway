"""Tests for duo_workflow_service.asyncio_io_context."""

import asyncio
import contextvars
import gc
import socket
import weakref
from typing import Any
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.asyncio_io_context import (
    _MARKER,
    install_clean_io_handle_context,
)

request_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "request_var", default=None
)
startup_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "startup_var", default=None
)


class _Payload:  # plain object() cannot be weak-referenced
    pass


@pytest.fixture(name="socket_pair")
def socket_pair_fixture():
    left, right = socket.socketpair()
    left.setblocking(False)
    right.setblocking(False)
    yield left, right
    left.close()
    right.close()


def _register_reader_under_request_context(
    loop: asyncio.AbstractEventLoop, fd: int, callback
) -> weakref.ref:
    refs: list[weakref.ref] = []

    def register():
        payload = _Payload()
        refs.append(weakref.ref(payload))
        request_var.set(payload)
        loop.add_reader(fd, callback)

    contextvars.copy_context().run(register)
    return refs[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True], ids=["without_fix", "with_fix"])
async def test_reader_handle_retains_registration_context_only_without_fix(
    installed, socket_pair
):
    loop = asyncio.get_running_loop()
    if installed:
        assert install_clean_io_handle_context(loop) is True
    left, _ = socket_pair

    payload_ref = _register_reader_under_request_context(
        loop, left.fileno(), lambda: None
    )
    try:
        gc.collect()
        if installed:
            assert payload_ref() is None
        else:
            assert payload_ref() is not None
    finally:
        loop.remove_reader(left.fileno())

    gc.collect()
    assert payload_ref() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True], ids=["without_fix", "with_fix"])
async def test_transport_resume_reading_does_not_retain_context_with_fix(
    installed, socket_pair
):
    loop = asyncio.get_running_loop()
    if installed:
        install_clean_io_handle_context(loop)
    left, _ = socket_pair

    transport, _ = await loop.create_connection(asyncio.Protocol, sock=left)
    refs: list[weakref.ref] = []

    def resume_under_request_context():
        payload = _Payload()
        refs.append(weakref.ref(payload))
        request_var.set(payload)
        transport.pause_reading()
        transport.resume_reading()

    try:
        contextvars.copy_context().run(resume_under_request_context)
        gc.collect()
        assert (refs[0]() is None) is installed
    finally:
        transport.close()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True], ids=["without_fix", "with_fix"])
async def test_timer_armed_inside_reader_callback_retains_context_only_without_fix(
    installed, socket_pair
):
    """A timer armed inside an fd callback copies that callback's context (aiohttp's keep-alive cleanup does this)."""
    loop = asyncio.get_running_loop()
    if installed:
        install_clean_io_handle_context(loop)
    left, right = socket_pair
    timers: list[asyncio.TimerHandle] = []
    armed = asyncio.Event()

    def on_readable():
        loop.remove_reader(left.fileno())
        timers.append(loop.call_later(3600, lambda: None))
        armed.set()

    payload_ref = _register_reader_under_request_context(
        loop, left.fileno(), on_readable
    )
    right.send(b"x")
    await asyncio.wait_for(armed.wait(), timeout=5)
    try:
        gc.collect()
        assert (payload_ref() is None) is installed
    finally:
        timers[0].cancel()


@pytest.mark.asyncio
async def test_coroutine_and_done_callback_contexts_are_unchanged(socket_pair):
    loop = asyncio.get_running_loop()
    install_clean_io_handle_context(loop)
    left, right = socket_pair

    async def receiver():
        request_var.set("task-value")
        data = await loop.sock_recv(left, 1)
        return data, request_var.get()

    task = asyncio.create_task(receiver())
    await asyncio.sleep(0)
    right.send(b"x")
    assert await task == (b"x", "task-value")

    future: asyncio.Future = loop.create_future()
    seen: list[Any] = []

    def register_done_callback():
        request_var.set("callback-value")
        future.add_done_callback(lambda _: seen.append(request_var.get()))

    contextvars.copy_context().run(register_done_callback)

    def on_readable():
        loop.remove_reader(left.fileno())
        future.set_result(True)

    loop.add_reader(left.fileno(), on_readable)
    right.send(b"x")
    await future
    await asyncio.sleep(0)
    assert seen == ["callback-value"]


@pytest.mark.asyncio
async def test_reader_callback_runs_in_startup_context_and_forwards_args(
    socket_pair,
):
    loop = asyncio.get_running_loop()
    left, right = socket_pair
    token = startup_var.set("from-startup")
    try:
        install_clean_io_handle_context(loop)
        seen: dict[str, Any] = {}
        readable = asyncio.Event()

        def on_readable(tag):
            seen.update(tag=tag, startup=startup_var.get(), request=request_var.get())
            loop.remove_reader(left.fileno())
            readable.set()

        def register():
            request_var.set("request-value")
            return loop._add_reader(left.fileno(), on_readable, "arg1")

        handle = contextvars.copy_context().run(register)
        assert isinstance(handle, asyncio.Handle)

        right.send(b"x")
        await asyncio.wait_for(readable.wait(), timeout=5)
        assert seen == {"tag": "arg1", "startup": "from-startup", "request": None}
    finally:
        startup_var.reset(token)


@pytest.mark.asyncio
async def test_install_is_idempotent():
    loop = asyncio.get_running_loop()
    assert install_clean_io_handle_context(loop) is True
    add_reader, add_writer = loop._add_reader, loop._add_writer

    assert install_clean_io_handle_context(loop) is True
    assert loop._add_reader is add_reader
    assert loop._add_writer is add_writer
    assert getattr(loop, _MARKER) is True


def test_install_skips_loops_without_private_registration_methods():
    foreign_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    with capture_logs() as logs:
        assert install_clean_io_handle_context(foreign_loop) is False
    assert not hasattr(foreign_loop, _MARKER)
    assert logs[0]["log_level"] == "warning"

    selector_loop = asyncio.new_event_loop()
    try:
        selector_loop._add_writer = None
        assert install_clean_io_handle_context(selector_loop) is False
        assert "_add_reader" not in vars(selector_loop)
        assert not hasattr(selector_loop, _MARKER)
    finally:
        selector_loop.close()
