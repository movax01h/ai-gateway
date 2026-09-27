# pylint: disable=file-naming-for-tests
"""Integration test: a read_file image response becomes an Anthropic image block.

Drives the CLI's own registry flow (developer/2.0.0-interactive) through the
real gRPC server with a FakeExecutor answering ``runReadFile`` with the typed
``ActionResponse.imageResponse`` the client executor emits for image files. The model is
scripted at the Anthropic SDK boundary: the first call returns a read_file
tool use, the second call captures the request payload. The assertion is the
whole point of the perception path: the tool_result reaching the SDK contains
a real base64 image block, byte-identical to the "file" the executor served.

The shared container fixture wires Anthropic to FakeModel under
``mock_model_responses``, so this module builds its own container with real
model wiring; the SDK method itself is patched, so no network I/O happens.
"""

import base64
import hashlib
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import grpc
import pytest
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.messages import AsyncMessages
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

import tests.duo_workflow_service.integration.conftest as integration_conftest
from contract import contract_pb2
from duo_workflow_service.interceptors import X_GITLAB_VERSION_HEADER
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    FeatureFlagInterceptor,
)
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from duo_workflow_service.server import DuoWorkflowService
from tests.duo_workflow_service.integration.conftest import (
    FakeExecutor,
    run_exchange,
    start_registry_flow_event,
)

# Kept deliberately tiny: this payload ends up in every assertion context and
# CI log line on failure, so the canary carries no more bytes than the
# assertions need.
FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"deterministic canary payload" * 2
FAKE_PNG_BASE64 = base64.b64encode(FAKE_PNG_BYTES).decode()


@pytest.fixture(name="mock_duo_workflow_service_container", scope="module")
def real_model_container_fixture():
    """Rebuild the DI container with real model wiring (no FakeModel).

    Same construction as the shared fixture in tests/duo_workflow_service/
    conftest.py, but with ``mock_model_responses=False`` so the Anthropic
    provider resolves to the real ChatAnthropic factory.
    """
    # pylint: disable=import-outside-toplevel
    from ai_gateway.config import Config
    from ai_gateway.container import ContainerApplication
    from tests.duo_workflow_service.conftest import CONTAINER_APPLICATION_PACKAGES

    with (
        patch("ai_gateway.models.base.PredictionServiceAsyncClient"),
        patch("ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"),
        patch(
            "ai_gateway.models.v2.container.connect_google_gen_vertex_ai",
            return_value=None,
        ),
    ):
        config = Config(
            _env_file=None, _env_prefix="AIGW_TEST", mock_model_responses=False
        )
        container = ContainerApplication()
        container.config.from_dict(config.model_dump())
        container.wire(packages=CONTAINER_APPLICATION_PACKAGES)
        try:
            yield container
        finally:
            # Wiring is process-global: without this, the real-model container
            # leaks into sibling test modules that run after this one in the
            # same worker (order-dependent failures under xdist + randomly).
            container.unwire()


@pytest.fixture(name="agent_privileges_names")
def agent_privileges_names_fixture() -> list[str]:
    # The developer flow's read_file tool is gated on this Rails-granted
    # privilege; the shared fixture default is [].
    return ["read_only_files"]


@pytest.fixture(name="workflow_config")
def workflow_config_fixture(  # pylint: disable=too-many-arguments
    workflow_id: str,
    agent_privileges_names: list[str],
    allow_agent_to_request_user: bool,
    mcp_enabled: bool,
    first_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    # Mirrors the shared fixture, but pre-approves the read-only privilege so
    # tool approval short-circuits before its GitLab session-approval GraphQL
    # call (which FakeExecutor has no business answering).
    return {
        "workflow_id": workflow_id,
        "project_id": 1,
        "agent_privileges_names": agent_privileges_names,
        "pre_approved_agent_privileges_names": ["read_only_files"],
        "allow_agent_to_request_user": allow_agent_to_request_user,
        "mcp_enabled": mcp_enabled,
        "first_checkpoint": first_checkpoint,
        "latest_checkpoint": None,
        "workflow_status": "",
        "gitlab_host": "gitlab.com",
        "archived": False,
        "stalled": False,
    }


@pytest.fixture(autouse=True)
def anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def anthropic_model_metadata(monkeypatch):
    monkeypatch.setattr(
        integration_conftest,
        "MODEL_METADATA_HEADER",
        (
            ModelMetadataInterceptor.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA,
            # The header shape Rails sends for a user-selected model: pins the
            # anthropic-direct claude_sonnet_4_6 (zero-dialect path).
            json.dumps(
                {
                    "provider": "gitlab",
                    "feature_setting": "duo_agent_platform",
                    "identifier": "claude_sonnet_4_6",
                }
            ),
        ),
    )


class ImageFakeExecutor(FakeExecutor):
    """FakeExecutor that also answers runReadFile with a typed image response."""

    HANDLED_ACTIONS = ("runHTTPRequest", "runReadFile")

    def response_for(self, action: contract_pb2.Action) -> contract_pb2.ClientEvent:
        if action.HasField("runReadFile"):
            return contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    requestID=action.requestID,
                    imageResponse=contract_pb2.ImageResponse(
                        mime_type="image/png", data=FAKE_PNG_BYTES
                    ),
                )
            )
        return super().response_for(action)


async def _tool_use_event_stream():
    """The raw Anthropic event stream for one read_file tool call."""
    yield RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_scripted_1",
            type="message",
            role="assistant",
            model="claude-sonnet-4-6",
            stop_reason=None,
            usage=Usage(input_tokens=10, output_tokens=1),
            content=[],
        ),
    )
    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=ToolUseBlock(
            type="tool_use", id="toolu_scripted_1", name="read_file", input={}
        ),
    )
    yield RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=InputJSONDelta(
            type="input_json_delta",
            partial_json=json.dumps({"file_path": "./screenshot.png"}),
        ),
    )
    yield RawContentBlockStopEvent(type="content_block_stop", index=0)
    yield RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason="tool_use", stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=5),
    )
    yield RawMessageStopEvent(type="message_stop")


async def _final_answer_event_stream():
    """A clean end-of-turn answer, so the flow parks on its human-input interrupt and the server closes the stream
    normally.

    Erroring out of the second call instead would work, but the failed workflow's teardown logs the whole state as rich
    tracebacks — slow enough to trip the exchange timeout on a loaded CI runner, and megabytes of captured output when
    the test is red.
    """
    yield RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_scripted_2",
            type="message",
            role="assistant",
            model="claude-sonnet-4-6",
            stop_reason=None,
            usage=Usage(input_tokens=10, output_tokens=1),
            content=[],
        ),
    )
    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=TextBlock(type="text", text=""),
    )
    yield RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=TextDelta(type="text_delta", text="The image shows the canary."),
    )
    yield RawContentBlockStopEvent(type="content_block_stop", index=0)
    yield RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason="end_turn", stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=5),
    )
    yield RawMessageStopEvent(type="message_stop")


def _message_shapes(messages: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    """Role + content-block types per message, for failure output.

    Payloads (base64 image data, prompt text) stay out of assertion messages so a red run never sprays them into the CI
    log.
    """
    shapes: list[tuple[Any, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            block_types = [
                block.get("type") if isinstance(block, dict) else type(block).__name__
                for block in content
            ]
            shapes.append((message.get("role"), block_types))
        else:
            shapes.append((message.get("role"), type(content).__name__))
    return shapes


def _iter_strings(value: Any) -> Iterator[str]:
    """Every string anywhere inside a nested payload."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _find_tool_result(messages: list[dict[str, Any]]) -> Any:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") == "toolu_scripted_1"
            ):
                return block
    return None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
async def test_read_file_image_reaches_anthropic_as_image_block(
    servicer: DuoWorkflowService,
):
    executor = ImageFakeExecutor(integration_conftest.WORKFLOW_ID)

    calls: list[dict[str, Any]] = []

    async def scripted_model(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _tool_use_event_stream()
        if len(calls) == 2:
            return _final_answer_event_stream()
        # Guard, not a flow step: nothing after the end-of-turn answer should
        # reach the SDK, and a raise here can never escape to the network.
        raise RuntimeError("unexpected third model call")

    with (
        # The DWS Anthropic factory passes `betas=[...]`, so langchain-anthropic
        # routes through the beta messages surface; patch both to be robust.
        patch.object(
            AsyncBetaMessages,
            "create",
            new_callable=AsyncMock,
            side_effect=scripted_model,
        ),
        patch.object(
            AsyncMessages, "create", new_callable=AsyncMock, side_effect=scripted_model
        ),
    ):
        exchange = await run_exchange(
            servicer,
            executor,
            start_registry_flow_event(
                goal="Read ./screenshot.png and describe it.",
                flow_config_id="developer",
                schema_version="v1",
                version="2.0.0-interactive",
                # The client half of the switch: declared by a client whose
                # executor answers read_file with an image, as the real one
                # will in its first image-capable release.
                client_capabilities=("read_file_image",),
            ),
            # The instance half: the flag arrives the way production sends it,
            # via the request header. The version header is what lets declared
            # capabilities count at all (is_client_capable ignores them below
            # GitLab 18.7, the first Workhorse that forwards them).
            extra_metadata=(
                (
                    FeatureFlagInterceptor.X_GITLAB_ENABLED_FEATURE_FLAGS,
                    "dap_tool_image_input",
                ),
                (X_GITLAB_VERSION_HEADER, "19.5.0"),
            ),
        )

    assert exchange.code == grpc.StatusCode.OK, exchange.details

    assert len(calls) == 2, (
        f"expected exactly two model calls (tool use, then final answer), got "
        f"{len(calls)}; actions seen: "
        f"{[a.WhichOneof('action') for a in executor.actions]}"
    )

    tool_result = _find_tool_result(calls[1]["messages"])
    assert tool_result is not None, (
        f"no tool_result in: {_message_shapes(calls[1]['messages'])}"
    )

    blocks = tool_result["content"]
    assert isinstance(blocks, list), f"tool_result content not a list: {type(blocks)}"

    block_types = [b.get("type") for b in blocks]
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1, f"expected exactly one image block in: {block_types}"
    source = image_blocks[0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    # Digest compare so a mismatch prints two hashes, not two base64 payloads.
    assert (
        hashlib.sha256(source["data"].encode()).hexdigest()
        == hashlib.sha256(FAKE_PNG_BASE64.encode()).hexdigest()
    ), "image bytes not identical end to end"

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert any("Read image file" in b["text"] for b in text_blocks)

    # The base64 payload must appear ONLY inside the image block's source.data,
    # never as plain text anywhere else in the payload. Walk the structure
    # rather than serializing it, so the one legitimate occurrence can be
    # excluded precisely.
    leaks = [
        text
        for text in _iter_strings(calls[1]["messages"])
        if FAKE_PNG_BASE64 in text and text != source["data"]
    ]
    if leaks:
        pytest.fail("the image base64 leaked into the model payload as plain text")
