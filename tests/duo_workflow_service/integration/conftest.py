"""Shared harness for the Duo Workflow Service integration tests.

These tests exercise a flow the way a client does: a real ``grpc.aio`` server serving the real
``DuoWorkflowService``, a real ``DuoWorkflowStub`` on the other end of a socket, and a real LangGraph graph
compiled from a real flow config. Nothing about the flow itself is mocked, so they cover the seams unit tests
stub out — flow request normalization, flow resolution, config validation, component and router wiring, prompt
rendering, the checkpoint notifier, and the outbox/Action protocol.

Two things stand in for the outside world:

* **The LLM.** The DI container runs with ``mock_model_responses=True``, so every prompt resolves to
    :class:`ai_gateway.models.mock.FakeModel`, which answers ``"mock"`` without any network call.
* **The executor.** :class:`FakeExecutor` plays the client half of the bidirectional stream, answering
    ``runHTTPRequest`` actions with canned GitLab REST responses the same way a real executor would.

The GraphQL bootstrap (``fetch_workflow_and_container_data``) is the one call still patched out, via the shared
``mock_fetch_workflow_and_container_data`` fixture, because faking it means hand-rolling the entire workflow
GraphQL document.
"""

import asyncio
import json
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from contract import contract_pb2, contract_pb2_grpc
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.metadata_context_interceptor import (
    MetadataContextInterceptor,
)
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from duo_workflow_service.server import DuoWorkflowService

WORKFLOW_ID = "42"

# Flow configs under test declare no model of their own, so — exactly as in production — the model comes from
# the feature setting named in this header, resolved by ModelMetadataInterceptor.
MODEL_METADATA_HEADER = (
    ModelMetadataInterceptor.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA,
    json.dumps(
        {"provider": "gitlab", "feature_setting": "duo_agent_platform_agentic_chat"}
    ),
)

# How long the whole exchange may take before we call it a hang. The flow does no real I/O,
# so this only ever trips on a genuine deadlock.
EXCHANGE_TIMEOUT_SECONDS = 30


class FakeExecutor:
    """The client half of the ExecuteWorkflow stream.

    Answers ``runHTTPRequest`` actions with canned GitLab REST responses and records every action the server
    sent, so tests can assert on the conversation rather than on internal state.
    """

    def __init__(self, workflow_id: str):
        self._workflow_id = workflow_id
        self.actions: list[contract_pb2.Action] = []
        self.requested_paths: list[tuple[str, str]] = []

    def response_for(self, action: contract_pb2.Action) -> contract_pb2.ClientEvent:
        """Build the ActionResponse a real executor would return for `action`."""
        request = action.runHTTPRequest
        self.requested_paths.append((request.method, request.path))

        status_code, body = self._http_body(request)

        return contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(
                requestID=action.requestID,
                httpResponse=contract_pb2.HttpResponse(
                    statusCode=status_code,
                    body=json.dumps(body),
                ),
            )
        )

    def _http_body(self, request: contract_pb2.RunHTTPRequest) -> tuple[int, Any]:
        workflow_path = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}"

        # Checkpoint reads return an empty list: this is a brand new session.
        if request.path.startswith(f"{workflow_path}/checkpoints"):
            if request.method == "GET":
                return 200, []
            return 200, {}

        if request.path.startswith(workflow_path):
            # Rails reports the workflow as running; status transitions are PATCHed back.
            if request.method == "GET":
                return 200, {"id": self._workflow_id, "status": "running"}
            return 200, {}

        raise AssertionError(
            f"FakeExecutor received an unexpected request: {request.method} {request.path}"
        )

    def checkpoints(self) -> list[contract_pb2.NewCheckpoint]:
        return [
            action.newCheckpoint
            for action in self.actions
            if action.HasField("newCheckpoint")
        ]

    def agent_messages(self) -> list[str]:
        """Every agent message the flow streamed to the UI chat log, oldest first."""
        return [
            message["content"]
            for checkpoint in self.checkpoints()
            for message in json.loads(checkpoint.checkpoint)["channel_values"][
                "ui_chat_log"
            ]
            if message["message_type"] == "agent"
        ]


def start_registry_flow_event(
    goal: str, flow_config_id: str, schema_version: str, version: str
) -> contract_pb2.ClientEvent:
    """A start request that names a flow from the server-side registry."""
    return contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            workflowID=WORKFLOW_ID,
            goal=goal,
            flowConfigId=flow_config_id,
            flowConfigSchemaVersion=schema_version,
            flowVersion=version,
        )
    )


def start_inline_config_event(
    goal: str, workflow_definition: str, flow_config: dict[str, Any]
) -> contract_pb2.ClientEvent:
    """A start request that ships its own flow config instead of naming a registry flow."""
    return contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            workflowID=WORKFLOW_ID,
            goal=goal,
            workflowDefinition=workflow_definition,
            flowConfig=ParseDict(flow_config, Struct()),
            flowConfigSchemaVersion="v1",
        )
    )


class Exchange(NamedTuple):
    """How the server closed the RPC."""

    code: grpc.StatusCode
    details: str


async def run_exchange(
    servicer: DuoWorkflowService,
    executor: FakeExecutor,
    start_event: contract_pb2.ClientEvent,
) -> Exchange:
    """Drive one full ExecuteWorkflow RPC against a real gRPC server.

    Returns the terminal status of the call, whether the server closed the stream normally or aborted it.
    """
    server = grpc.aio.server(
        interceptors=[
            MetadataContextInterceptor(MagicMock()),
            ModelMetadataInterceptor(),
        ]
    )
    contract_pb2_grpc.add_DuoWorkflowServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    await server.start()

    # Responses the executor wants to send back, produced while reading the action stream.
    client_events: asyncio.Queue[contract_pb2.ClientEvent] = asyncio.Queue()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")

    async def request_iterator():
        yield start_event
        while True:
            yield await client_events.get()

    try:
        stub = contract_pb2_grpc.DuoWorkflowStub(channel)
        call = stub.ExecuteWorkflow(
            request_iterator(), metadata=[MODEL_METADATA_HEADER]
        )

        async def consume_actions():
            async for action in call:
                executor.actions.append(action)
                if action.HasField("runHTTPRequest"):
                    client_events.put_nowait(executor.response_for(action))

        try:
            await asyncio.wait_for(consume_actions(), timeout=EXCHANGE_TIMEOUT_SECONDS)
        except grpc.aio.AioRpcError as error:
            # The server rejected the request outright (context.abort), so no actions were streamed.
            return Exchange(error.code(), error.details() or "")

        return Exchange(await call.code(), await call.details())
    finally:
        await channel.close()
        await server.stop(grace=None)


@pytest.fixture(name="scopes")
def scopes_fixture() -> list[str]:
    # `duo_chat` is needed alongside `duo_agent_platform` because the unit primitive is chosen from the
    # workflow definition: a `chat` definition demands DUO_CHAT. Granting both keeps the authorization gate
    # out of the way so these tests exercise flow resolution rather than permissions.
    return ["duo_agent_platform", "duo_chat"]


@pytest.fixture(name="servicer")
def servicer_fixture(auth_user) -> DuoWorkflowService:
    current_user.set(auth_user)

    return DuoWorkflowService()


@pytest.fixture(name="executor")
def executor_fixture() -> FakeExecutor:
    return FakeExecutor(WORKFLOW_ID)


@pytest.fixture(autouse=True)
def stub_usage_quota(mock_duo_workflow_service_container):
    """Satisfy the `has_sufficient_usage_quota` decorator on ExecuteWorkflow."""
    service = MagicMock()
    service.execute = AsyncMock()
    service.aclose = AsyncMock()

    mock_duo_workflow_service_container.usage_quota.service.override(service)

    yield service

    mock_duo_workflow_service_container.usage_quota.service.reset_override()
