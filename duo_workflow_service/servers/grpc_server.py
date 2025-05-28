# pylint: disable=direct-environment-variable-reference

import asyncio
import json
import os
from typing import AsyncIterable, AsyncIterator

import aiohttp
import grpc
import structlog
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    TokenAuthority,
)
from grpc_reflection.v1alpha import reflection

from contract import contract_pb2, contract_pb2_grpc
from duo_workflow_service.gitlab.connection_pool import connection_pool
from duo_workflow_service.interceptors.authentication_interceptor import (
    AuthenticationInterceptor,
    current_user,
)
from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdInterceptor,
)
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    FeatureFlagInterceptor,
)
from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from duo_workflow_service.interceptors.monitoring_interceptor import (
    MonitoringInterceptor,
)
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.structured_logging import set_workflow_id
from duo_workflow_service.tracking import MonitoringContext, current_monitoring_context
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TypeWorkflow,
)
from duo_workflow_service.workflows.registry import resolve_workflow_class

log = structlog.stdlib.get_logger("server")


def string_to_category_enum(category_string: str) -> CategoryEnum:
    try:
        return CategoryEnum(category_string)
    except ValueError:
        # Handle case when string doesn't match any enum value
        # We will return default workflow type
        # Since it isn't a blocker for workflow run
        log.warning(f"Unknown category string: {category_string}")
        return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


def clean_start_request(start_workflow_request: contract_pb2.ClientEvent):
    request = contract_pb2.ClientEvent()
    request.CopyFrom(start_workflow_request)
    request.startRequest.ClearField("workflowMetadata")
    return request


class GrpcServer(contract_pb2_grpc.DuoWorkflowServicer):
    # Set to 2 seconds to provide a reasonable balance between:
    # - Giving tasks enough time to properly clean up resources
    # - Not delaying the server response for too long when handling errors
    TASK_CANCELLATION_TIMEOUT = 2.0

    # pylint: disable=invalid-overridden-method
    async def ExecuteWorkflow(
        self,
        request_iterator: AsyncIterable[contract_pb2.ClientEvent],
        context: grpc.ServicerContext,
    ) -> AsyncIterator[contract_pb2.Action]:
        user: CloudConnectorUser = current_user.get()

        if not user.can(GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW):
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to execute workflow"
            )

        monitoring_context: MonitoringContext = current_monitoring_context.get()

        # Fetch the start workflow call
        start_workflow_request: contract_pb2.ClientEvent = await anext(
            aiter(request_iterator)
        )
        workflow_id = start_workflow_request.startRequest.workflowID
        set_workflow_id(workflow_id)
        log.info("Starting workflow %s", clean_start_request(start_workflow_request))

        goal = start_workflow_request.startRequest.goal
        workflow_metadata = {}
        workflow_definition = start_workflow_request.startRequest.workflowDefinition
        monitoring_context.workflow_id = workflow_id
        monitoring_context.workflow_definition = workflow_definition
        if start_workflow_request.startRequest.workflowMetadata:
            workflow_metadata = json.loads(
                start_workflow_request.startRequest.workflowMetadata
            )

        context_elements = []
        if start_workflow_request.startRequest.context:
            context_elements = list(start_workflow_request.startRequest.context)

        workflow_type = string_to_category_enum(workflow_definition)
        workflow_class: TypeWorkflow = resolve_workflow_class(workflow_definition)

        invocation_metadata = dict(context.invocation_metadata())

        workflow: AbstractWorkflow = workflow_class(
            workflow_id=workflow_id,
            workflow_metadata=workflow_metadata,
            workflow_type=workflow_type,
            context_elements=context_elements,
            invocation_metadata={
                "base_url": invocation_metadata.get("x-gitlab-base-url", ""),
                "gitlab_token": invocation_metadata.get("x-gitlab-oauth-token", ""),
            },
        )

        async def send_events():
            while not workflow.is_done:
                try:
                    streaming_action = workflow.get_from_streaming_outbox()
                    if isinstance(streaming_action, contract_pb2.Action):
                        yield streaming_action
                        _event: contract_pb2.ClientEvent = await anext(  # noqa: F841
                            aiter(request_iterator)
                        )

                        if workflow.outbox_empty():
                            continue

                    action = await workflow.get_from_outbox()

                    if isinstance(action, contract_pb2.Action):
                        log.info(
                            "Read action from the egress queue",
                            requestID=action.requestID,
                            action_class=action.WhichOneof("action"),
                        )

                    yield action

                    event: contract_pb2.ClientEvent = await anext(
                        aiter(request_iterator)
                    )

                    workflow.add_to_inbox(event)
                    if (
                        isinstance(event, contract_pb2.ClientEvent)
                        and event.actionResponse
                    ):
                        log.info(
                            "Wrote ClientEvent into the ingres queue",
                            requestID=event.actionResponse.requestID,
                        )
                except TimeoutError as err:
                    log.debug("Timeout on reading from queue, trying again", err=err)

        workflow_task = None
        try:
            workflow_task = asyncio.create_task(workflow.run(goal))
            async for action in send_events():
                yield action

            await workflow_task
        except BaseException as err:
            log_exception(err, extra={"workflow_id": workflow_id})
            if workflow_task and not workflow_task.done():
                workflow_task.cancel(
                    f"Terminated workflow {workflow_id} execution due to an {type(err).__name__}: {err}"
                )
                # https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel
                # The asyncio documentation states that cancel() only "arranges for a CancelledError
                # to be thrown into the wrapped coroutine on the next cycle through the event loop."
                # By awaiting the task after cancellation, the code now allows the event loop to
                # complete that cycle and properly handle the cancellation before proceeding
                try:
                    await asyncio.wait_for(
                        workflow_task, timeout=self.TASK_CANCELLATION_TIMEOUT
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            await context.abort(grpc.StatusCode.INTERNAL, "Something went wrong")
        finally:
            await workflow.cleanup(workflow_id)

    async def GenerateToken(
        self, request: contract_pb2.GenerateTokenRequest, context: grpc.ServicerContext
    ) -> contract_pb2.GenerateTokenResponse:
        user: CloudConnectorUser = current_user.get()

        if not user.can(
            GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW,
            disallowed_issuers=[CloudConnectorConfig().service_name],
        ):
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
            )

        metadata = dict(context.invocation_metadata())
        global_user_id = metadata.get("x-gitlab-global-user-id")
        gitlab_realm = metadata.get("x-gitlab-realm")
        gitlab_instance_id = metadata.get("x-gitlab-instance-id")

        token_authority = TokenAuthority(
            os.environ.get("DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY")
        )
        token, expires_at = token_authority.encode(
            global_user_id,
            gitlab_realm,
            user,
            gitlab_instance_id,
            [GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW],
        )

        return contract_pb2.GenerateTokenResponse(token=token, expiresAt=expires_at)

    # pylint: enable=invalid-overridden-method


async def grpc_serve(port: int) -> None:
    """grpc.keepalive_time_ms: The period (in milliseconds) after which a keepalive ping is sent on the transport.

    grpc.keepalive_timeout_ms: The amount of time (in milliseconds) the sender of the keepalive     ping waits for an
    acknowledgement. If it does not receive an acknowledgement within     this time, it will close the connection.
    grpc.http2.min_ping_interval_without_data_ms: Minimum allowed time (in milliseconds)     between a server receiving
    successive ping frames without sending any data/header frame. grpc.keepalive_permit_without_calls: If set to 1 (0 :
    false; 1 : true), allows keepalive     pings to be sent even if there are no calls in flight. For more details,
    check:
    https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    """
    connection_pool.set_options(
        pool_size=100,  # Adjust based on your needs
        timeout=aiohttp.ClientTimeout(total=30),
    )
    async with connection_pool:
        server_options = [
            ("grpc.keepalive_time_ms", 20 * 1000),
            ("grpc.http2.min_ping_interval_without_data_ms", 10 * 1000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.so_reuseport", 0),
        ]

        server = grpc.aio.server(
            interceptors=[
                CorrelationIdInterceptor(),
                AuthenticationInterceptor(),
                FeatureFlagInterceptor(),
                InternalEventsInterceptor(),
                MonitoringInterceptor(),
            ],
            options=server_options,
        )
        contract_pb2_grpc.add_DuoWorkflowServicer_to_server(GrpcServer(), server)
        server.add_insecure_port(f"[::]:{port}")
        # enable reflection for faster local development and debugging
        # this can be removed when we are closer to production
        service_names = (
            contract_pb2.DESCRIPTOR.services_by_name["DuoWorkflow"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        log.info("Starting gRPC server on port %d", port)
        await server.start()
        log.info("Started server")
        await server.wait_for_termination()
