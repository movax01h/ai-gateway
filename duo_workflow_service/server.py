import asyncio
import json
import os
from typing import AsyncIterable, AsyncIterator, Type

import grpc
import structlog
from dotenv import load_dotenv
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    TokenAuthority,
)
from grpc_reflection.v1alpha import reflection
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from contract import contract_pb2, contract_pb2_grpc
from duo_workflow_service.interceptors.authentication_interceptor import (
    AuthenticationInterceptor,
    current_user,
)
from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdInterceptor,
)
from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from duo_workflow_service.interceptors.monitoring_interceptor import (
    MonitoringInterceptor,
)
from duo_workflow_service.internal_events.client import DuoWorkflowInternalEvent
from duo_workflow_service.llm_factory import validate_llm_access
from duo_workflow_service.monitoring import setup_monitoring
from duo_workflow_service.profiling import setup_profiling
from duo_workflow_service.structured_logging import set_workflow_id, setup_logging
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.tracking.sentry_error_tracking import setup_error_tracking
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import Registry

log = structlog.stdlib.get_logger("server")


class DuoWorkflowService(contract_pb2_grpc.DuoWorkflowServicer):
    OUTBOX_CHECK_INTERVAL = 0.5

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

        # Fetch the start workflow call
        start_workflow_request: contract_pb2.ClientEvent = await anext(
            aiter(request_iterator)
        )
        workflow_id = start_workflow_request.startRequest.workflowID
        set_workflow_id(workflow_id)
        log.info("Starting workflow %s", start_workflow_request)
        goal = start_workflow_request.startRequest.goal
        workflow_definition = start_workflow_request.startRequest.workflowDefinition
        workflow_metadata = {}
        if start_workflow_request.startRequest.workflowMetadata:
            workflow_metadata = json.loads(
                start_workflow_request.startRequest.workflowMetadata
            )
        workflow_class: Type[AbstractWorkflow] = Registry.resolve(workflow_definition)
        workflow: AbstractWorkflow = workflow_class(
            workflow_id=workflow_id,
            workflow_metadata=workflow_metadata,
        )

        async def send_events():
            while not workflow.is_done:
                try:
                    action = workflow.get_from_outbox()

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

                except asyncio.QueueEmpty as err:
                    log.debug("Queue is empty, waiting", err=err)
                    await asyncio.sleep(self.OUTBOX_CHECK_INTERVAL)

        try:
            workflow_task = asyncio.create_task(workflow.run(goal))
            async for action in send_events():
                yield action

            await workflow_task
        except BaseException as err:
            log_exception(err, extra={"workflow_id": workflow_id})
            workflow_task.cancel(
                f"Terminated workflow {workflow_id} execution due to an {type(err).__name__}: {err}"
            )

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


async def serve(port: int) -> None:
    """
    grpc.keepalive_time_ms: The period (in milliseconds) after which a keepalive ping is
        sent on the transport.
    grpc.keepalive_timeout_ms: The amount of time (in milliseconds) the sender of the keepalive
        ping waits for an acknowledgement. If it does not receive an acknowledgement within
        this time, it will close the connection.
    grpc.http2.min_ping_interval_without_data_ms: Minimum allowed time (in milliseconds)
        between a server receiving successive ping frames without sending any data/header frame.
    grpc.keepalive_permit_without_calls: If set to 1 (0 : false; 1 : true), allows keepalive
        pings to be sent even if there are no calls in flight.
    For more details, check: https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    """
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
            InternalEventsInterceptor(),
            MonitoringInterceptor(),
        ],
        options=server_options,
    )
    contract_pb2_grpc.add_DuoWorkflowServicer_to_server(DuoWorkflowService(), server)
    server.add_insecure_port(f"[::]:{port}")
    # enable reflection for faster local development and debugging
    # this can be removed when we are closer to production
    service_names = (
        contract_pb2.DESCRIPTOR.services_by_name["DuoWorkflow"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    log.info("Starting server on port %d", port)
    await server.start()
    log.info("Started server")
    await server.wait_for_termination()


def configure_cache() -> None:
    if os.environ.get("LLM_CACHE") == "true":
        set_llm_cache(SQLiteCache(database_path=".llm_cache.db"))

def setup_cloud_connector():
    cloud_connector_service_name = os.environ.get("DUO_WORKFLOW_CLOUD_CONNECTOR_SERVICE_NAME", "gitlab-duo-workflow-service")
    os.environ["CLOUD_CONNECTOR_SERVICE_NAME"] = cloud_connector_service_name

def run():
    load_dotenv()
    setup_cloud_connector()
    setup_profiling()
    setup_error_tracking()
    setup_monitoring()
    setup_logging(json_format=True, to_file=None)
    configure_cache()
    validate_llm_access()
    DuoWorkflowInternalEvent.setup()
    port = int(os.environ.get("PORT", "50052"))
    asyncio.get_event_loop().run_until_complete(serve(port))


if __name__ == "__main__":
    run()
