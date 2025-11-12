# pylint: disable=direct-environment-variable-reference

import asyncio
import functools
import json
import os
import signal
from itertools import chain
from typing import AsyncIterable, AsyncIterator, Optional

import aiohttp
import grpc
import structlog
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    TokenAuthority,
    data_model,
)
from google.protobuf.struct_pb2 import Struct
from grpc_reflection.v1alpha import reflection
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.utils.function_calling import convert_to_openai_tool

import duo_workflow_service.workflows.registry as flow_registry
from ai_gateway.app import get_config
from ai_gateway.config import Config, setup_litellm
from ai_gateway.container import ContainerApplication
from contract import contract_pb2, contract_pb2_grpc
from duo_workflow_service.components import tools_registry
from duo_workflow_service.executor.outbox import OutboxSignal
from duo_workflow_service.gitlab.connection_pool import connection_pool
from duo_workflow_service.interceptors.authentication_interceptor import (
    AuthenticationInterceptor,
)
from duo_workflow_service.interceptors.authentication_interceptor import (
    current_user as current_user_context_var,
)
from duo_workflow_service.interceptors.client_type_interceptor import (
    ClientTypeInterceptor,
)
from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdInterceptor,
)
from duo_workflow_service.interceptors.enabled_instance_verbose_ai_logs_interceptor import (
    EnabledInstanceVerboseAiLogsInterceptor,
)
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    FeatureFlagInterceptor,
)
from duo_workflow_service.interceptors.gitlab_realm_interceptor import (
    GitLabRealmInterceptor,
)
from duo_workflow_service.interceptors.gitlab_version_interceptor import (
    GitLabVersionInterceptor,
)
from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from duo_workflow_service.interceptors.language_server_version_interceptor import (
    LanguageServerVersionInterceptor,
    language_server_version,
)
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from duo_workflow_service.interceptors.monitoring_interceptor import (
    MonitoringInterceptor,
)
from duo_workflow_service.interceptors.prompt_caching_interceptor import (
    PromptCachingInterceptor,
)
from duo_workflow_service.llm_factory import validate_llm_access
from duo_workflow_service.monitoring import duo_workflow_metrics, setup_monitoring
from duo_workflow_service.profiling import setup_profiling
from duo_workflow_service.structured_logging import set_workflow_id, setup_logging
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tracking import MonitoringContext, current_monitoring_context
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.tracking.sentry_error_tracking import setup_error_tracking
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import FlowFactory, resolve_workflow_class
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
    OUTGOING_MESSAGE_TOO_LARGE,
    AdditionalContext,
)
from lib.internal_events import InternalEventsClient
from lib.internal_events.context import (
    InternalEventAdditionalProperties,
    current_event_context,
)
from lib.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)

CONTAINER_APPLICATION_PACKAGES = ["duo_workflow_service"]

MAX_MESSAGE_SIZE = 4 * 1024 * 1024

# Defines the limit for metadata/headers sent by the client:
# https://github.com/grpc/grpc/blob/06f6f5d376a8c7abf067d060a28bf12afb664a7e/include/grpc/impl/channel_arg_names.h#L216C37-L216C59
# Default is 8KB, we increase it to 24KB
# Also used to increase the max header size of the HTTP client:
# https://docs.aiohttp.org/en/stable/client_reference.html
MAX_METADATA_SIZE = 24 * 1024

log = structlog.stdlib.get_logger("server")

catalog = data_model.load_catalog()
allowed_ijwt_scopes = {
    unit_primitive.name
    for unit_primitive in catalog.unit_primitives
    if "duo_workflow_service" in unit_primitive.backend_services
}


def string_to_category_enum(category_string: str) -> CategoryEnum:
    try:
        if "/" in category_string:
            _, flow_config_path = flow_registry.parse_workflow_definition(
                category_string
            )
            category_string = flow_config_path

        return CategoryEnum(category_string)
    except ValueError:
        # Handle case when string doesn't match any enum value
        # We will return default workflow type
        # Since it isn't a blocker for workflow run
        log.warning(f"Unknown category string: {category_string}")
        return CategoryEnum.UNKNOWN


def clean_start_request(start_workflow_request: contract_pb2.ClientEvent):
    request = contract_pb2.ClientEvent()
    request.CopyFrom(start_workflow_request)
    # Remove the goal from being logged to prevent logging sensitive user content
    request.startRequest.ClearField("goal")
    request.startRequest.ClearField("workflowMetadata")
    return request


class DuoWorkflowService(contract_pb2_grpc.DuoWorkflowServicer):
    # Set to 2 seconds to provide a reasonable balance between:
    # - Giving tasks enough time to properly clean up resources
    # - Not delaying the server response for too long when handling errors
    TASK_CANCELLATION_TIMEOUT = 2.0

    # pylint: disable=invalid-overridden-method
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    @inject
    async def ExecuteWorkflow(
        self,
        request_iterator: AsyncIterable[contract_pb2.ClientEvent],
        context: grpc.ServicerContext,
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
    ) -> AsyncIterator[contract_pb2.Action]:
        user: CloudConnectorUser = current_user_context_var.get()

        # Fetch the start workflow call
        start_workflow_request: contract_pb2.ClientEvent = await anext(
            aiter(request_iterator)
        )

        workflow_definition = start_workflow_request.startRequest.workflowDefinition
        unit_primitive = choose_unit_primitive(workflow_definition)
        legacy_unit_primitive = choose_legacy_unit_primitive(workflow_definition)

        if not user.can(unit_primitive):
            # DUO_WORKFLOW_EXECUTE_WORKFLOW unit primitive is being deprecated and replaced with DUO_AGENT_PLATFORM
            # While the migration is in progress, also check DUO_WORKFLOW_EXECUTE_WORKFLOW
            if legacy_unit_primitive is None:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    f"Unauthorized to execute {workflow_definition or 'workflow'}",
                )

            if not user.can(legacy_unit_primitive):
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    f"Unauthorized to execute {workflow_definition or 'workflow'}",
                )

        monitoring_context: MonitoringContext = current_monitoring_context.get()

        workflow_id = start_workflow_request.startRequest.workflowID
        set_workflow_id(workflow_id)

        # Get event context for enhanced logging
        event_context = current_event_context.get()

        # Build extra logging context with safe attribute access
        extra_context = {
            "workflow_id": workflow_id,
            "workflow_definition": workflow_definition,
        }

        if event_context is not None:
            extra_context.update(
                {
                    "instance_id": (
                        str(event_context.instance_id)
                        if event_context.instance_id is not None
                        else "None"
                    ),
                    "host_name": (
                        str(event_context.host_name)
                        if event_context.host_name is not None
                        else "None"
                    ),
                    "realm": (
                        str(event_context.realm)
                        if event_context.realm is not None
                        else "None"
                    ),
                    "is_gitlab_team_member": (
                        str(event_context.is_gitlab_team_member)
                        if event_context.is_gitlab_team_member is not None
                        else "None"
                    ),
                    "global_user_id": (
                        str(event_context.global_user_id)
                        if event_context.global_user_id is not None
                        else "None"
                    ),
                    "correlation_id": (
                        str(event_context.correlation_id)
                        if event_context.correlation_id is not None
                        else "None"
                    ),
                }
            )
        else:
            log.debug("Event context not available for enhanced logging")

        # Enhanced logging with additional context
        log.info(
            "Starting workflow %s",
            clean_start_request(start_workflow_request),
            extra=extra_context,
        )
        workflow_type = string_to_category_enum(workflow_definition)
        duo_workflow_metrics.count_agent_platform_receive_start_counter(
            flow_type=workflow_type
        )
        internal_event_client.track_event(
            event_name=EventEnum.RECEIVE_START_REQUEST.value,
            additional_properties=InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_RECEIVE_START_REQUEST_LABEL.value,
                property=EventPropertyEnum.WORKFLOW_ID.value,
                value=workflow_id,
            ),
            category=workflow_type.value,
        )

        goal = start_workflow_request.startRequest.goal

        if start_workflow_request.startRequest.additional_context:
            additional_context = [
                AdditionalContext(
                    category=e.category,
                    id=e.id,
                    content=e.content,
                    metadata=json.loads(e.metadata),
                )
                for e in start_workflow_request.startRequest.additional_context
            ]
        else:
            additional_context = None

        workflow_metadata = {}
        monitoring_context.workflow_id = workflow_id
        monitoring_context.workflow_definition = workflow_definition
        if start_workflow_request.startRequest.workflowMetadata:
            workflow_metadata = json.loads(
                start_workflow_request.startRequest.workflowMetadata
            )

        mcp_tools = []
        if start_workflow_request.startRequest.mcpTools:
            mcp_tools = list(start_workflow_request.startRequest.mcpTools)

        flow_config = start_workflow_request.startRequest.flowConfig
        flow_config_schema_version = (
            start_workflow_request.startRequest.flowConfigSchemaVersion or None
        )

        workflow_class: FlowFactory = resolve_workflow_class(
            workflow_definition, flow_config, flow_config_schema_version
        )

        invocation_metadata = dict(context.invocation_metadata())

        workflow: AbstractWorkflow = workflow_class(
            workflow_id=workflow_id,
            workflow_metadata=workflow_metadata,
            workflow_type=workflow_type,
            user=user,
            mcp_tools=mcp_tools,
            additional_context=additional_context,
            invocation_metadata={
                "base_url": invocation_metadata.get("x-gitlab-base-url", ""),
                "gitlab_token": invocation_metadata.get("x-gitlab-oauth-token", ""),
            },
            approval=start_workflow_request.startRequest.approval,
            language_server_version=language_server_version.get(),
            preapproved_tools=list(
                start_workflow_request.startRequest.preapproved_tools
            ),
        )

        workflow_task = asyncio.create_task(workflow.run(goal))

        async def send_events() -> AsyncIterator[contract_pb2.Action]:
            while True:
                item: contract_pb2.Action | OutboxSignal = (
                    await workflow.get_from_outbox()
                )

                if item == OutboxSignal.NO_MORE_OUTBOUND_REQUESTS:
                    log.info("No more outbound requests. End send_events loop.")
                    break

                if not isinstance(item, contract_pb2.Action):
                    raise RuntimeError(
                        "Can not send an action that is not the Action type"
                    )

                payload_size = item.ByteSize()

                if payload_size > MAX_MESSAGE_SIZE:
                    log.error(
                        "Failed to send an outgoing action: Message size too large",
                        request_id=item.requestID,
                        payload_size=payload_size,
                        action_class=item.WhichOneof("action"),
                    )
                    workflow_task.cancel(OUTGOING_MESSAGE_TOO_LARGE)

                    # We also need to fail this action in the outbox otherwise
                    # we can block the checkpointing thread in langgraph which
                    # is waiting on a response from a checkpoint that is too
                    # large.
                    workflow.fail_outbox_action(
                        item.requestID, OUTGOING_MESSAGE_TOO_LARGE
                    )

                    # Continue so we have a chance to cleanup. Some outgoing
                    # status updates hopefully make it through
                    continue

                log.info(
                    "Sending an outgoing action",
                    request_id=item.requestID,
                    payload_size=payload_size,
                    action_class=item.WhichOneof("action"),
                )

                yield item

                log.info(
                    "Sent an outgoing action",
                    request_id=item.requestID,
                    action_class=item.WhichOneof("action"),
                )

        async def receive_events():
            while True:
                event = await next_client_event(request_iterator)

                if event is None:
                    log.info("Skipping ClientEvent None")
                    workflow_task.cancel("Client-side streaming has been closed.")
                    break

                log.info(
                    "Received a client event.",
                    responseType=event.WhichOneof("response"),
                    request_id=event.actionResponse.requestID,
                )

                if event.HasField("heartbeat"):
                    continue

                if event.HasField("actionResponse"):
                    workflow.set_action_response(event)
                    continue

                if event.HasField("stopWorkflow"):
                    log.info(
                        "Stopping workflow...",
                        reason=event.stopWorkflow.reason,
                    )
                    monitoring_context.workflow_stop_reason = event.stopWorkflow.reason
                    workflow_task.cancel(AIO_CANCEL_STOP_WORKFLOW_REQUEST)
                    continue

        receive_events_task = asyncio.create_task(receive_events())

        async def abort_workflow(
            workflow_task: Optional[asyncio.Task], err: BaseException
        ):
            if workflow_task and not workflow_task.done():
                log.info("Aborting workflow...")
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
                except (asyncio.TimeoutError, asyncio.CancelledError) as ex:
                    log_exception(ex, extra={"source": __name__})

        try:
            async for action in send_events():
                yield action

            await workflow_task

            if workflow.successful_execution():
                context.set_code(grpc.StatusCode.OK)
                context.set_details("workflow execution success")
            elif AIO_CANCEL_STOP_WORKFLOW_REQUEST in str(workflow.last_error):
                context.set_code(grpc.StatusCode.OK)
                context.set_details(
                    f"workflow execution stopped: {workflow.last_gitlab_status}"
                )
            elif str(workflow.last_error) == OUTGOING_MESSAGE_TOO_LARGE:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("Outgoing message too large.")
            elif workflow.last_error:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"workflow execution failure: {type(workflow.last_error).__name__}: {workflow.last_error}"
                )
            else:
                context.set_code(grpc.StatusCode.UNKNOWN)
                context.set_details(
                    f"RPC ended with unknown workflow state: {workflow.last_gitlab_status}"
                )
        except asyncio.CancelledError as err:
            if AIO_CANCEL_STOP_WORKFLOW_REQUEST in str(err):
                context.set_code(grpc.StatusCode.OK)
                context.set_details(
                    f"workflow execution stopped: {workflow.last_gitlab_status}"
                )
            else:
                # This exception is raised when RPC is cancelled by the client.
                context.set_code(grpc.StatusCode.CANCELLED)
                context.set_details("RPC cancelled by client")
                log_exception(err, extra={"source": __name__})

            await abort_workflow(workflow_task, err)
            # Task cancellation must be reraised to the grpc server side so that the rpc task can be shutdown properly.
            raise
        except BaseException as err:
            if AIO_CANCEL_STOP_WORKFLOW_REQUEST not in str(err):
                log_exception(
                    err,
                    extra={
                        "workflow_id": workflow_id,
                        "source": __name__,
                    },
                )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Something went wrong: {type(err).__name__}: {err}")
            await abort_workflow(workflow_task, err)
        finally:
            await workflow.cleanup(workflow_id)
            receive_events_task.cancel()
            monitoring_context.workflow_last_gitlab_status = workflow.last_gitlab_status

    async def ListTools(
        self, request: contract_pb2.ListToolsRequest, context: grpc.ServicerContext
    ):
        log.info("Listing all available tools")
        tool_classes = set(
            (
                tools_registry._DEFAULT_TOOLS
                + tools_registry._READ_ONLY_GITLAB_TOOLS
                + list(chain.from_iterable(tools_registry._AGENT_PRIVILEGES.values()))
            )
        )
        response = contract_pb2.ListToolsResponse()
        for tool_cls in tool_classes:
            spec_struct = Struct()
            tool: DuoBaseTool = tool_cls()  # type: ignore[assignment]
            spec_struct.update(convert_to_openai_tool(tool))
            response.tools.append(spec_struct)

            for prompt in tool.eval_prompts or []:
                struct = Struct()
                struct.update({"prompt": prompt})
                response.eval_dataset.append(struct)

        return response

    async def ListFlows(
        self, request: contract_pb2.ListFlowsRequest, context: grpc.ServicerContext
    ):
        log.info("Listing all available flows")
        response = contract_pb2.ListFlowsResponse()

        configs = flow_registry.list_configs()

        # Apply filters if provided
        if request.filters:
            filtered_configs = []
            for config in configs:
                # Filter by name if provided
                if (
                    request.filters.flow_identifier
                    and config.get("flow_identifier")
                    not in request.filters.flow_identifier
                ):
                    continue

                # Filter by environment if provided
                if (
                    request.filters.environment
                    and config.get("environment") not in request.filters.environment
                ):
                    continue

                # Filter by version if provided
                if (
                    request.filters.version
                    and config.get("version") not in request.filters.version
                ):
                    continue

                filtered_configs.append(config)
            configs = filtered_configs

        for config in configs:
            spec_struct = Struct()
            spec_struct.update(config)
            response.configs.append(spec_struct)

        return response

    async def GenerateToken(
        self, request: contract_pb2.GenerateTokenRequest, context: grpc.ServicerContext
    ) -> contract_pb2.GenerateTokenResponse:
        user: CloudConnectorUser = current_user_context_var.get()

        workflow_definition = request.workflowDefinition
        unit_primitive = choose_unit_primitive(workflow_definition)
        legacy_unit_primitive = choose_legacy_unit_primitive(workflow_definition)

        if not user.can(
            unit_primitive=unit_primitive,
            disallowed_issuers=[CloudConnectorConfig().service_name],
        ):
            # DUO_WORKFLOW_EXECUTE_WORKFLOW unit primitive is being deprecated and replaced with DUO_AGENT_PLATFORM
            # While the migration is in progress, also check DUO_WORKFLOW_EXECUTE_WORKFLOW
            if legacy_unit_primitive is None:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
                )

            if not user.can(
                unit_primitive=legacy_unit_primitive,
                disallowed_issuers=[CloudConnectorConfig().service_name],
            ):
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
                )

        metadata = dict(context.invocation_metadata())
        global_user_id = metadata.get("x-gitlab-global-user-id")
        gitlab_realm = metadata.get("x-gitlab-realm")
        gitlab_instance_id = metadata.get("x-gitlab-instance-id")

        scopes = []
        if user.is_debug:
            scopes = list(allowed_ijwt_scopes)
        else:
            scopes = list(set(user.claims.scopes) & allowed_ijwt_scopes)

        token_authority = TokenAuthority(
            os.environ.get("DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY")
        )
        token, expires_at = token_authority.encode(
            global_user_id,
            gitlab_realm,
            user,
            gitlab_instance_id,
            scopes,
        )

        return contract_pb2.GenerateTokenResponse(token=token, expiresAt=expires_at)

    # pylint: enable=invalid-overridden-method
    # pylint: enable=too-many-statements


async def next_client_event(
    request_iterator: AsyncIterable[contract_pb2.ClientEvent],
) -> contract_pb2.ClientEvent | None:
    """Fetch a client event from gRPC stream.

    Return:
        contract_pb2.ClientEvent: A client event sent from the client via gRPC stream.
    """

    try:
        log.info("Waiting for next ClientEvent")
        event = await anext(aiter(request_iterator))
    except StopAsyncIteration:
        log.info("Client-side streaming has been closed.")
        return None

    return event


async def serve(port: int) -> None:
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
        pool_size=100,
        timeout=aiohttp.ClientTimeout(total=30),
        max_line_size=MAX_METADATA_SIZE,  # For entire header lines
        max_field_size=MAX_METADATA_SIZE,  # For individual header values
    )
    async with connection_pool:
        server_options = [
            ("grpc.keepalive_time_ms", 20 * 1000),
            ("grpc.http2.min_ping_interval_without_data_ms", 10 * 1000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.so_reuseport", 0),
            (
                "grpc.max_receive_message_length",
                MAX_MESSAGE_SIZE,
            ),
            (
                "grpc.max_send_message_length",
                MAX_MESSAGE_SIZE,
            ),
            (
                "grpc.max_metadata_size",
                MAX_METADATA_SIZE,
            ),
        ]

        server = grpc.aio.server(
            interceptors=[
                CorrelationIdInterceptor(),
                AuthenticationInterceptor(),
                FeatureFlagInterceptor(),
                EnabledInstanceVerboseAiLogsInterceptor(),
                LanguageServerVersionInterceptor(),
                GitLabVersionInterceptor(),
                GitLabRealmInterceptor(),
                ClientTypeInterceptor(),
                InternalEventsInterceptor(),
                ModelMetadataInterceptor(),
                PromptCachingInterceptor(),
                MonitoringInterceptor(),
            ],
            options=server_options,
        )
        contract_pb2_grpc.add_DuoWorkflowServicer_to_server(
            DuoWorkflowService(), server
        )
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

        # Set up graceful shutdown
        loop = asyncio.get_running_loop()
        setup_signal_handlers(server, loop)

        await server.wait_for_termination()
        log.info("Server shutdown complete")


def setup_signal_handlers(
    server: grpc.aio.Server, loop: asyncio.AbstractEventLoop
) -> None:
    """Set up signal handlers for graceful server shutdown."""

    grace_period_env = os.environ.get("DUO_WORKFLOW_SHUTDOWN_GRACE_PERIOD_S")
    grace_period = int(grace_period_env) if grace_period_env else None

    def handle_shutdown(sig):
        log.info(f"Received signal {sig}, initiating graceful shutdown")
        asyncio.create_task(server.stop(grace=grace_period))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, functools.partial(handle_shutdown, sig))


def configure_cache() -> None:
    if os.environ.get("LLM_CACHE") == "true":
        set_llm_cache(SQLiteCache(database_path=".llm_cache.db"))
    else:
        set_llm_cache(None)


def setup_cloud_connector():
    cloud_connector_service_name = os.environ.get(
        "DUO_WORKFLOW_CLOUD_CONNECTOR_SERVICE_NAME", "gitlab-duo-workflow-service"
    )
    os.environ["CLOUD_CONNECTOR_SERVICE_NAME"] = cloud_connector_service_name


def choose_unit_primitive(workflow_definition: str) -> GitLabUnitPrimitive:
    if workflow_definition == "chat":
        return GitLabUnitPrimitive.DUO_CHAT

    return GitLabUnitPrimitive.DUO_AGENT_PLATFORM


def choose_legacy_unit_primitive(
    workflow_definition: str,
) -> Optional[GitLabUnitPrimitive]:
    if workflow_definition == "chat":
        return None

    return GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW


def setup_container(config: Config):
    container_application = ContainerApplication()
    container_application.wire(packages=CONTAINER_APPLICATION_PACKAGES)
    container_application.config.from_dict(config.model_dump())


def run(config: Config):
    self_hosted_mode = config.custom_models.enabled

    setup_container(config)
    setup_litellm(config)
    setup_cloud_connector()
    setup_profiling()
    setup_error_tracking()
    setup_monitoring()
    setup_logging()
    configure_cache()
    if not self_hosted_mode:
        validate_llm_access()
    port = int(os.environ.get("PORT", "50052"))
    asyncio.get_event_loop().run_until_complete(serve(port))


def run_app():
    run(get_config())


if __name__ == "__main__":
    run_app()
