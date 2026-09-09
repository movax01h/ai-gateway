import asyncio
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from typing import Awaitable, Callable, Iterator, Optional, override

import grpc
import sentry_sdk
import structlog
from gitlab_cloud_connector.auth import (
    AUTH_TYPE_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_VERSION_HEADER,
)
from grpc.aio import ServerInterceptor
from prometheus_client import REGISTRY, Counter
from sentry_sdk.consts import OP
from sentry_sdk.tracing import TransactionSource

from duo_workflow_service.interceptors import GRPC_HEALTH_METHODS
from duo_workflow_service.tracking import MonitoringContext, current_monitoring_context
from duo_workflow_service.tracking.errors import log_exception
from lib.context import (
    METADATA_LABELS,
    build_metadata_labels,
    client_type,
    language_server_version,
)
from lib.feature_flags import current_feature_flag_context

log = structlog.stdlib.get_logger("grpc")

CANCELLED_BEFORE_START = "CANCELLED_BEFORE_START"
EXECUTE_WORKFLOW_METHOD_NAME = "ExecuteWorkflow"


class GRPCMethodType(StrEnum):
    UNARY = "UNARY"
    SERVER_STREAMING = "SERVER_STREAM"
    CLIENT_STREAMING = "CLIENT_STREAM"
    BIDI_STREAMING = "BIDI_STREAM"
    UNKNOWN = "UNKNOWN"


class MonitoringInterceptor(ServerInterceptor):
    def __init__(self, registry=REGISTRY):
        self._requests_counter: Counter = Counter(
            "grpc_server_handled_total",
            "Total number of RPCs completed on the server, regardless of success or failure.",
            [
                "grpc_type",
                "grpc_service",
                "grpc_method",
                "grpc_code",
                "flow_type",
            ]
            + METADATA_LABELS,
            registry=registry,
        )

    @override
    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Optional[grpc.RpcMethodHandler]:
        # Health checks are infrastructure-level calls that don't need application monitoring.
        if handler_call_details.method in GRPC_HEALTH_METHODS:
            return await continuation(handler_call_details)

        stream_fn, unary_fn = self._build_behavior_functions(handler_call_details)

        handler = await continuation(handler_call_details)

        if handler is None:
            return None

        # Wrap an RPC handler with the behavior that captures metrics.
        # The handler is of RpcMethodHandler type:
        #
        # https://github.com/grpc/grpc/blob/46c658ac018ba750e3e42c00a5fa1864780cc0f5/src/python/grpcio/grpc/__init__.py#L1325
        #
        # The handler contains implementations which are called based on the request/response types.
        # We wrap the implementations based on whether response is streamed or not with the behavior that captures the
        # metrics.
        if handler.request_streaming and handler.response_streaming:
            handler_factory = grpc.stream_stream_rpc_method_handler
            handler_func = stream_fn(
                handler.stream_stream, GRPCMethodType.BIDI_STREAMING
            )
        elif handler.request_streaming and not handler.response_streaming:
            handler_factory = grpc.stream_unary_rpc_method_handler
            handler_func = unary_fn(
                handler.stream_unary, GRPCMethodType.CLIENT_STREAMING
            )
        elif not handler.request_streaming and handler.response_streaming:
            handler_factory = grpc.unary_stream_rpc_method_handler
            handler_func = stream_fn(
                handler.unary_stream, GRPCMethodType.SERVER_STREAMING
            )
        else:
            handler_factory = grpc.unary_unary_rpc_method_handler
            handler_func = unary_fn(handler.unary_unary, GRPCMethodType.UNARY)

        # As a result, an grpc.RpcMethodHandler object is build with the correct arguments set.
        # For example, for stream_stream case:
        #
        # https://github.com/grpc/grpc/blob/b64756acca2eb942c97a416850ce5ab95a544d3e/src/python/grpcio/grpc/__init__.py#L1653
        return handler_factory(
            handler_func,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _build_behavior_functions(self, handler_call_details: grpc.HandlerCallDetails):
        _, grpc_service_name, grpc_method_name = handler_call_details.method.split("/")
        invocation_metadata = dict(handler_call_details.invocation_metadata)

        def handle_response_unary_behavior(
            behavior: Callable,
            grpc_type: GRPCMethodType,
        ) -> Callable:
            async def unary_behavior(request_or_iterator, servicer_context):
                # `sentry_root_span` wraps `monitoring` so the transaction is still
                # open when `monitoring` logs the exception, which is what links the
                # error event to the transaction.
                with (
                    self.sentry_root_span(
                        grpc_type=grpc_type,
                        grpc_service_name=grpc_service_name,
                        grpc_method_name=grpc_method_name,
                        invocation_metadata=invocation_metadata,
                    ),
                    self.monitoring(
                        grpc_type=grpc_type,
                        grpc_service_name=grpc_service_name,
                        grpc_method_name=grpc_method_name,
                        servicer_context=servicer_context,
                        invocation_metadata=invocation_metadata,
                    ),
                ):
                    response_or_iterator = await behavior(
                        request_or_iterator, servicer_context
                    )
                    return response_or_iterator

            return unary_behavior

        def handle_response_stream_behavior(
            behavior: Callable,
            grpc_type: GRPCMethodType,
        ) -> Callable:
            async def stream_behavior(request_or_iterator, servicer_context):
                # `sentry_root_span` wraps `monitoring` so the transaction is still
                # open when `monitoring` logs the exception, which is what links the
                # error event to the transaction.
                with (
                    self.sentry_root_span(
                        grpc_type=grpc_type,
                        grpc_service_name=grpc_service_name,
                        grpc_method_name=grpc_method_name,
                        invocation_metadata=invocation_metadata,
                    ),
                    self.monitoring(
                        grpc_type=grpc_type,
                        grpc_service_name=grpc_service_name,
                        grpc_method_name=grpc_method_name,
                        servicer_context=servicer_context,
                        invocation_metadata=invocation_metadata,
                    ),
                ):
                    async for behavior_response in behavior(
                        request_or_iterator, servicer_context
                    ):
                        yield behavior_response

            return stream_behavior

        return handle_response_stream_behavior, handle_response_unary_behavior

    @contextmanager
    def sentry_root_span(
        self,
        *,
        grpc_type: GRPCMethodType,
        grpc_service_name: str,
        grpc_method_name: str,
        invocation_metadata: dict,
    ) -> Iterator[None]:
        """Open the root Sentry transaction for streaming RPCs.

        Sentry's own gRPC interceptor only covers unary-unary, so without this the `gen_ai.*` spans on streaming calls
        are parentless and dropped.
        """
        if grpc_type is GRPCMethodType.UNARY or not sentry_sdk.is_initialized():
            yield
            return

        # `start_transaction` assigns to `scope.span` on the current scope, and
        # concurrent RPCs inherit the same scope object. Forking gives each call its
        # own scope so their child spans cannot overwrite each other. Sentry's own
        # unary-unary wrapper does the same.
        with sentry_sdk.isolation_scope():
            transaction = sentry_sdk.continue_trace(
                invocation_metadata,
                op=OP.GRPC_SERVER,
                name=f"/{grpc_service_name}/{grpc_method_name}",
                source=TransactionSource.CUSTOM,
            )

            with sentry_sdk.start_transaction(transaction=transaction):
                try:
                    yield
                finally:
                    context: MonitoringContext = current_monitoring_context.get()

                    for key, value in (
                        ("flow_type", context.workflow_definition),
                        ("flow_version", context.flow_version),
                        ("schema_version", context.schema_version),
                        ("workflow_stop_reason", context.workflow_stop_reason),
                        (
                            "workflow_last_gitlab_status",
                            context.workflow_last_gitlab_status,
                        ),
                    ):
                        if value:
                            transaction.set_tag(key, value)

                    # Unbounded values; tagging them would degrade tag search.
                    for key, value in (
                        ("workflow_id", context.workflow_id),
                        ("flow_id", context.flow_id),
                    ):
                        if value:
                            transaction.set_data(key, value)

    @contextmanager
    def monitoring(
        self,
        *,
        grpc_type,
        grpc_service_name,
        grpc_method_name,
        servicer_context,
        invocation_metadata,
    ):
        start_time_total = time.perf_counter()
        start_time_cpu = time.process_time()
        request_arrived_at = datetime.now(timezone.utc)
        current_monitoring_context.set(MonitoringContext())

        try:
            yield

            context: MonitoringContext = current_monitoring_context.get()
            if not context.workflow_no_start_reason:
                self._increase_grpc_server_handled_total_counter(
                    grpc_type,
                    grpc_service_name,
                    grpc_method_name,
                    servicer_context.code(),
                )
        except BaseException as e:  # We handle all BaseException to ensure we include asyncio.CancelledError
            context = current_monitoring_context.get()
            if self._is_cancelled_before_workflow_start(e, grpc_method_name, context):
                context.workflow_no_start_reason = CANCELLED_BEFORE_START
            else:
                self._handle_error(
                    e,
                    grpc_type,
                    grpc_service_name,
                    grpc_method_name,
                    servicer_context,
                )

                log_exception(e)

            raise e
        finally:
            context = current_monitoring_context.get()

            # When the client connected but never sent a StartRequest (or sent one
            # without a workflowID), the workflow never actually ran.  Skip the normal
            # "Finished RPC" log (the Prometheus counter is already skipped above in
            # the try block) so that these no-op connections don't appear as executed
            # flows, but emit a dedicated info entry so the connection is still traceable.
            if context.workflow_no_start_reason:
                log.info(
                    "connection closed before workflow started",
                    grpc_type=grpc_type,
                    grpc_service_name=grpc_service_name,
                    grpc_method_name=grpc_method_name,
                    request_arrived_at=request_arrived_at.isoformat(),
                    no_start_reason=context.workflow_no_start_reason,
                    gitlab_host_name=invocation_metadata.get(
                        X_GITLAB_HOST_NAME_HEADER.lower()
                    ),
                    gitlab_realm=invocation_metadata.get(X_GITLAB_REALM_HEADER.lower()),
                    gitlab_instance_id=invocation_metadata.get(
                        X_GITLAB_INSTANCE_ID_HEADER.lower()
                    ),
                    gitlab_authentication_type=invocation_metadata.get(
                        AUTH_TYPE_HEADER.lower()
                    ),
                    user_agent=invocation_metadata.get("user-agent"),
                )
            else:
                elapsed_time = time.perf_counter() - start_time_total
                cpu_time = time.process_time() - start_time_cpu
                servicer_context_code = ""

                if context_code := servicer_context.code():
                    servicer_context_code = context_code.name

                fields = {
                    "duration_s": elapsed_time,
                    "request_arrived_at": request_arrived_at.isoformat(),
                    "cpu_s": cpu_time,
                    "grpc_type": grpc_type,
                    "grpc_service_name": grpc_service_name,
                    "grpc_method_name": grpc_method_name,
                    "servicer_context_code": servicer_context_code,
                    "servicer_context_details": servicer_context.details(),
                    "gitlab_host_name": invocation_metadata.get(
                        X_GITLAB_HOST_NAME_HEADER.lower()
                    ),
                    "gitlab_realm": invocation_metadata.get(
                        X_GITLAB_REALM_HEADER.lower()
                    ),
                    "gitlab_instance_id": invocation_metadata.get(
                        X_GITLAB_INSTANCE_ID_HEADER.lower()
                    ),
                    "gitlab_authentication_type": invocation_metadata.get(
                        AUTH_TYPE_HEADER.lower()
                    ),
                    "gitlab_version": invocation_metadata.get(
                        X_GITLAB_VERSION_HEADER.lower()
                    ),
                    "user_agent": invocation_metadata.get("user-agent"),
                }

                if lsp_version := language_server_version.get():
                    fields["language_server_version"] = str(lsp_version.version)

                if client_type_value := client_type.get():
                    fields["gitlab_client_type"] = client_type_value

                if feature_flags := current_feature_flag_context.get():
                    fields["feature_flags"] = feature_flags

                fields.update(context.model_dump())

                log.info(
                    f"""Finished {grpc_method_name} RPC""",
                    **fields,
                )

    @staticmethod
    def _is_cancelled_before_workflow_start(
        error: BaseException, grpc_method_name: str, context: MonitoringContext
    ) -> bool:
        # Only ExecuteWorkflow has a StartRequest; a stream cancelled before reading it never executed a flow.
        return (
            isinstance(error, asyncio.CancelledError)
            and grpc_method_name == EXECUTE_WORKFLOW_METHOD_NAME
            and not context.workflow_id
            and not context.workflow_no_start_reason
        )

    def _handle_error(
        self,
        _e: BaseException,
        grpc_type: GRPCMethodType,
        grpc_service_name: str,
        grpc_method_name: str,
        servicer_context: grpc.ServicerContext,
    ) -> None:
        status_code = servicer_context.code()
        if not status_code or status_code == grpc.StatusCode.OK:
            status_code = grpc.StatusCode.UNKNOWN

        self._increase_grpc_server_handled_total_counter(
            grpc_type, grpc_service_name, grpc_method_name, status_code
        )

    def _increase_grpc_server_handled_total_counter(
        self,
        grpc_type: GRPCMethodType,
        grpc_service_name: str,
        grpc_method_name: str,
        grpc_code: grpc.StatusCode,
    ) -> None:
        grpc_code = grpc_code or grpc.StatusCode.OK

        context: MonitoringContext = current_monitoring_context.get()

        flow_type = "unknown"
        if context and context.workflow_definition:
            flow_type = context.workflow_definition

        self._requests_counter.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
            grpc_code=grpc_code.name,
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()
