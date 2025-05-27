import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from typing import Awaitable, Callable, Optional

import grpc
import structlog
from gitlab_cloud_connector.auth import (
    AUTH_TYPE_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
)
from grpc.aio import ServerInterceptor
from prometheus_client import REGISTRY, Counter

from duo_workflow_service.tracking import MonitoringContext, current_monitoring_context

log = structlog.stdlib.get_logger("grpc")


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
            ["grpc_type", "grpc_service", "grpc_method", "grpc_code"],
            registry=registry,
        )

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Optional[grpc.RpcMethodHandler]:
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
        # We wrap the implementations based on whether response is streamed or not with the behavior that captures the metrics.
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
                with self.monitoring(
                    grpc_type=grpc_type,
                    grpc_service_name=grpc_service_name,
                    grpc_method_name=grpc_method_name,
                    servicer_context=servicer_context,
                    invocation_metadata=invocation_metadata,
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
                with self.monitoring(
                    grpc_type=grpc_type,
                    grpc_service_name=grpc_service_name,
                    grpc_method_name=grpc_method_name,
                    servicer_context=servicer_context,
                    invocation_metadata=invocation_metadata,
                ):
                    async for behavior_response in behavior(
                        request_or_iterator, servicer_context
                    ):
                        yield behavior_response

            return stream_behavior

        return handle_response_stream_behavior, handle_response_unary_behavior

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
        exception_fields = {}

        start_time_total = time.perf_counter()
        start_time_cpu = time.process_time()
        request_arrived_at = datetime.now(timezone.utc)
        current_monitoring_context.set(MonitoringContext())

        try:
            yield

            self._increase_grpc_server_handled_total_counter(
                grpc_type,
                grpc_service_name,
                grpc_method_name,
                servicer_context.code(),
            )
        except Exception as e:
            self._handle_error(
                e,
                grpc_type,
                grpc_service_name,
                grpc_method_name,
                servicer_context,
            )

            exception_fields["exception_message"] = str(e)
            exception_fields["exception_class"] = type(e).__name__
            exception_fields["exception_backtrace"] = traceback.format_exc()

            raise e
        finally:
            elapsed_time = time.perf_counter() - start_time_total
            cpu_time = time.process_time() - start_time_cpu

            fields = {
                "duration_s": elapsed_time,
                "request_arrived_at": request_arrived_at.isoformat(),
                "cpu_s": cpu_time,
                "grpc_type": grpc_type,
                "grpc_service_name": grpc_service_name,
                "grpc_method_name": grpc_method_name,
                "servicer_context_code": (
                    servicer_context.code() or grpc.StatusCode.OK
                ).name,
                "gitlab_host_name": invocation_metadata.get(
                    X_GITLAB_HOST_NAME_HEADER.lower()
                ),
                "gitlab_realm": invocation_metadata.get(X_GITLAB_REALM_HEADER.lower()),
                "gitlab_instance_id": invocation_metadata.get(
                    X_GITLAB_INSTANCE_ID_HEADER.lower()
                ),
                "gitlab_authentication_type": invocation_metadata.get(
                    AUTH_TYPE_HEADER.lower()
                ),
                "user_agent": invocation_metadata.get("user-agent"),
            }
            fields.update(exception_fields)

            context: MonitoringContext = current_monitoring_context.get()
            fields.update(context.model_dump())

            log.info(
                f"""Finished {grpc_method_name} RPC""",
                **fields,
            )

    # pylint: disable=too-many-positional-arguments
    def _handle_error(
        self,
        e: Exception,  # pylint: disable=unused-argument
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

    # pylint: enable=too-many-positional-arguments

    def _increase_grpc_server_handled_total_counter(
        self,
        grpc_type: GRPCMethodType,
        grpc_service_name: str,
        grpc_method_name: str,
        grpc_code: grpc.StatusCode,
    ) -> None:
        grpc_code = grpc_code or grpc.StatusCode.OK

        self._requests_counter.labels(
            grpc_type=grpc_type,
            grpc_service=grpc_service_name,
            grpc_method=grpc_method_name,
            grpc_code=grpc_code.name,
        ).inc()
