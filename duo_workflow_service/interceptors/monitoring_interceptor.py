from enum import StrEnum
from typing import Awaitable, Callable, Optional

import grpc
from grpc.aio import ServerInterceptor
from prometheus_client import REGISTRY, Counter


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

        def handle_response_unary_behavior(
            behavior: Callable,
            grpc_type: GRPCMethodType,
        ) -> Callable:
            async def unary_behavior(request_or_iterator, servicer_context):
                try:
                    response_or_iterator = await behavior(
                        request_or_iterator, servicer_context
                    )

                    self._increase_grpc_server_handled_total_counter(
                        grpc_type,
                        grpc_service_name,
                        grpc_method_name,
                        servicer_context.code(),
                    )

                    return response_or_iterator
                except Exception as e:
                    self._handle_error(
                        e,
                        grpc_type,
                        grpc_service_name,
                        grpc_method_name,
                        servicer_context,
                    )
                    raise e

            return unary_behavior

        def handle_response_stream_behavior(
            behavior: Callable,
            grpc_type: GRPCMethodType,
        ) -> Callable:
            async def stream_behavior(request_or_iterator, servicer_context):
                try:
                    async for behavior_response in behavior(
                        request_or_iterator, servicer_context
                    ):
                        yield behavior_response

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
                    raise e

            return stream_behavior

        return handle_response_stream_behavior, handle_response_unary_behavior

    # pylint: disable=too-many-positional-arguments
    def _handle_error(
        self,
        e: Exception,
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
