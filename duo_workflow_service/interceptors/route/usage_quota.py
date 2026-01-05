import functools
import inspect
from collections.abc import AsyncIterable
from typing import Callable

from google.protobuf.message import Message
from grpc import StatusCode
from grpc.aio import ServicerContext

from contract import contract_pb2, contract_pb2_grpc
from lib.events import GLReportingEventContext
from lib.usage_quota import EventType, InsufficientCredits, UsageQuotaService


def has_sufficient_usage_quota(
    event: EventType,
    customersdot_url: str,
    user: str | None = None,
    token: str | None = None,
):
    def decorator(func: Callable):
        service = UsageQuotaService(
            customersdot_url, customersdot_api_user=user, customersdot_api_token=token
        )

        if inspect.isasyncgenfunction(func):
            return _process_stream(func, service, event)

        # TODO: investigate what unary endpoints need to be wrapped with '@has_sufficient_usage_quota'.
        # TODO: At this moment, it sends empty feature_qualified_name,
        # TODO: replicating the behaviour of the removed interceptor.
        return _process_unary(func, service, event)

    return decorator


async def abort_route_interceptor(
    context: ServicerContext, code: StatusCode, message: str
):
    await context.abort(code, message)
    return


def _process_stream(func: Callable, service: UsageQuotaService, event: EventType):
    if func.__name__ != contract_pb2_grpc.DuoWorkflowServicer.ExecuteWorkflow.__name__:
        raise TypeError(f"unsupported method to intercept '{func.__qualname__}'")

    @functools.wraps(func)
    async def wrapper(
        obj,
        request: AsyncIterable[contract_pb2.ClientEvent],
        grpc_context: ServicerContext,
        *args,
        **kwargs,
    ):
        try:
            message = await anext(aiter(request))
            gl_events_context = GLReportingEventContext.from_workflow_definition(
                message.startRequest.workflowDefinition,
                has_flow_config=bool(message.startRequest.flowConfig),
            )

            async def _chained():
                yield message
                async for _item in request:
                    yield _item

            await service.execute(gl_events_context, event)

            async for item in func(obj, _chained(), grpc_context, *args, **kwargs):
                yield item

        except InsufficientCredits as e:
            await abort_route_interceptor(
                grpc_context,
                StatusCode.RESOURCE_EXHAUSTED,
                f"{str(e).rstrip(".")}. Error code: USAGE_QUOTA_EXCEEDED",
            )

    return wrapper


def _process_unary(func: Callable, service: UsageQuotaService, event: EventType):
    @functools.wraps(func)
    async def wrapper(
        obj,
        request: AsyncIterable[Message],
        context: ServicerContext,
        *args,
        **kwargs,
    ):
        # Create a minimal GLReportingEventContext for unary calls
        gl_events_context = GLReportingEventContext("", "", is_ai_catalog_item=False)
        await service.execute(gl_events_context, event)

        return await func(obj, request, context, *args, **kwargs)

    return wrapper
