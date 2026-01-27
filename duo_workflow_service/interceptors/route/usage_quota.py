import functools
from collections.abc import AsyncIterable
from typing import Callable

from grpc import StatusCode
from grpc.aio import ServicerContext

from contract import contract_pb2, contract_pb2_grpc
from lib.events import FeatureQualifiedNameStatic, GLReportingEventContext
from lib.usage_quota import InsufficientCredits, UsageQuotaEvent, UsageQuotaService


def has_sufficient_usage_quota(
    event: UsageQuotaEvent,
    customersdot_url: str,
    user: str | None = None,
    token: str | None = None,
):
    def decorator(func: Callable):
        service = UsageQuotaService(
            customersdot_url, customersdot_api_user=user, customersdot_api_token=token
        )

        match func.__name__:
            case contract_pb2_grpc.DuoWorkflowServicer.ExecuteWorkflow.__name__:
                return _process_execute_workflow_stream(func, service, event)
            case contract_pb2_grpc.DuoWorkflowServicer.GenerateToken.__name__:
                return _process_generate_token_unary(func, service, event)
            case (
                contract_pb2_grpc.DuoWorkflowServicer.TrackSelfHostedExecuteWorkflow.__name__
            ):
                return _process_track_self_hosted_execute_workflow_stream(
                    func, service, event
                )
            case _:
                raise TypeError(
                    f"unsupported method to intercept '{func.__qualname__}'"
                )

    return decorator


async def abort_route_interceptor(
    context: ServicerContext, code: StatusCode, message: str
):
    await context.abort(code, message)
    return


def _process_execute_workflow_stream(
    func: Callable, service: UsageQuotaService, event: UsageQuotaEvent
):
    if event is not event.DAP_FLOW_ON_EXECUTE:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be '{event.DAP_FLOW_ON_EXECUTE}'"
        )

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
                is_ai_catalog_item=bool(message.startRequest.flowConfig),
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
                f"{str(e).rstrip('.')}. Error code: USAGE_QUOTA_EXCEEDED",
            )

    return wrapper


def _process_track_self_hosted_execute_workflow_stream(
    func: Callable, service: UsageQuotaService, event: UsageQuotaEvent
):
    if event is not event.DAP_FLOW_ON_EXECUTE:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be '{event.DAP_FLOW_ON_EXECUTE}'"
        )

    @functools.wraps(func)
    async def wrapper(
        obj,
        request: AsyncIterable[contract_pb2.TrackSelfHostedClientEvent],
        grpc_context: ServicerContext,
        *args,
        **kwargs,
    ):
        try:
            message = await anext(aiter(request))
        except StopAsyncIteration:
            return

        try:
            gl_events_context = GLReportingEventContext.from_workflow_definition(
                message.featureQualifiedName,
                is_ai_catalog_item=message.featureAiCatalogItem,
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
                f"{str(e).rstrip('.')}. Error code: USAGE_QUOTA_EXCEEDED",
            )

    return wrapper


def _process_generate_token_unary(
    func: Callable, service: UsageQuotaService, event: UsageQuotaEvent
):
    if event is not event.DAP_FLOW_ON_GENERATE_TOKEN:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be {event.DAP_FLOW_ON_GENERATE_TOKEN}"
        )

    @functools.wraps(func)
    async def wrapper(
        obj,
        request: contract_pb2.GenerateTokenRequest,
        grpc_context: ServicerContext,
        *args,
        **kwargs,
    ):
        try:
            if request.workflowDefinition:
                gl_events_context = GLReportingEventContext.from_workflow_definition(
                    request.workflowDefinition,
                    # Legacy support: we pass None since GenerateTokenRequest doesn't provide any information
                    # to resolve the value correctly. The method is limited to the DAP_FLOW_ON_GENERATE_TOKEN event only
                    # and this behavior is known to CustomerDot.
                    is_ai_catalog_item=None,
                )
            else:
                gl_events_context = GLReportingEventContext.from_static_name(
                    FeatureQualifiedNameStatic.DAP_FEATURE_LEGACY,
                    is_ai_catalog_item=None,
                )

            await service.execute(gl_events_context, event)

            return await func(obj, request, grpc_context, *args, **kwargs)
        except InsufficientCredits as e:
            await abort_route_interceptor(
                grpc_context,
                StatusCode.RESOURCE_EXHAUSTED,
                f"{str(e).rstrip('.')}. Error code: USAGE_QUOTA_EXCEEDED",
            )

    return wrapper
