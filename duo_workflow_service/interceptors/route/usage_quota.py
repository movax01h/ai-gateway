import functools
from collections.abc import AsyncIterable
from typing import Callable

from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector.user import CloudConnectorUser
from grpc import StatusCode
from grpc.aio import ServicerContext

from ai_gateway.container import ContainerApplication
from contract import contract_pb2, contract_pb2_grpc
from duo_workflow_service.interceptors.authentication_interceptor import (
    current_user as current_user_context_var,
)
from lib.events import FeatureQualifiedNameStatic, GLReportingEventContext
from lib.usage_quota import InsufficientCredits, UsageQuotaEvent, UsageQuotaService
from lib.usage_quota.client import should_skip_usage_quota_for_user


def has_sufficient_usage_quota(
    event: UsageQuotaEvent,
):
    def decorator(func: Callable):
        match func.__name__:
            case contract_pb2_grpc.DuoWorkflowServicer.ExecuteWorkflow.__name__:
                return _process_execute_workflow_stream(func, event)
            case contract_pb2_grpc.DuoWorkflowServicer.GenerateToken.__name__:
                return _process_generate_token_unary(func, event)
            case (
                contract_pb2_grpc.DuoWorkflowServicer.TrackSelfHostedExecuteWorkflow.__name__
            ):
                return _process_track_self_hosted_execute_workflow_stream(func, event)
            case _:
                raise TypeError(
                    f"unsupported method to intercept '{func.__qualname__}'"
                )

    return decorator


async def abort_route_interceptor(
    context: ServicerContext, code: StatusCode, message: str
):
    await context.abort(code, message)


def _process_execute_workflow_stream(func: Callable, event: UsageQuotaEvent):
    if event is not event.DAP_FLOW_ON_EXECUTE:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be '{event.DAP_FLOW_ON_EXECUTE}'"
        )

    @functools.wraps(func)
    @inject
    async def wrapper(
        obj,
        request: AsyncIterable[contract_pb2.ClientEvent],
        grpc_context: ServicerContext,
        *args,
        service: UsageQuotaService = Provide[ContainerApplication.usage_quota.service],
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

            current_user: CloudConnectorUser = current_user_context_var.get(None)

            if not should_skip_usage_quota_for_user(current_user):
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
    func: Callable, event: UsageQuotaEvent
):
    if event is not event.DAP_FLOW_ON_EXECUTE:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be '{event.DAP_FLOW_ON_EXECUTE}'"
        )

    @functools.wraps(func)
    @inject
    async def wrapper(
        obj,
        request: AsyncIterable[contract_pb2.TrackSelfHostedClientEvent],
        grpc_context: ServicerContext,
        *args,
        service: UsageQuotaService = Provide[ContainerApplication.usage_quota.service],
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


def _process_generate_token_unary(func: Callable, event: UsageQuotaEvent):
    if event is not event.DAP_FLOW_ON_GENERATE_TOKEN:
        raise ValueError(
            f"Unsupported event type '{event.value}'. Expected to be {event.DAP_FLOW_ON_GENERATE_TOKEN}"
        )

    @functools.wraps(func)
    @inject
    async def wrapper(
        obj,
        request: contract_pb2.GenerateTokenRequest,
        grpc_context: ServicerContext,
        *args,
        service: UsageQuotaService = Provide[ContainerApplication.usage_quota.service],
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

            current_user: CloudConnectorUser = current_user_context_var.get(None)

            if not should_skip_usage_quota_for_user(current_user):
                await service.execute(gl_events_context, event)

            return await func(obj, request, grpc_context, *args, **kwargs)
        except InsufficientCredits as e:
            await abort_route_interceptor(
                grpc_context,
                StatusCode.RESOURCE_EXHAUSTED,
                f"{str(e).rstrip('.')}. Error code: USAGE_QUOTA_EXCEEDED",
            )

    return wrapper
