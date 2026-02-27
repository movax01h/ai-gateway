import json
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Mapping

import fastapi
import litellm
from fastapi import status
from fastapi.responses import JSONResponse
from gitlab_cloud_connector import GitLabUnitPrimitive
from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    create_pass_through_route,
)
from litellm.proxy.proxy_server import UserAPIKeyAuth
from pydantic import BaseModel
from starlette.background import BackgroundTask

from ai_gateway.config import ConfigModelLimits
from ai_gateway.instrumentators.model_requests import (
    ModelRequestInstrumentator,
    get_llm_operations,
    init_llm_operations,
)
from ai_gateway.vendor.langchain_litellm.litellm import _create_usage_metadata
from lib.billing_events import BillingEvent, BillingEventsClient
from lib.events import FeatureQualifiedNameStatic, GLReportingEventContext
from lib.internal_events.client import InternalEventsClient


async def extract_json_body(request: fastapi.Request) -> Any:
    body = await request.body()

    try:
        json_body = json.loads(body)
    except json.JSONDecodeError:
        raise fastapi.HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON"
        )

    return json_body


def _create_headers_to_upstream(
    headers_from_downstream: Mapping[str, str], allowed_headers_to_upstream: list[str]
) -> dict[str, str]:
    return {
        key: headers_from_downstream[key]
        for key in allowed_headers_to_upstream
        if key in headers_from_downstream
    }


def _create_headers_to_downstream(
    headers_from_upstream: dict[str, str], allowed_headers_to_downstream: list[str]
) -> dict[str, str]:
    return {
        key: headers_from_upstream[key]
        for key in allowed_headers_to_downstream
        if key in headers_from_upstream
    }


class ProxyModel(BaseModel):
    base_url: str
    """Base URL of the upstream service."""

    model_name: str
    """Model name from the request."""

    upstream_path: str
    """Upstream target path from the request."""

    stream: bool
    """Stream flag from the request."""

    upstream_service: str
    """Name of the upstream service."""

    headers_to_upstream: dict[str, str]
    """Update headers for vendor specific requirements."""

    allowed_upstream_models: list[str]
    """Allowed models to the upstream service."""

    allowed_headers_to_upstream: list[str]
    """Allowed request headers to the upstream service."""

    allowed_headers_to_downstream: list[str]
    """Allowed response headers to the downstream service."""


class BaseProxyModelFactory(ABC):
    @abstractmethod
    async def factory(self, request: fastapi.Request) -> ProxyModel:
        pass


class ProxyClient:
    def __init__(
        self,
        limits: ConfigModelLimits,
        internal_event_client: InternalEventsClient,
        billing_event_client: BillingEventsClient,
    ):
        self.limits = limits
        self.internal_event_client = internal_event_client
        self.billing_event_client = billing_event_client
        self.watcher = None
        self.user = None
        current_proxy_client.set(self)

    async def proxy(
        self, request: fastapi.Request, model: ProxyModel
    ) -> fastapi.Response:
        headers_to_upstream = _create_headers_to_upstream(
            request.headers, model.allowed_headers_to_upstream
        )
        headers_to_upstream.update(model.headers_to_upstream)

        upstream_service = model.upstream_service
        model_name = model.model_name

        try:
            with ModelRequestInstrumentator(
                model_engine=upstream_service,
                model_provider=upstream_service,
                model_name=model_name,
                limits=self.limits.for_model(engine=upstream_service, name=model_name),
            ).watch(
                stream=model.stream,
                unit_primitive=GitLabUnitPrimitive.AI_GATEWAY_MODEL_PROVIDER_PROXY,
                internal_event_client=self.internal_event_client,
            ) as watcher:
                # Setup token tracking
                self.watcher = watcher
                self.user = request.user
                init_llm_operations()

                endpoint_func = create_pass_through_route(
                    endpoint="",  # This is unused in the litellm code
                    target=f"{model.base_url}{model.upstream_path}",
                    custom_headers=headers_to_upstream,
                )
                response_from_upstream = await endpoint_func(
                    request,
                    None,  # FastAPI response object, to inject litellm headers into it, which we don't need
                    UserAPIKeyAuth(),  # LiteLLM-Proxy auth, which we don't use
                )
        except ProxyException as e:
            try:
                error_content = json.loads(e.message)
            except json.decoder.JSONDecodeError:
                error_content = {"message": e.message}
            # Return api.anthropic.com response for Anthropic error codes invalid_request_error
            return JSONResponse(
                status_code=int(e.code),
                content=error_content,
            )
        except Exception:
            raise fastapi.HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Bad Gateway"
            )

        headers_to_downstream = _create_headers_to_downstream(
            response_from_upstream.headers,
            model.allowed_headers_to_downstream,
        )

        if isinstance(response_from_upstream, fastapi.responses.StreamingResponse):
            return fastapi.responses.StreamingResponse(
                response_from_upstream.body_iterator,
                status_code=response_from_upstream.status_code,
                headers=headers_to_downstream,
                background=BackgroundTask(func=watcher.afinish),
            )
        return fastapi.Response(
            content=response_from_upstream.body,
            status_code=response_from_upstream.status_code,
            headers=headers_to_downstream,
        )


current_proxy_client: ContextVar[ProxyClient | None] = ContextVar(
    "current_proxy_client", default=None
)


async def litellm_async_success_callback(
    _kwargs, completion_obj, _start_time, _end_time
):
    proxy_client = current_proxy_client.get()
    if not proxy_client:
        return

    if usage := completion_obj.get("usage"):
        # LiteLLM supported model, get parsed data
        usage_metadata = _create_usage_metadata(usage)
    elif response := completion_obj.get("response"):
        # Raw response, fallback to simple parsing
        response_json = json.loads(response)
        usage_metadata = response_json.get("usage")

    if not usage_metadata:
        return

    # Reset proxy client so the callback doesn't try to process other requests
    current_proxy_client.set(None)

    watcher = proxy_client.watcher
    # Only track token usage if we are indeed in the middle of an instrumentation call
    if not watcher:
        return

    gl_event_context = GLReportingEventContext.from_static_name(
        FeatureQualifiedNameStatic.AIGW_PROXY_USE, is_ai_catalog_item=False
    )
    # We're intentionally not using the model name returned by the upstream provider, because it may differ from the
    # one in the request (e.g. `gpt-5` vs `gpt-5-2025-08-07`), and for tracking we want to stick to known model values.
    proxy_client.watcher.register_token_usage(
        proxy_client.watcher.labels["model_name"], usage_metadata
    )

    metadata = {
        "llm_operations": get_llm_operations(),
        "feature_qualified_name": gl_event_context.feature_qualified_name,
        "feature_ai_catalog_item": gl_event_context.feature_ai_catalog_item,
    }

    # Track event only after `func` returns so we don't trigger a billable event if an exception occurred
    proxy_client.billing_event_client.track_billing_event(
        proxy_client.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category=__name__,
        unit_of_measure="request",
        quantity=1,
        metadata=metadata,
    )


litellm.logging_callback_manager.add_litellm_async_success_callback(
    litellm_async_success_callback
)
