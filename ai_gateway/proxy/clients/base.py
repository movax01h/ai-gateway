import json
import re
import typing
from abc import ABC, abstractmethod
from contextvars import ContextVar

import fastapi
import httpx
import litellm
from fastapi import status
from gitlab_cloud_connector import GitLabUnitPrimitive
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.proxy_server import UserAPIKeyAuth
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


class BaseProxyClient(ABC):
    def __init__(
        self,
        client: httpx.AsyncClient,
        limits: ConfigModelLimits,
        internal_event_client: InternalEventsClient,
        billing_event_client: BillingEventsClient,
    ):
        self.client = client
        self.limits = limits
        self.internal_event_client = internal_event_client
        self.billing_event_client = billing_event_client
        self.watcher = None
        current_proxy_client.set(self)

    async def proxy(self, request: fastapi.Request) -> fastapi.Response:
        upstream_path = self._extract_upstream_path(request.url.__str__())
        json_body = await self._extract_json_body(request)
        model_name = self._extract_model_name(upstream_path, json_body)

        if model_name not in self._allowed_upstream_models():
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model"
            )

        stream = self._extract_stream_flag(upstream_path, json_body)
        headers_to_upstream = self._create_headers_to_upstream(request.headers)
        self._update_headers_to_upstream(headers_to_upstream)
        upstream_service = self._upstream_service()

        try:
            with ModelRequestInstrumentator(
                llm_provider=upstream_service,
                model_engine=upstream_service,
                model_provider=upstream_service,
                model_name=model_name,
                limits=self.limits.for_model(
                    engine=self._upstream_service(), name=model_name
                ),
            ).watch(
                stream=stream,
                unit_primitives=[GitLabUnitPrimitive.AI_GATEWAY_MODEL_PROVIDER_PROXY],
                internal_event_client=self.internal_event_client,
            ) as watcher:
                # Setup token tracking
                self.watcher = watcher
                self.user = request.user
                init_llm_operations()

                endpoint_func = pass_through_endpoints.create_pass_through_route(
                    endpoint="",  # This is unused in the litellm code
                    target=f"{self._base_url()}{upstream_path}",
                    custom_headers=headers_to_upstream,
                )
                response_from_upstream = await endpoint_func(
                    request,
                    None,  # FastAPI response object, to inject litellm headers into it, which we don't need
                    UserAPIKeyAuth(),  # LiteLLM-Proxy auth, which we don't use
                )
        except Exception:
            raise fastapi.HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Bad Gateway"
            )

        headers_to_downstream = self._create_headers_to_downstream(
            response_from_upstream.headers
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

    @abstractmethod
    def _base_url(self) -> str:
        """Base URL of the upstream service."""
        pass

    @abstractmethod
    def _allowed_upstream_paths(self) -> list[str]:
        """Allowed paths to the upstream service."""
        pass

    @abstractmethod
    def _allowed_headers_to_upstream(self) -> list[str]:
        """Allowed request headers to the upstream service."""
        pass

    @abstractmethod
    def _allowed_headers_to_downstream(self) -> list[str]:
        """Allowed response headers to the downstream service."""
        pass

    @abstractmethod
    def _upstream_service(self) -> str:
        """Name of the upstream service."""
        pass

    @abstractmethod
    def _allowed_upstream_models(self) -> list[str]:
        """Allowed models to the upstream service."""
        pass

    @abstractmethod
    def _extract_model_name(self, upstream_path: str, json_body: typing.Any) -> str:
        """Extract model name from the request."""
        pass

    @abstractmethod
    def _extract_stream_flag(self, upstream_path: str, json_body: typing.Any) -> bool:
        """Extract stream flag from the request."""
        pass

    @abstractmethod
    def _update_headers_to_upstream(self, headers: dict[str, str]) -> None:
        """Update headers for vendor specific requirements."""
        pass

    def _extract_upstream_path(self, request_path: str) -> str:
        path = re.sub(f"^(.*?)/{self._upstream_service()}/", "/", request_path)

        if path not in self._allowed_upstream_paths():
            raise fastapi.HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
            )

        return path

    async def _extract_json_body(self, request: fastapi.Request) -> typing.Any:
        body = await request.body()

        try:
            json_body = json.loads(body)
        except json.JSONDecodeError:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON"
            )

        return json_body

    def _create_headers_to_upstream(self, headers_from_downstream) -> dict[str, str]:
        return {
            key: headers_from_downstream[key]
            for key in self._allowed_headers_to_upstream()
            if key in headers_from_downstream
        }

    def _create_headers_to_downstream(self, headers_from_upstream) -> dict[str, str]:
        return {
            key: headers_from_upstream.get(key)
            for key in self._allowed_headers_to_downstream()
            if key in headers_from_upstream
        }


current_proxy_client: ContextVar[BaseProxyClient | None] = ContextVar(
    "current_proxy_client", default=None
)


def _get_async_httpx_client(*args, **kwargs):
    result = get_async_httpx_client(*args, **kwargs)

    proxy_client = current_proxy_client.get()
    if proxy_client:
        # Override client with our own. We've proposed to support this natively upstream.
        # See https://github.com/BerriAI/litellm/pull/19465
        result.client = proxy_client.client

    return result


pass_through_endpoints.get_async_httpx_client = _get_async_httpx_client


# Monkey-patches to support calling litellm-proxy's code without it handling fastapi routing
def _is_registered_pass_through_route(
    route: str,
) -> bool:  # pylint: disable=unused-argument
    return True


pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route = (
    _is_registered_pass_through_route
)


def _get_registered_pass_through_route(
    route: str,
) -> dict[str, typing.Any]:  # pylint: disable=unused-argument
    return {}


pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route = (
    _get_registered_pass_through_route
)


async def litellm_async_success_callback(
    _kwargs, completion_obj, _start_time, _end_time
):
    proxy_client = current_proxy_client.get()
    if not proxy_client:
        return

    usage = completion_obj.get("usage")
    if not usage:
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
    usage_metadata = _create_usage_metadata(usage)
    proxy_client.watcher.register_token_usage(
        completion_obj.get("model"), usage_metadata
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
