import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import structlog
from asgi_correlation_id.context import correlation_id
from fastapi import status
from langsmith.run_helpers import tracing_context
from starlette.datastructures import CommaSeparatedStrings, MutableHeaders
from starlette.middleware.base import Request
from starlette_context import context as starlette_context
from uvicorn.protocols.utils import get_path_with_query_string

from ai_gateway.internal_events import (
    EventContext,
    current_event_context,
    tracked_internal_events,
)
from ai_gateway.tracking.errors import log_exception

from .headers import (
    X_GITLAB_CLIENT_NAME,
    X_GITLAB_CLIENT_TYPE,
    X_GITLAB_CLIENT_VERSION,
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_INTERFACE,
    X_GITLAB_LANGUAGE_SERVER_VERSION,
    X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_TEAM_MEMBER_HEADER,
    X_GITLAB_VERSION_HEADER,
)

log = logging.getLogger("codesuggestions")
access_logger = structlog.stdlib.get_logger("api.access")


class _PathResolver:
    def __init__(self, endpoints: list[str]):
        self.endpoints = set(endpoints)

    @classmethod
    def from_optional_list(cls, endpoints: Optional[list] = None) -> "_PathResolver":
        if endpoints is None:
            endpoints = []
        return cls(endpoints)

    def skip_path(self, path: str) -> bool:
        return path in self.endpoints


class AccessLogMiddleware:
    """Middleware for access logging."""

    def __init__(self, app, skip_endpoints):
        self.app = app
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        # These context vars will be added to all log entries emitted during the request
        request_id = correlation_id.get()
        structlog.contextvars.bind_contextvars(correlation_id=request_id)

        start_time_total = time.perf_counter()
        start_time_cpu = time.process_time()
        response_start_duration_s = 0.0
        first_chunk_duration_s = 0.0
        request_arrived_at = datetime.now(timezone.utc)
        # duration_request represents latency added by sending request from Rails to AI gateway
        try:
            wait_duration = time.time() - float(
                request.headers.get(X_GITLAB_MODEL_GATEWAY_REQUEST_SENT_AT)
            )
        except (ValueError, TypeError):
            wait_duration = -1

        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        content_type = "unknown"

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                nonlocal status_code, response_start_duration_s, first_chunk_duration_s, content_type
                status_code = message["status"]

                headers = MutableHeaders(scope=message)

                if "content-type" in headers:
                    content_type = headers["content-type"]

                response_start_duration_s = time.perf_counter() - start_time_total
                headers.append("X-Process-Time", str(response_start_duration_s))

            if message["type"] == "http.response.body":
                if first_chunk_duration_s == 0.0:
                    first_chunk_duration_s = time.perf_counter() - start_time_total

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            if isinstance(e, BaseExceptionGroup):
                e = e.exceptions[0]
            starlette_context.data["exception_message"] = str(e)
            starlette_context.data["exception_class"] = type(e).__name__
            starlette_context.data["exception_backtrace"] = traceback.format_exc()
            log_exception(e)
            raise e
        finally:
            elapsed_time = time.perf_counter() - start_time_total
            cpu_time = time.process_time() - start_time_cpu
            url = get_path_with_query_string(request.scope)
            client_host = request.client.host
            client_port = request.client.port
            http_method = request.method
            http_version = request.scope["http_version"]

            fields = {
                "url": str(request.url),
                "path": url,
                "status_code": status_code,
                "method": http_method,
                "correlation_id": request_id,
                "http_version": http_version,
                "client_ip": client_host,
                "client_port": client_port,
                "duration_s": elapsed_time,
                "duration_request": wait_duration,
                "request_arrived_at": request_arrived_at.isoformat(),
                "response_start_duration_s": response_start_duration_s,
                "first_chunk_duration_s": first_chunk_duration_s,
                "cpu_s": cpu_time,
                "content_type": content_type,
                "user_agent": request.headers.get("User-Agent"),
                "gitlab_language_server_version": request.headers.get(
                    X_GITLAB_LANGUAGE_SERVER_VERSION
                ),
                "gitlab_instance_id": request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
                "gitlab_global_user_id": request.headers.get(
                    X_GITLAB_GLOBAL_USER_ID_HEADER
                ),
                "gitlab_host_name": request.headers.get(X_GITLAB_HOST_NAME_HEADER),
                "gitlab_version": request.headers.get(X_GITLAB_VERSION_HEADER),
                "gitlab_saas_duo_pro_namespace_ids": request.headers.get(
                    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER
                ),
                "gitlab_feature_enabled_by_namespace_ids": request.headers.get(
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER
                ),
                "gitlab_feature_enablement_type": request.headers.get(
                    X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER
                ),
                "gitlab_realm": request.headers.get(X_GITLAB_REALM_HEADER),
            }
            fields.update(starlette_context.data)

            # Recreate the Uvicorn access log format, but add all parameters as structured information
            access_logger.info(
                f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
                **fields,
            )


class InternalEventMiddleware:
    def __init__(self, app, skip_endpoints, enabled, environment):
        self.app = app
        self.enabled = enabled
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        # Fetching a list of namespaces that allow the user to use the tracked feature.
        # This is relevant for requests coming from gitlab.com, and unrelated to self-managed or dedicated instances.
        feature_enabled_by_namespace_ids = list(
            CommaSeparatedStrings(
                request.headers.get(
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER, ""
                )
            )
        )
        # Supporting the legacy header
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561.
        if not feature_enabled_by_namespace_ids:
            feature_enabled_by_namespace_ids = list(
                CommaSeparatedStrings(
                    request.headers.get(X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER, "")
                )
            )

        try:
            feature_enabled_by_namespace_ids = [
                int(str_id) for str_id in feature_enabled_by_namespace_ids
            ]
        except ValueError:
            feature_enabled_by_namespace_ids = None

        context = EventContext(
            environment=self.environment,
            source="ai-gateway-python",
            realm=request.headers.get(X_GITLAB_REALM_HEADER),
            instance_id=request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
            host_name=request.headers.get(X_GITLAB_HOST_NAME_HEADER),
            instance_version=request.headers.get(X_GITLAB_VERSION_HEADER),
            global_user_id=request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            is_gitlab_team_member=request.headers.get(X_GITLAB_TEAM_MEMBER_HEADER),
            client_type=request.headers.get(X_GITLAB_CLIENT_TYPE),
            client_name=request.headers.get(X_GITLAB_CLIENT_NAME),
            client_version=request.headers.get(X_GITLAB_CLIENT_VERSION),
            interface=request.headers.get(X_GITLAB_INTERFACE),
            feature_enabled_by_namespace_ids=feature_enabled_by_namespace_ids,
            feature_enablement_type=request.headers.get(
                X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER
            ),
            context_generated_at=datetime.now().isoformat(),
            correlation_id=correlation_id.get(),
        )
        current_event_context.set(context)
        tracked_internal_events.set(set())

        await self.app(scope, receive, send)

        starlette_context["tracked_internal_events"] = list(
            tracked_internal_events.get()
        )


class DistributedTraceMiddleware:
    """Middleware for distributed tracing."""

    def __init__(self, app, skip_endpoints, environment):
        self.app = app
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        if self.environment == "development" and "langsmith-trace" in request.headers:
            # Set the distributed tracing LangSmith header to the tracing context, which is sent from
            # Langsmith::RunHelpers of GitLab-Rails/Sidekiq.
            # See https://docs.gitlab.com/ee/development/ai_features/duo_chat.html#tracing-with-langsmith
            # and https://docs.smith.langchain.com/how_to_guides/tracing/distributed_tracing
            with tracing_context(parent=request.headers["langsmith-trace"]):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
