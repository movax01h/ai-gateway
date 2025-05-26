import asyncio
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, cast

import litellm
import structlog
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from gitlab_cloud_connector import (
    CompositeProvider,
    GitLabOidcProvider,
    LocalAuthProvider,
)
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette_context import context
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.middleware import (
    AccessLogMiddleware,
    DistributedTraceMiddleware,
    FeatureFlagMiddleware,
    InternalEventMiddleware,
    MiddlewareAuthentication,
    ModelConfigMiddleware,
)
from ai_gateway.api.middleware.self_hosted_logging import (
    EnabledInstanceVerboseAiLogsHeaderPlugin,
)
from ai_gateway.api.monitoring import router as http_monitoring_router
from ai_gateway.api.server_utils import extract_retry_after_header
from ai_gateway.api.v1 import api_router as http_api_router_v1
from ai_gateway.api.v2 import api_router as http_api_router_v2
from ai_gateway.api.v3 import api_router as http_api_router_v3
from ai_gateway.api.v4 import api_router as http_api_router_v4
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.instrumentators.threads import monitor_threads
from ai_gateway.models import ModelAPIError
from ai_gateway.models.base import ModelAPICallError
from ai_gateway.profiling import setup_profiling
from ai_gateway.structured_logging import can_log_request_data, setup_app_logging

__all__ = [
    "create_fast_api_server",
]

_SKIP_ENDPOINTS = ["/monitoring/healthz", "/monitoring/ready", "/metrics"]

ExceptionHandler = Callable[[Request, Exception], Awaitable[Response]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.extra["extra"]["config"]
    container_application = ContainerApplication()
    container_application.config.from_dict(config.model_dump())

    if config.instrumentator.thread_monitoring_enabled:
        loop = asyncio.get_running_loop()
        loop.create_task(
            monitor_threads(
                loop, interval=config.instrumentator.thread_monitoring_interval
            )
        )

    setup_litellm(config)

    yield


def create_fast_api_server(config: Config):
    auth_provider = CompositeProvider(
        [
            LocalAuthProvider(
                structlog,
                signing_key=config.self_signed_jwt.signing_key,
                validation_key=config.self_signed_jwt.validation_key,
            ),
            GitLabOidcProvider(
                structlog,
                oidc_providers={
                    "Gitlab": config.gitlab_url,
                    "CustomersDot": config.customer_portal_url,
                },
            ),
        ],
        structlog,
        bypass_auth_jwt_signature=config.auth.bypass_jwt_signature,
    )

    fastapi_app = FastAPI(
        title="GitLab AI Gateway",
        description="GitLab AI Gateway API to execute AI actions",
        openapi_url=config.fastapi.openapi_url,
        docs_url=config.fastapi.docs_url,
        redoc_url=config.fastapi.redoc_url,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        lifespan=lifespan,
        middleware=[
            Middleware(
                RawContextMiddleware,
                plugins=(EnabledInstanceVerboseAiLogsHeaderPlugin(),),
            ),
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["POST"],
                allow_headers=["*"],
            ),
            Middleware(
                AccessLogMiddleware,
                skip_endpoints=[],
            ),
            Middleware(
                DistributedTraceMiddleware,
                skip_endpoints=_SKIP_ENDPOINTS,
                environment=config.environment,
            ),
            MiddlewareAuthentication(
                auth_provider,
                bypass_auth=config.auth.bypass_external,
                bypass_auth_with_header=config.auth.bypass_external_with_header,
                skip_endpoints=_SKIP_ENDPOINTS,
            ),
            Middleware(
                FeatureFlagMiddleware,
                disallowed_flags=config.feature_flags.disallowed_flags,
            ),
            Middleware(
                InternalEventMiddleware,
                skip_endpoints=_SKIP_ENDPOINTS,
                enabled=config.internal_event.enabled,
                environment=config.environment,
            ),
            Middleware(ModelConfigMiddleware),
        ],
        extra={"config": config},
    )

    fastapi_app.state.cloud_connector_auth_provider = (
        auth_provider  # For readiness check
    )
    setup_custom_exception_handlers(fastapi_app)
    setup_router(fastapi_app)
    setup_app_logging(fastapi_app)
    setup_prometheus_fastapi_instrumentator(fastapi_app)
    setup_profiling(config.google_cloud_profiler)
    setup_gcp_service_account(config)

    return fastapi_app


async def custom_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    context["http_exception_details"] = str(exc)
    return await http_exception_handler(request, exc)


async def model_api_exception_handler(request: Request, exc: ModelAPIError) -> Response:
    if isinstance(exc, ModelAPICallError) and exc.code == 429:
        wrapped_exception = StarletteHTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
        )

        # When a 429 error is returned from some API, like Anthropic
        # the response includes a retry-after header
        # which we propagate to the client
        # https://docs.anthropic.com/en/api/rate-limits#response-headers
        retry_after = extract_retry_after_header(exc)

        response = await http_exception_handler(request, wrapped_exception)

        if retry_after:
            response.headers["Retry-After"] = retry_after

        return response

    # Default to 503 for all other error types
    wrapped_exception = StarletteHTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Inference failed",
    )
    return await http_exception_handler(request, wrapped_exception)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    if can_log_request_data():
        context["exception_message"] = str(exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error"},
    )


def setup_custom_exception_handlers(app: FastAPI):
    app.add_exception_handler(
        StarletteHTTPException, cast(ExceptionHandler, custom_http_exception_handler)
    )
    app.add_exception_handler(
        ModelAPIError, cast(ExceptionHandler, model_api_exception_handler)
    )
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_exception_handler)
    )


def setup_litellm(config: Config):
    litellm.vertex_project = config.google_cloud_platform.project


def setup_router(app: FastAPI):
    sub_router = APIRouter()
    sub_router.include_router(http_api_router_v1, prefix="/v1")
    sub_router.include_router(http_api_router_v2, prefix="/v2")
    sub_router.include_router(http_api_router_v3, prefix="/v3")
    sub_router.include_router(http_api_router_v4, prefix="/v4")
    sub_router.include_router(http_monitoring_router)

    app.include_router(sub_router)


def setup_prometheus_fastapi_instrumentator(app: FastAPI):
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=False,
        excluded_handlers=_SKIP_ENDPOINTS,
    )
    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            should_exclude_streaming_duration=True,
            buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
        )
    )
    instrumentator.instrument(app)


def setup_gcp_service_account(config: Config):
    """
    Inject service account credential from the `AIGW_GOOGLE_CLOUD_PLATFORM__SERVICE_ACCOUNT_JSON_KEY` environment variable.
    This method should only be used for testing purpose such as CI/CD pipelines.
    For production environment, we don't use this method but use Application Default Credentials (ADC) authentication instead.
    """
    if config.google_cloud_platform.service_account_json_key:
        with open("/tmp/gcp-service-account.json", "w") as f:
            f.write(config.google_cloud_platform.service_account_json_key.strip("'"))
            # pylint: disable=direct-environment-variable-reference
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                "/tmp/gcp-service-account.json"
            )
            # pylint: enable=direct-environment-variable-reference
