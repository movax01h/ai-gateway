import contextvars
import os
from typing import Callable

import grpc
import structlog
from gitlab_cloud_connector import (
    AuthProvider,
    CompositeProvider,
    GitLabOidcProvider,
    LocalAuthProvider,
    authenticate,
)
from grpc.aio import ServicerContext

current_user: contextvars.ContextVar = contextvars.ContextVar("current_user")


class AuthenticationError(Exception):
    pass


class AuthenticationInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self):
        pass

    async def intercept_service(
        self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        if os.environ.get("DUO_WORKFLOW_AUTH__ENABLED", True) == "false":
            print("[WARN] Auth is disabled, all users allowed")
            cloud_connector_user, _cloud_connector_error = authenticate(
                {}, None, bypass_auth=True
            )
            current_user.set(cloud_connector_user)

            return await continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata)

        cloud_connector_user, cloud_connector_error = authenticate(
            metadata, self._oidc_auth_provider()
        )

        if cloud_connector_error:
            return self._abort_handler(
                grpc.StatusCode.UNAUTHENTICATED, cloud_connector_error.error_message
            )

        current_user.set(cloud_connector_user)
        return await continuation(handler_call_details)

    def _abort_handler(
        self, code: grpc.StatusCode, details: str
    ) -> grpc.RpcMethodHandler:
        async def handler(request: object, context: ServicerContext) -> object:
            await context.abort(code, details)
            return None

        return grpc.unary_unary_rpc_method_handler(handler)

    def _oidc_auth_provider(self) -> AuthProvider:
        gitlab_url: str = os.environ.get(
            "DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL", "https://gitlab.com"
        )
        customer_portal_url: str = os.environ.get(
            "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL",
            "https://customers.gitlab.com",
        )
        signing_key: str = os.environ.get(
            "DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY", ""
        )
        validation_key: str = os.environ.get(
            "DUO_WORKFLOW_SELF_SIGNED_JWT__VALIDATION_KEY", ""
        )

        return CompositeProvider(
            [
                LocalAuthProvider(structlog, signing_key, validation_key),
                GitLabOidcProvider(
                    structlog,
                    oidc_providers={
                        "Gitlab": gitlab_url,
                        "CustomersDot": customer_portal_url,
                    },
                ),
            ],
            structlog,
        )
