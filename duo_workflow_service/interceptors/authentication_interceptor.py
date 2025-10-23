# pylint: disable=direct-environment-variable-reference

import contextvars
import os
from typing import Callable, Dict

import grpc
import structlog
from gitlab_cloud_connector import (
    AuthProvider,
    CompositeProvider,
    GitLabOidcProvider,
    LocalAuthProvider,
    authenticate,
)
from gitlab_cloud_connector.auth import AUTH_HEADER, PREFIX_BEARER_HEADER
from grpc.aio import ServicerContext

from ai_gateway.auth.glgo import cloud_connector_token_context_var

current_user: contextvars.ContextVar = contextvars.ContextVar("current_user")


class AuthenticationError(Exception):
    pass


class AuthenticationInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self):
        self.oidc_auth_provider = self._init_oidc_auth_provider()

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
            metadata, self.oidc_auth_provider
        )

        cloud_connector_token_context_var.set(
            self._extract_cloud_connector_token(metadata)
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
        # pylint: disable=unused-argument
        async def handler(request: object, context: ServicerContext) -> object:
            await context.abort(code, details)
            return None

        return grpc.unary_unary_rpc_method_handler(handler)

    def _init_oidc_auth_provider(self) -> AuthProvider:
        # Reuse the AIGW_GITLAB_URL so that GitLab Self-Hosted Duo customers can
        # use the same URL for both AIGW and Duo Workflow Service
        gitlab_url: str = (
            os.environ.get("DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL")
            or os.environ.get("AIGW_GITLAB_URL")
            or "https://gitlab.com"
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

    def _extract_cloud_connector_token(self, headers: Dict[str, str]) -> str:
        auth_header = headers.get(AUTH_HEADER.lower())
        if not auth_header:
            return ""

        bearer, _, token = auth_header.partition(" ")
        if bearer.lower() != PREFIX_BEARER_HEADER:
            return ""

        return token
