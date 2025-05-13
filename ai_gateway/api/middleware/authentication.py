import logging
from typing import Optional, Tuple

from fastapi import status
from fastapi.encoders import jsonable_encoder
from gitlab_cloud_connector import (
    AuthProvider,
    CloudConnectorAuthError,
    CloudConnectorUser,
)
from gitlab_cloud_connector import authenticate as cloud_connector_authenticate
from gitlab_cloud_connector.auth import AUTH_HEADER
from starlette.authentication import (
    AuthCredentials,
    AuthenticationError,
    HTTPConnection,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import (
    AuthenticationBackend,
    AuthenticationMiddleware,
)
from starlette.middleware.base import Request
from starlette.responses import JSONResponse
from starlette_context import context as starlette_context

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.middleware.base import _PathResolver
from ai_gateway.api.timing import timing

log = logging.getLogger("codesuggestions")


class MiddlewareAuthentication(Middleware):
    class AuthBackend(AuthenticationBackend):
        def __init__(
            self,
            oidc_auth_provider: AuthProvider,
            bypass_auth: bool,
            bypass_auth_with_header: bool,
            path_resolver: _PathResolver,
        ):
            self.oidc_auth_provider = oidc_auth_provider
            self.bypass_auth = bypass_auth
            self.bypass_auth_with_header = bypass_auth_with_header
            self.path_resolver = path_resolver

        async def authenticate(
            self, conn: HTTPConnection
        ) -> Optional[Tuple[AuthCredentials, StarletteUser]]:
            """
            Ref: https://www.starlette.io/authentication/
            """

            if self.path_resolver.skip_path(conn.url.path):
                return None

            if self.bypass_auth:
                log.critical("Auth is disabled, all users allowed")
                (
                    cloud_connector_user,
                    _cloud_connector_error,
                ) = cloud_connector_authenticate(
                    dict(conn.headers), None, bypass_auth=True
                )

                return AuthCredentials(), StarletteUser(cloud_connector_user)

            if (
                self.bypass_auth_with_header  # Should only be set and used for test & dev
                and conn.headers.get("Bypass-Auth") == "true"
            ):
                log.critical(
                    "Auth is disabled, all requests with `Bypass-Auth` header set allowed"
                )
                (
                    cloud_connector_user,
                    _cloud_connector_error,
                ) = cloud_connector_authenticate(
                    dict(conn.headers), None, bypass_auth=True
                )

                return AuthCredentials(), StarletteUser(cloud_connector_user)

            cloud_connector_user, cloud_connector_error = self.cloud_connector_auth(
                conn.headers
            )

            if hasattr(cloud_connector_user.claims, "issuer"):
                starlette_context["token_issuer"] = cloud_connector_user.claims.issuer

            # We will send this with an HTTP header field going forward since we are
            # retiring direct access to the gateway from clients, which was the main
            # reason this value was carried in the access token.
            if (
                hasattr(cloud_connector_user.claims, "gitlab_realm")
                and cloud_connector_user.claims.gitlab_realm
            ):
                starlette_context["gitlab_realm"] = (
                    cloud_connector_user.claims.gitlab_realm
                )

            if cloud_connector_error:
                raise AuthenticationError(cloud_connector_error.error_message)

            auth_header = conn.headers.get(AUTH_HEADER)

            # The auth header is already validated as part of the above `cloud_connector_auth` call. It is
            # guaranteed that the value is present. This assert is a safeguard against changes in the
            # external`cloud_connector_auth` call and appeases type checker.
            assert auth_header
            _, _, cloud_connector_token = auth_header.partition(" ")
            return AuthCredentials(cloud_connector_user.claims.scopes), StarletteUser(
                cloud_connector_user,
                cloud_connector_token,
            )

        @timing("auth_duration_s")
        def cloud_connector_auth(
            self, headers
        ) -> Tuple[CloudConnectorUser, Optional[CloudConnectorAuthError]]:
            return cloud_connector_authenticate(dict(headers), self.oidc_auth_provider)

    @staticmethod
    def on_auth_error(_: Request, e: Exception):
        content = jsonable_encoder({"error": str(e)})
        starlette_context["auth_error_details"] = str(e)
        starlette_context["http_exception_details"] = str(e)
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content=content)

    def __init__(
        self,
        oidc_auth_provider: AuthProvider,
        bypass_auth: bool = False,
        bypass_auth_with_header: bool = False,
        skip_endpoints: Optional[list] = None,
    ):
        path_resolver = _PathResolver.from_optional_list(skip_endpoints)

        super().__init__(
            AuthenticationMiddleware,
            backend=MiddlewareAuthentication.AuthBackend(
                oidc_auth_provider,
                bypass_auth,
                bypass_auth_with_header,
                path_resolver,
            ),
            on_error=MiddlewareAuthentication.on_auth_error,
        )
