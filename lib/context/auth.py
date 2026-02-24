"""Authentication context variables and user classes shared between services.

Provides the cloud connector token context variable and the StarletteUser wrapper class used for request authentication.
"""

import contextvars
from typing import Optional, override

from fastapi import Request
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from starlette.authentication import BaseUser

# Context variable for cloud connector token
cloud_connector_token_context_var: contextvars.ContextVar = contextvars.ContextVar(
    "cloud_connector_token"
)


class StarletteUser(BaseUser):
    """Wrapper around CloudConnectorUser for Starlette authentication."""

    def __init__(
        self,
        cloud_connector_user: CloudConnectorUser,
        cloud_connector_token: Optional[str] = None,
    ):
        self.cloud_connector_user = cloud_connector_user
        self.cloud_connector_token = cloud_connector_token

    @property
    @override
    def is_authenticated(self) -> bool:
        return self.cloud_connector_user.is_authenticated

    @property
    def global_user_id(self) -> str | None:
        return self.cloud_connector_user.global_user_id

    @property
    def claims(self) -> Optional[UserClaims]:
        return self.cloud_connector_user.claims

    @property
    def is_debug(self) -> bool:
        return self.cloud_connector_user.is_debug

    @property
    def unit_primitives(self) -> list[GitLabUnitPrimitive]:
        return self.cloud_connector_user.unit_primitives

    def can(
        self,
        unit_primitive: GitLabUnitPrimitive,
        disallowed_issuers: Optional[list[str]] = None,
    ) -> bool:
        return self.cloud_connector_user.can(
            unit_primitive,
            disallowed_issuers,
        )


async def get_current_user(request: Request) -> StarletteUser:
    """Get the current authenticated user from the request."""
    return request.user
