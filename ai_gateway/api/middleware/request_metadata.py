from starlette.requests import Request

from lib.context import is_gitlab_team_member

from .headers import X_GITLAB_TEAM_MEMBER_HEADER


class RequestMetadataMiddleware:
    """Reads the X-Gitlab-Is-Team-Member header and sets the shared is_gitlab_team_member ContextVar for
    instrumentation."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        team_member_value = request.headers.get(X_GITLAB_TEAM_MEMBER_HEADER)
        if team_member_value is not None:
            is_gitlab_team_member.set(team_member_value.lower() == "true")
        else:
            is_gitlab_team_member.set(None)

        await self.app(scope, receive, send)
