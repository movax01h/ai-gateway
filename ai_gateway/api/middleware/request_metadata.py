from starlette.requests import Request

from lib.context import gitlab_user_id, is_gitlab_team_member

from .headers import X_GITLAB_TEAM_MEMBER_HEADER, X_GITLAB_USER_ID_HEADER


class RequestMetadataMiddleware:
    """Reads request metadata headers into the shared ContextVars.

    - ``X-Gitlab-Is-Team-Member`` -> ``is_gitlab_team_member`` (instrumentation)
    - ``x-gitlab-user-id`` -> ``gitlab_user_id`` (per-user identity forwarding to custom models)
    """

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

        gitlab_user_id.set(request.headers.get(X_GITLAB_USER_ID_HEADER) or None)

        await self.app(scope, receive, send)
