import json
from typing import Optional

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.feature_allowlist import (
    ClientModelRequest,
    check_client_model_metadata,
    note_prompt_feature,
)
from ai_gateway.model_metadata import IamRoleNotAllowedError, create_model_metadata
from ai_gateway.model_selection import ModelSelectionError
from ai_gateway.prompts.registry import feature_setting_for_prompt_id
from lib.context import StarletteUser, current_model_metadata_context

# Routes that serve one feature. The allowlist check uses this feature, so the client's
# `feature_setting` does not select the allowlist on these routes. GitLab Rails sends code
# embeddings requests without a `feature_setting`.
_ROUTE_FEATURE_SETTINGS = {
    "/v1/embeddings/code_embeddings": "embeddings_code",
}

# Routes whose path ends with a prompt ID. The middleware logs a note when the feature of the
# prompt would not allow the requested model, but the check still uses the client's feature.
_PROMPT_ROUTES = ("/v1/prompts/", "/v2/prompts/")


class ModelConfigMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def fetch_model_metadata() -> Message | JSONResponse:
            body_parts = []
            max_chunks = 1000
            chunk_count = 0

            while chunk_count < max_chunks:
                chunk_count += 1
                message = await receive()

                if message["type"] == "http.request":
                    body_part = message.get("body", b"")
                    if body_part:
                        body_parts.append(body_part)

                    if not message.get("more_body", False):
                        break
                elif body_parts:
                    continue
                else:
                    return message

            full_body = b"".join(body_parts) if body_parts else b""

            replay: Message = {
                "type": "http.request",
                "body": full_body,
                "more_body": False,
            }

            if b"model_metadata" not in full_body:
                return replay

            try:
                body_str = full_body.decode("utf-8")
                data = json.loads(body_str)

                if "model_metadata" in data:
                    raw_metadata = data["model_metadata"]
                    # Read the request first: create_model_metadata changes the mapping it gets.
                    request = ClientModelRequest.from_raw(raw_metadata)
                    client_model_metadata = create_model_metadata(raw_metadata)
                    path = scope.get("path", "")
                    if prompt_id := _route_prompt_id(path):
                        note_prompt_feature(
                            request,
                            client_model_metadata,
                            prompt_feature_setting=feature_setting_for_prompt_id(
                                prompt_id
                            ),
                            log_context={"path": path, "prompt_id": prompt_id},
                        )
                    model_metadata = check_client_model_metadata(
                        request,
                        client_model_metadata,
                        surface="http",
                        feature_setting=_route_feature_setting(path),
                        user=_authenticated_user(scope),
                        log_context={"path": path},
                    )
                    # None leaves the request on the prompt's own default model.
                    current_model_metadata_context.set(model_metadata)

            except IamRoleNotAllowedError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
            except ModelSelectionError as exc:
                return JSONResponse(status_code=403, content={"detail": str(exc)})
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                pass

            return replay

        first = await fetch_model_metadata()
        if isinstance(first, JSONResponse):
            await first(scope, receive, send)
            return

        pending: list[Message] = [first]

        async def replay_receive() -> Message:
            if pending:
                return pending.pop()
            return await receive()

        await self.app(scope, replay_receive, send)


def _authenticated_user(scope: Scope) -> Optional[StarletteUser]:
    """Return the user that the authentication middleware set, or None for an unauthenticated request."""
    user = scope.get("user")
    return user if isinstance(user, StarletteUser) else None


def _route_feature_setting(path: str) -> Optional[str]:
    """Return the feature that the route at ``path`` serves, or None when the route serves more than one feature."""
    for route, feature_setting in _ROUTE_FEATURE_SETTINGS.items():
        if path == route or path.startswith(f"{route}/"):
            return feature_setting
    return None


def _route_prompt_id(path: str) -> Optional[str]:
    """Return the prompt ID of a prompt route at ``path``, or None for any other route."""
    for prefix in _PROMPT_ROUTES:
        if path.startswith(prefix) and (prompt_id := path.removeprefix(prefix)):
            return prompt_id
    return None
