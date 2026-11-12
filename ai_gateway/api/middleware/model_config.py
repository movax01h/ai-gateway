import json
from typing import Optional

import structlog
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette_context import context as starlette_context

from ai_gateway.api.middleware.base import _PathResolver
from ai_gateway.feature_allowlist import (
    ClientModelRequest,
    PromptModelCheck,
    check_client_model_metadata,
    prompt_model_check_context,
)
from ai_gateway.model_metadata import (
    IamRoleNotAllowedError,
    create_model_metadata,
    drop_default_model_ref,
)
from ai_gateway.model_selection import ModelSelectionError
from lib.context import StarletteUser, current_model_metadata_context

log = structlog.stdlib.get_logger("model_config")

# Routes that serve one feature. The allowlist check uses this feature, so the client's
# `feature_setting` does not select the allowlist on these routes. GitLab Rails sends code
# embeddings requests without a `feature_setting`.
_ROUTE_FEATURE_SETTINGS = {
    "/v1/embeddings/code_embeddings": "embeddings_code",
}

# Routes whose path ends with a prompt ID. The allowlist depends on the prompt version, which only the
# route resolves, so the route checks the model metadata of these requests (`PromptModelCheck`).
_PROMPT_ROUTES = ("/v1/prompts/", "/v2/prompts/")

# Routes that forward the body to a model provider. They do not use `model_metadata`, so the middleware does not parse
# their body. It still reads the body, so the access log records its size.
_PROXY_ROUTES = ("/v1/proxy/",)


class ModelConfigMiddleware:
    def __init__(self, app: ASGIApp, skip_endpoints: Optional[list[str]] = None):
        self.app = app
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        # GET routes and the skipped endpoints do not use `model_metadata`. Do not read the body
        # for them, because the skipped endpoints accept requests without authentication.
        if (
            scope["type"] != "http"
            or scope["method"] == "GET"
            or self.path_resolver.skip_path(scope["path"])
        ):
            await self.app(scope, receive, send)
            return

        async def fetch_model_metadata() -> Message | JSONResponse:
            prompt_model_check_context.set(None)
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
            # The access log records these fields. We use them to choose a size cap for the parse.
            starlette_context["request_body_bytes"] = len(full_body)
            starlette_context["request_body_has_model_metadata"] = False

            replay: Message = {
                "type": "http.request",
                "body": full_body,
                "more_body": False,
            }

            if (
                scope["path"].startswith(_PROXY_ROUTES)
                or b"model_metadata" not in full_body
            ):
                return replay

            request: Optional[ClientModelRequest] = None
            try:
                body_str = full_body.decode("utf-8")
                data = json.loads(body_str)
                # A JSON array or string can contain the text `model_metadata`, but it has no such key.
                if not isinstance(data, dict):
                    return replay

                if "model_metadata" in data:
                    starlette_context["request_body_has_model_metadata"] = True
                    raw_metadata = data["model_metadata"]
                    # The load-balanced default is no model choice, so the check sees an unpinned request.
                    drop_default_model_ref(raw_metadata)
                    # Read the request first: create_model_metadata changes the mapping it gets.
                    request = ClientModelRequest.from_raw(raw_metadata)
                    client_model_metadata = create_model_metadata(raw_metadata)
                    path = scope.get("path", "")
                    if prompt_id := _route_prompt_id(path):
                        prompt_model_check_context.set(
                            PromptModelCheck(
                                request=request,
                                metadata=client_model_metadata,
                                user=_authenticated_user(scope),
                                log_context={"path": path, "prompt_id": prompt_id},
                            )
                        )
                    else:
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
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            except ValueError as error:
                # The request runs without the client's model metadata, for example when a `gitlab` request names
                # an identifier that is not in models.yml. Never log the raw metadata or the error text: both can
                # carry a provider API key.
                log.warning(
                    "Model metadata not applied",
                    path=scope.get("path", ""),
                    error_type=type(error).__name__,
                    provider=request.provider if request else None,
                    requested_name=request.name if request else None,
                    requested_identifier=request.identifier if request else None,
                    feature_setting=request.feature_setting if request else None,
                )

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
