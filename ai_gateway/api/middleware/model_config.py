import json

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.model_metadata import IamRoleNotAllowedError, create_model_metadata
from lib.context import current_model_metadata_context


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
                    model_metadata = create_model_metadata(data["model_metadata"])
                    current_model_metadata_context.set(model_metadata)

            except IamRoleNotAllowedError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
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
