import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_gateway.model_metadata import (
    create_model_metadata,
    current_model_metadata_context,
)


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

        async def fetch_model_metadata() -> Message:
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

            if b"model_metadata" not in full_body:
                return {"type": "http.request", "body": full_body, "more_body": False}

            try:
                body_str = full_body.decode("utf-8")
                data = json.loads(body_str)

                if "model_metadata" in data:
                    model_metadata = create_model_metadata(data["model_metadata"])
                    current_model_metadata_context.set(model_metadata)

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            return {"type": "http.request", "body": full_body, "more_body": False}

        await self.app(scope, fetch_model_metadata, send)
