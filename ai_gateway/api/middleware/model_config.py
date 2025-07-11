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
            message = await receive()

            body: bytes = message.get("body", b"")

            if b"model_metadata" not in body:
                return message

            body_str: str = body.decode("utf-8")

            try:
                data = json.loads(body_str)
            except json.JSONDecodeError:
                data = {}

            if "model_metadata" in data:
                current_model_metadata_context.set(
                    create_model_metadata(data["model_metadata"])
                )
            return message

        await self.app(scope, fetch_model_metadata, send)
