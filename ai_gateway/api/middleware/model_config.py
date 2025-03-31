import json

from ai_gateway.model_metadata import (
    create_model_metadata,
    current_model_metadata_context,
)


class ModelConfigMiddleware:
    def __init__(self, app, custom_models_enabled=False):
        self.app = app
        self.custom_models_enabled = custom_models_enabled

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.custom_models_enabled:
            await self.app(scope, receive, send)
            return

        async def fetch_model_metadata():
            message = await receive()

            body: bytes = message.get("body", b"")

            if b"model_metadata" not in body:
                return message

            body: str = body.decode("utf-8")

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {}

            if "model_metadata" in data:
                current_model_metadata_context.set(
                    create_model_metadata(data["model_metadata"])
                )
            return message

        await self.app(scope, fetch_model_metadata, send)
