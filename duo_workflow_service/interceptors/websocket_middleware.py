from typing import Sequence

from fastapi import WebSocket


class WebSocketMiddleware:
    """Base class for WebSocket middleware."""

    async def __call__(self, websocket: WebSocket):
        pass


class MiddlewareChain:
    """
    A manager class that runs multiple WebSocket middlewares in sequence.

    Since FastAPI's built-in middleware system doesn't work with WebSocket connections,
    we need to use the `Depends` instead. This class serves as a workaround to
    chain multiple middleware components together in a list format, rather than
    having to declare them as separate parameters in each WebSocket endpoint.

    The chain preserves the execution order of middlewares and maintains a similar
    pattern to our gRPC interceptors.

    Usage:
        middleware_chain = MiddlewareChain([
            CorrelationIdMiddleware(),
            AuthenticationMiddleware(),
            MonitoringMiddleware(),
        ])

        @app.websocket("/ws")
        async def websocket_endpoint(
                websocket: WebSocket,
                _=Depends(middleware_chain),
        ):
            await websocket.accept()
            # Handle WebSocket connection...
    """

    def __init__(self, middlewares: Sequence[WebSocketMiddleware]):
        self.middlewares = middlewares

    async def __call__(self, websocket: WebSocket):
        """
        Execute all middlewares in sequence, passing all parameters to each.

        Args:
            websocket: The WebSocket connection
        """
        for middleware in self.middlewares:
            await middleware(websocket)
