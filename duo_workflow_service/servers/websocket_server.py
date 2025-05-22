import asyncio
import time

import structlog
import uvicorn
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect

from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdMiddleware,
)
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    FeatureFlagMiddleware,
)
from duo_workflow_service.interceptors.websocket_middleware import MiddlewareChain

log = structlog.stdlib.get_logger("websocket_server")


class WebSocketServer:
    def __init__(self):
        self.app = FastAPI()
        self.active_connection = None
        self.active_connection_lock = asyncio.Lock()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(
            websocket: WebSocket,
            _=Depends(
                MiddlewareChain(
                    [
                        CorrelationIdMiddleware(),
                        FeatureFlagMiddleware(),
                    ]
                )
            ),
        ):
            # Only allow one active connection at a time
            async with self.active_connection_lock:
                if self.active_connection is not None:
                    log.warning(
                        "Connection rejected: Another client is already connected"
                    )
                    await websocket.close(
                        code=1008, reason="Another client is already connected"
                    )
                    return

                await websocket.accept()
                self.active_connection = websocket

            try:
                client_ip = websocket.client.host if websocket.client else "unknown"
                log.info(f"WebSocket connection established from {client_ip}")
                await websocket.send_text("Hello World!")

                # Track last ping response time
                last_pong_time = time.time()
                last_ping_time = 0
                heartbeat_interval = 10  # seconds
                pong_timeout = 15  # seconds

                while True:
                    current_time = time.time()

                    # Check if we need to send a heartbeat
                    if current_time - last_ping_time > heartbeat_interval:
                        await websocket.send_text("ping")
                        log.debug("Sent ping")
                        last_ping_time = current_time  # type: ignore

                    # Check if we've missed too many pongs
                    if current_time - last_pong_time > pong_timeout:
                        log.warning(
                            f"No pong received in {pong_timeout} seconds, closing connection"
                        )
                        await websocket.close(code=1001, reason="Client not responding")
                        break

                    # Wait for a message with a short timeout to allow regular heartbeat checks
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=min(1.0, max(0.1, heartbeat_interval / 2)),
                        )
                        log.info(f"Received message: {message}")

                        # Handle pong messages and regular messages
                        if message == "pong":
                            log.debug("Received pong")
                            last_pong_time = time.time()  # Update last pong time
                        else:
                            await websocket.send_text(f"Echo: {message}")
                            last_pong_time = time.time()

                    except asyncio.TimeoutError:
                        # This is normal - just continue the loop to check heartbeats
                        continue
                    except WebSocketDisconnect:
                        log.info("WebSocket disconnected by client")
                        break

            except Exception as e:
                log.error(f"WebSocket error: {str(e)}")
            finally:
                async with self.active_connection_lock:
                    if self.active_connection == websocket:
                        self.active_connection = None

                log.info("WebSocket connection closed")

    async def run(self, port: int = 8080):
        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=port)
        server = uvicorn.Server(config)
        log.info(f"Starting WebSocket server on port {port}")
        await server.serve()


async def websocket_serve(port: int = 8080):
    server = WebSocketServer()
    await server.run(port)
