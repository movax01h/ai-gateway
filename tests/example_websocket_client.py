import argparse
import asyncio
import logging

import websockets

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("websocket_client")


async def connect_to_server(host, port):
    uri = f"ws://{host}:{port}/ws"
    logger.info("Attempting to connect to %s", uri)

    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")

            # Message handling loop - single receiver to avoid concurrency issues
            try:
                hello_msg = "Hello from client!"
                await websocket.send(hello_msg)
                logger.info("Sent message: %s", hello_msg)

                # Main message loop
                last_ping_time = 0
                while True:
                    message = await websocket.recv()
                    logger.info("Received from server: %s", message)

                    if message == "ping":
                        logger.info("Responding to ping with pong")
                        await websocket.send("pong")

                    # Send a regular message every 10 seconds to keep the connection alive
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_ping_time > 10:
                        message = f"Hello server at {current_time}"
                        await websocket.send(message)
                        logger.info("Sent periodic message: %s", message)
                        last_ping_time = current_time  # type: ignore

            except websockets.exceptions.ConnectionClosed as e:
                logger.info("Connection closed: %s", e)

    except websockets.exceptions.ConnectionClosed as e:
        logger.error("Connection closed: %s", e)
    except websockets.exceptions.WebSocketException as e:
        logger.error("WebSocket error: %s", e)
    except ConnectionRefusedError:
        logger.error(
            "Connection refused to %s. Make sure the server is running and the path is correct.",
            uri,
        )
    except Exception as e:
        logger.error("Unexpected error: %s: %s", type(e).__name__, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket client for testing")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8080, help="Port to connect to")
    args = parser.parse_args()

    try:
        logger.info("Starting client connecting to %s:%s", args.host, args.port)
        asyncio.run(connect_to_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
