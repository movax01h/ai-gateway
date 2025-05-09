import asyncio
import os

import structlog
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from duo_workflow_service.internal_events.client import DuoWorkflowInternalEvent
from duo_workflow_service.llm_factory import validate_llm_access
from duo_workflow_service.monitoring import setup_monitoring
from duo_workflow_service.profiling import setup_profiling
from duo_workflow_service.servers import grpc_serve
from duo_workflow_service.servers.websocket_server import websocket_serve
from duo_workflow_service.structured_logging import setup_logging
from duo_workflow_service.tracking.sentry_error_tracking import setup_error_tracking

log = structlog.stdlib.get_logger("server")


def configure_cache() -> None:
    if os.environ.get("LLM_CACHE") == "true":
        set_llm_cache(SQLiteCache(database_path=".llm_cache.db"))
    else:
        set_llm_cache(None)


async def start_servers():
    tasks = []

    grpc_port = int(os.environ.get("PORT", "50052"))
    tasks.append(grpc_serve(grpc_port))

    if os.environ.get("WEBSOCKET_SERVER", "false").lower() == "true":
        ws_port = int(os.environ.get("WEBSOCKET_PORT", "8080"))
        tasks.append(websocket_serve(ws_port))

    await asyncio.gather(*tasks)


def run():
    load_dotenv()
    setup_profiling()
    setup_error_tracking()
    setup_monitoring()
    setup_logging(json_format=True, to_file=None)
    configure_cache()
    validate_llm_access()
    DuoWorkflowInternalEvent.setup()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_servers())
    except KeyboardInterrupt:
        log.info("Shutting down servers...")
    finally:
        loop.close()


if __name__ == "__main__":
    run()
