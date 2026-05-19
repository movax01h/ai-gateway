from logging.config import dictConfig

from fastapi import FastAPI

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import get_config
from ai_gateway.prometheus import start_metrics_server
from ai_gateway.structured_logging import setup_logging

config = get_config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def get_app() -> FastAPI:
    setup_logging(config.logging, config.custom_models.enabled)
    start_metrics_server(config)
    app = create_fast_api_server(config)
    return app
