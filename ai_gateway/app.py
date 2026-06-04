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

    if config.mock_usage_credits:
        from lib.usage_quota.mock_server import (  # pylint: disable=import-outside-toplevel
            start_mock_server,
        )

        start_mock_server(port=config.mock_usage_quota_server.port)

    app = create_fast_api_server(config)
    return app
