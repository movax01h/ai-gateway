from logging.config import dictConfig

from dotenv import load_dotenv
from fastapi import FastAPI

from ai_gateway.api import create_fast_api_server
from ai_gateway.config import Config
from ai_gateway.prometheus import start_metrics_server
from ai_gateway.structured_logging import setup_logging

# load env variables from .env if exists
load_dotenv()

# prepare configuration settings
config = Config()

# configure logging
dictConfig(config.fastapi.uvicorn_logger)


def get_config() -> Config:
    return config


def get_app() -> FastAPI:
    setup_logging(config.logging, config.custom_models.enabled)
    start_metrics_server(config)
    app = create_fast_api_server(config)
    return app
