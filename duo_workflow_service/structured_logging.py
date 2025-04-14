import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer
from structlog.typing import Processor

from duo_workflow_service.interceptors.correlation_id_interceptor import (
    correlation_id,
    gitlab_global_user_id,
)

_workflow_id: ContextVar[str] = ContextVar("workflow_id", default="undefined")


def set_workflow_id(wrk_id: str):
    _workflow_id.set(wrk_id)


class LoggingConfig:
    level: int
    json_format: bool
    to_file: Optional[str]
    env: str

    def __init__(self, **kwargs):
        self.level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper())
        self.json_format = kwargs.pop("json_format", True)
        self.to_file = kwargs.pop("to_file", None)
        self.env = os.environ.get("DUO_WORKFLOW_SERVICE_ENVIRONMENT", "development")


def setup_logging(json_format: bool, to_file: Optional[str]):
    """
    Set up structured logging.

    Args:s
        json_format: Whether to use JSON formatting (default: True)
        to_file: log file name (default: None)
    """
    logging_config = LoggingConfig(json_format=json_format, to_file=to_file)

    # Configure basic logging
    logging.basicConfig(format="%(message)s", level=logging_config.level)

    def add_correlation_id(_, __, event_dict):
        """Add correlation ID to structured log events."""
        event_dict["correlation_id"] = correlation_id.get()
        return event_dict

    def add_gitlab_global_user_id(_, __, event_dict):
        """Add gitlab_global_user_id to structured log events."""
        event_dict["gitlab_global_user_id"] = gitlab_global_user_id.get()
        return event_dict

    def add_workflow_id(_, __, event_dict):
        """Add workflow ID to structured log events."""
        event_dict["workflow_id"] = _workflow_id.get()
        return event_dict

    # Setup shared processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_correlation_id,
        add_gitlab_global_user_id,
        add_workflow_id,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processor: JSONRenderer | ConsoleRenderer
    # Configure formatter based on environment
    if logging_config.json_format:
        shared_processors.append(structlog.processors.format_exc_info)
        processor = structlog.processors.JSONRenderer()
    else:
        processor = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=processor,
    )

    # Apply formatter to handler
    handler: logging.Handler
    if logging_config.to_file:
        try:
            file = Path(logging_config.to_file).resolve()
            handler = logging.FileHandler(filename=str(file), mode="a")
        except IOError:
            handler = (
                logging.StreamHandler()
            )  # switch to logs stream when logging to file fails
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    # Remove existing handlers and add our new one
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging_config.level)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            structlog.stdlib.render_to_log_kwargs,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
