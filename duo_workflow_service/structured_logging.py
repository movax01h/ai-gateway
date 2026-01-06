# pylint: disable=direct-environment-variable-reference

import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DUO_WORKFLOW_LOGGING__")

    level: str = "INFO"
    json_format: Optional[bool] = None
    to_file: Optional[str] = None
    environment: str = Field(
        default="development", alias="DUO_WORKFLOW_SERVICE_ENVIRONMENT"
    )

    @field_validator("level")
    @classmethod
    def level_to_upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def set_json_format_default(self) -> "LoggingConfig":
        """Set json_format based on environment if not explicitly provided."""
        if self.json_format is None:
            self.json_format = self.environment != "development"
        return self


def setup_logging():
    logging_config = LoggingConfig()

    # Initialize AI Gateway logging globals so can_log_request_data() works correctly
    # when DWS uses AI Gateway's model factories through the prompt registry
    _setup_ai_gateway_logging_globals()

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


def _setup_ai_gateway_logging_globals():
    """Initialize only the AI Gateway logging globals needed by model factories in DWS.

    This sets the global constants that can_log_request_data() depends on, without running AI Gateway's full logging
    setup which would interfere with DWS logging.
    """
    # pylint: disable=import-outside-toplevel
    import os

    import ai_gateway.structured_logging as aigw_logging

    # Read AI Gateway environment variables directly
    enable_request_logging = (
        os.getenv("AIGW_LOGGING__ENABLE_REQUEST_LOGGING", "false").lower() == "true"
    )
    custom_models_enabled = (
        os.getenv("AIGW_CUSTOM_MODELS__ENABLED", "false").lower() == "true"
    )
    enable_litellm_logging = (
        os.getenv("AIGW_LOGGING__ENABLE_LITELLM_LOGGING", "false").lower() == "true"
    )

    # Set only the global constants needed by can_log_request_data()
    aigw_logging.ENABLE_REQUEST_LOGGING = enable_request_logging
    aigw_logging.CUSTOM_MODELS_ENABLED = custom_models_enabled

    # Optionally enable LiteLLM debugging if configured
    if enable_litellm_logging:
        import litellm

        litellm._turn_on_debug()
