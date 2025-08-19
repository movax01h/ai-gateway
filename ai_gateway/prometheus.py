import logging
import os

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    multiprocess,
    start_http_server,
)

from ai_gateway.config import Config


def start_metrics_server(config: Config):
    log = logging.getLogger("main")
    log.info(
        "Metrics HTTP server running on http://%s:%d",
        config.fastapi.metrics_host,
        config.fastapi.metrics_port,
    )

    registry = REGISTRY

    # pylint: disable=direct-environment-variable-reference
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    # pylint: enable=direct-environment-variable-reference

    start_http_server(
        addr=config.fastapi.metrics_host,
        port=config.fastapi.metrics_port,
        registry=registry,
    )
