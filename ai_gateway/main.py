import uvicorn
from structlog import get_logger

# pylint: disable=unused-import
# Importing this early patches Jinja2's default filter table to block
# reversed()-on-dict, a CPython memory-safety bug (python/cpython#154709).
# This is a critical security measure to prevent remote arbitrary code
# execution, so it must run before anything constructs a jinja2 Environment.
import ai_gateway.prompts.base  # noqa: F401
from ai_gateway.config import get_config

logger = get_logger("run_app")


def run_app():
    config = get_config()

    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    ssl_ciphers: str = "TLSv1"

    if config.fastapi.tls.enabled:
        ssl_certfile = config.fastapi.tls.cert_file
        ssl_keyfile = config.fastapi.tls.key_file
        ssl_ciphers = config.fastapi.tls.ssl_ciphers
        logger.info("Starting the server with TLS encryption.")

    # For now, trust all IPs for proxy headers until https://github.com/encode/uvicorn/pull/1611 is available.
    uvicorn.run(
        "ai_gateway.app:get_app",
        host=config.fastapi.api_host,
        port=config.fastapi.api_port,
        log_config=config.fastapi.uvicorn_logger,
        forwarded_allow_ips="*",
        reload=config.fastapi.reload,
        factory=True,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_ciphers=ssl_ciphers,
    )


if __name__ == "__main__":
    run_app()
