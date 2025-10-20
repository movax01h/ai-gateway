"""Connection pool singleton for HTTP clients."""

import os
import ssl
from types import TracebackType
from typing import Optional, Type

import aiohttp
import structlog

log = structlog.stdlib.get_logger(__name__)


def _create_ssl_context_with_custom_ca() -> Optional[ssl.SSLContext]:
    """Create SSL context with custom CA if environment variables are set.

    Environment variables:
    - DUO_WORKFLOW_GITLAB_SSL_CA_FILE: Path to the CA certificate file (optional)

    Returns:
        ssl.SSLContext: SSL context with loaded certificates, or None if not configured
    """
    ca_file = os.getenv("DUO_WORKFLOW_GITLAB_SSL_CA_FILE")

    if not ca_file or not os.path.exists(ca_file):
        return None

    try:
        ssl_context = ssl.create_default_context()

        ssl_context.load_verify_locations(ca_file)
        log.info("Loaded custom CA file", ca_file=ca_file)

        log.info(
            "SSL context created with custom CA",
        )
        return ssl_context
    except Exception as e:
        log.error(
            "Failed to create SSL context with custom CA",
            error=str(e),
            ca_file=ca_file,
        )

        return None


class ConnectionPoolManager:
    """Context manager for HTTP connection pool."""

    _instance = None
    _session = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionPoolManager, cls).__new__(cls)
        return cls._instance

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the current session or raise an error if not initialized."""
        if self._session is None:
            raise RuntimeError("HTTP client connection pool is not initialized")
        return self._session

    async def __aenter__(self):
        """Initialize the connection pool when entering the context."""
        # Use default values if not set externally
        pool_size = getattr(self, "_pool_size", 100)
        session_kwargs = getattr(self, "_session_kwargs", {})

        if self._session is None:
            log.info("Initializing HTTP connection pool", pool_size=pool_size)

            # Try to load custom CA from environment variables
            ssl_context = _create_ssl_context_with_custom_ca()

            if ssl_context is None:
                log.info("Using default SSL verification for HTTP connection pool")
                connector = aiohttp.TCPConnector(limit=pool_size)
            else:
                log.info("Using custom SSL context with custom CA")
                connector = aiohttp.TCPConnector(limit=pool_size, ssl=ssl_context)

            self._session = aiohttp.ClientSession(connector=connector, **session_kwargs)
            log.info("HTTP connection pool initialized")
        return self

    def set_options(self, pool_size: int = 100, **session_kwargs):
        self._pool_size = pool_size
        self._session_kwargs = session_kwargs

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Close the connection pool when exiting the context."""
        if self._session is not None:
            log.info("Closing HTTP connection pool")
            await self._session.close()
            self._session = None
            log.info("HTTP connection pool closed")


# Global singleton instance
connection_pool = ConnectionPoolManager()
