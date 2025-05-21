"""Connection pool singleton for HTTP clients."""

from types import TracebackType
from typing import Optional, Type

import aiohttp
import structlog

log = structlog.stdlib.get_logger(__name__)


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
            connector = aiohttp.TCPConnector(limit=pool_size)
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
