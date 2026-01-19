import hashlib
import inspect
import json
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional, Protocol, Sequence, Type, Union

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel

__all__ = [
    "BindToolsCacheProtocol",
    "BindToolsCache",
    "NoOpBindToolsCache",
    "compute_tool_signature",
]

log = structlog.get_logger(__name__)

# Prometheus Metrics
BIND_TOOLS_CACHE_HITS = Counter(
    "bind_tools_cache_hits_total",
    "Number of bind_tools cache hits",
    ["model_provider"],
)

BIND_TOOLS_CACHE_MISSES = Counter(
    "bind_tools_cache_misses_total",
    "Number of bind_tools cache misses",
    ["model_provider"],
)

BIND_TOOLS_DURATION = Histogram(
    "bind_tools_duration_seconds",
    "Time spent in bind_tools operation (including cache lookup)",
    ["cache_status", "model_provider"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

BIND_TOOLS_CACHE_SIZE = Gauge(
    "bind_tools_cache_size",
    "Current number of entries in bind_tools cache",
)

BIND_TOOLS_CACHE_EVICTIONS = Counter(
    "bind_tools_cache_evictions_total",
    "Number of cache evictions due to size limit",
)


def _get_tool_name(tool: Union[BaseTool, Type[BaseModel]]) -> str:
    """Get the name of a tool, handling both BaseTool instances and Type[BaseModel] classes."""
    if isinstance(tool, BaseTool):
        return tool.name
    if inspect.isclass(tool) and issubclass(tool, BaseModel):
        # For Type[BaseModel], use the class name
        return tool.__name__
    # Fallback: try to get name attribute or use class name
    return getattr(tool, "name", tool.__class__.__name__)


def _get_tool_description(tool: Union[BaseTool, Type[BaseModel]]) -> str:
    """Get the description of a tool, handling both BaseTool instances and Type[BaseModel] classes."""
    if isinstance(tool, BaseTool):
        return tool.description
    if inspect.isclass(tool) and issubclass(tool, BaseModel):
        # For Type[BaseModel], use the docstring
        return tool.__doc__ or ""
    # Fallback: try to get description attribute or use empty string
    return getattr(tool, "description", "")


@lru_cache(maxsize=256)
def _compute_hash(signature_string: str) -> str:
    """Compute SHA256 hash of signature string.

    This function is cached to avoid repeated hashing of identical signature strings.

    Args:
        signature_string: The concatenated tool signature string

    Returns:
        SHA256 hex digest of the signature string
    """
    return hashlib.sha256(signature_string.encode("utf-8")).hexdigest()


def compute_tool_signature(tools: Sequence[Union[BaseTool, Type[BaseModel]]]) -> str:
    """Compute a deterministic signature for a list of tools.

    The signature computation involves JSON serialization and SHA256 hashing.
    The hash computation is cached via @lru_cache to speed up repeated calls
    with identical tool configurations.

    Args:
        tools: List of tools, which can be either BaseTool instances or Type[BaseModel] classes

    Returns:
        SHA256 hash of the tool signatures
    """
    sorted_tools = sorted(tools, key=_get_tool_name)

    signature_parts = []
    for tool in sorted_tools:
        tool_name = _get_tool_name(tool)
        tool_desc = _get_tool_description(tool)

        signature_parts.append(f"name:{tool_name}")
        signature_parts.append(f"desc:{tool_desc}")

        # Handle schema for both BaseTool and Type[BaseModel]
        if (
            isinstance(tool, BaseTool)
            and hasattr(tool, "args_schema")
            and tool.args_schema
        ):
            # Handle both Pydantic models and plain dicts
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            else:
                schema = tool.args_schema.model_json_schema()
            schema_str = json.dumps(schema, sort_keys=True)
            signature_parts.append(f"schema:{schema_str}")
        elif inspect.isclass(tool) and issubclass(tool, BaseModel):
            # For Type[BaseModel], the tool itself is the schema
            schema = tool.model_json_schema()
            schema_str = json.dumps(schema, sort_keys=True)
            signature_parts.append(f"schema:{schema_str}")

    # Compute SHA256 hash (cached to avoid repeated hashing)
    signature_string = "|".join(signature_parts)
    return _compute_hash(signature_string)


class BindToolsCacheProtocol(Protocol):
    """Protocol defining the interface for bind_tools cache implementations.

    This allows different implementations (caching vs pass-through) to be used interchangeably via dependency injection.
    """

    def get_or_bind(
        self,
        model: BaseChatModel,
        model_id: str,
        tools: Sequence[Union[BaseTool, Type[BaseModel]]],
        tool_choice: Optional[str],
        model_provider: str = "unknown",
    ) -> Runnable[Any, BaseMessage]:
        """Bind tools to model, potentially using cache.

        Args:
            model: The language model to bind tools to
            model_id: Unique identifier for the model
            tools: Sequence of tools to bind
            tool_choice: Optional tool choice parameter
            model_provider: Model provider name for metrics

        Returns:
            Model with tools bound
        """

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """


class NoOpBindToolsCache:
    """Pass-through implementation that doesn't cache, only records metrics.

    This implementation provides instrumentation without caching overhead. Used when caching is disabled via
    configuration.
    """

    def get_or_bind(
        self,
        model: BaseChatModel,
        model_id: str,
        tools: Sequence[Union[BaseTool, Type[BaseModel]]],
        tool_choice: Optional[str],
        model_provider: str = "unknown",
    ) -> Runnable[Any, BaseMessage]:
        """Bind tools to model without caching, with metrics.

        Args:
            model: The language model to bind tools to
            model_id: Unique identifier for the model (unused)
            tools: Sequence of tools to bind
            tool_choice: Optional tool choice parameter
            model_provider: Model provider name for metrics

        Returns:
            Model with tools bound
        """
        start_time = time.perf_counter()

        bound_model = model.bind_tools(tools, tool_choice=tool_choice)

        duration = time.perf_counter() - start_time
        BIND_TOOLS_DURATION.labels(
            cache_status="disabled", model_provider=model_provider
        ).observe(duration)

        log.info(
            "bind_tools_no_cache",
            model_id=model_id,
            tool_count=len(tools),
            tool_choice=tool_choice,
            duration_ms=duration * 1000,
        )

        return bound_model

    def get_stats(self) -> dict:
        """Get statistics (always empty for no-op implementation).

        Returns:
            Dictionary indicating caching is disabled
        """
        return {
            "enabled": False,
            "size": 0,
            "max_size": 0,
        }


class BindToolsCache:
    """Thread-safe LRU cache for bind_tools results.

    Thread Safety:
        Uses RLock for thread-safe operations. Safe to use from multiple
        threads or asyncio tasks.

    LRU Eviction:
        When cache reaches max_size, least recently used entries are evicted.
        Access updates the "recently used" status.

    Args:
        max_size: Maximum number of cache entries (default: 128)
    """

    def __init__(self, max_size: int = 128):
        self._max_size = max_size
        self._cache: OrderedDict[
            tuple[str, str, Optional[str]], Runnable[Any, BaseMessage]
        ] = OrderedDict()
        self._lock = threading.RLock()

        log.info(
            "bind_tools_cache_initialized",
            max_size=self._max_size,
        )

    def get_or_bind(
        self,
        model: BaseChatModel,
        model_id: str,
        tools: Sequence[Union[BaseTool, Type[BaseModel]]],
        tool_choice: Optional[str],
        model_provider: str = "unknown",
    ) -> Runnable[Any, BaseMessage]:
        """Get cached bound model or bind tools and cache the result.

        Side Effects:
            - Updates cache on miss
            - Records Prometheus metrics
            - Logs cache hits/misses
            - May evict LRU entries if cache is full
        """
        start_time = time.perf_counter()
        tool_signature = compute_tool_signature(tools)
        cache_key = (model_id, tool_signature, tool_choice)

        with self._lock:
            if cache_key in self._cache:
                # Cache hit - move to end (most recently used)
                self._cache.move_to_end(cache_key)
                bound_model = self._cache[cache_key]

                # Record metrics
                BIND_TOOLS_CACHE_HITS.labels(model_provider=model_provider).inc()
                duration = time.perf_counter() - start_time
                BIND_TOOLS_DURATION.labels(
                    cache_status="hit", model_provider=model_provider
                ).observe(duration)

                log.debug(
                    "bind_tools_cache_hit",
                    model_id=model_id,
                    tool_count=len(tools),
                    tool_choice=tool_choice,
                    cache_size=len(self._cache),
                    duration_ms=duration * 1000,
                )

                return bound_model

            # Cache miss - bind tools
            log.info(
                "bind_tools_cache_miss",
                model_id=model_id,
                tool_count=len(tools),
                tool_choice=tool_choice,
                cache_size=len(self._cache),
            )

            # Perform expensive bind_tools operation
            bound_model = model.bind_tools(tools, tool_choice=tool_choice)

            self._cache[cache_key] = bound_model

            # Evict LRU entry if cache is full
            if len(self._cache) > self._max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                BIND_TOOLS_CACHE_EVICTIONS.inc()
                log.debug(
                    "bind_tools_cache_eviction",
                    evicted_model_id=evicted_key[0],
                    cache_size=len(self._cache),
                )

            BIND_TOOLS_CACHE_MISSES.labels(model_provider=model_provider).inc()
            BIND_TOOLS_CACHE_SIZE.set(len(self._cache))
            duration = time.perf_counter() - start_time
            BIND_TOOLS_DURATION.labels(
                cache_status="miss", model_provider=model_provider
            ).observe(duration)

            return bound_model

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            BIND_TOOLS_CACHE_SIZE.set(0)
            log.info("bind_tools_cache_cleared")

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        with self._lock:
            return {
                "enabled": True,
                "size": len(self._cache),
                "max_size": self._max_size,
            }
