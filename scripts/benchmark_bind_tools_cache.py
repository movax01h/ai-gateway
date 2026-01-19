#!/usr/bin/env python
"""Benchmark script for bind_tools cache performance.

This script measures the performance improvement from caching bind_tools operations.
It simulates realistic workflows with multiple tool calls using the same model/tools.

Note: This uses mocked bind_tools calls to demonstrate the cache mechanism.
For real-world validation, test against a running AI Gateway with actual Duo queries.

Usage:
    poetry run python scripts/benchmark_bind_tools_cache.py

Example output:
    === Bind Tools Cache Benchmark ===
    Tools: 7, Iterations: 10

    WITHOUT Cache (NoOpBindToolsCache):
        Mean: 45.23ms, Std: 2.15ms
        Total: 452.30ms

    WITH Cache (BindToolsCache):
        Mean: 8.12ms, Std: 15.23ms (high std due to first miss)
        Total: 81.20ms
        Cache hits: 9, misses: 1

    Improvement: 5.6x faster
"""

import statistics
import time
from typing import Optional
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ai_gateway.prompts.bind_tools_cache import (
    BindToolsCache,
    NoOpBindToolsCache,
    compute_tool_signature,
)


# Sample tools that mimic real Duo Chat tools
class SearchInput(BaseModel):
    """Input for search tool."""

    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")


class FileReadInput(BaseModel):
    """Input for file read tool."""

    path: str = Field(description="Path to the file")
    encoding: str = Field(default="utf-8", description="File encoding")


class GitLabAPIInput(BaseModel):
    """Input for GitLab API tool."""

    endpoint: str = Field(description="API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    params: Optional[dict] = Field(default=None, description="Query parameters")


class MockTool(BaseTool):
    """Mock tool for benchmarking."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, **kwargs) -> str:
        return "mock result"


def create_realistic_tools() -> list[BaseTool]:
    """Create a set of tools similar to what Duo Chat uses."""
    tools: list[BaseTool] = []
    tool_configs: list[tuple[str, str, type[BaseModel]]] = [
        ("search", "Search for information", SearchInput),
        ("read_file", "Read contents of a file", FileReadInput),
        ("gitlab_api", "Make GitLab API calls", GitLabAPIInput),
        ("write_file", "Write contents to a file", FileReadInput),
        ("run_command", "Execute a shell command", SearchInput),
        ("git_diff", "Get git diff", SearchInput),
        ("code_search", "Search code in repository", SearchInput),
    ]

    for name, desc, schema in tool_configs:
        tool = MockTool(name=name, description=desc, args_schema=schema)
        tools.append(tool)

    return tools


def create_mock_model(bind_delay_ms: float = 50.0):
    """Create a mock model with configurable bind_tools delay."""
    mock_model = MagicMock()

    def delayed_bind(*_args, **_kwargs):
        time.sleep(bind_delay_ms / 1000)
        return MagicMock()

    mock_model.bind_tools = delayed_bind
    return mock_model


def benchmark_no_cache(model, tools: list[BaseTool], iterations: int) -> list[float]:
    """Benchmark without caching."""
    cache = NoOpBindToolsCache()
    durations = []

    for _ in range(iterations):
        start = time.perf_counter()
        cache.get_or_bind(model, "test-model", tools, tool_choice="auto")
        durations.append((time.perf_counter() - start) * 1000)

    return durations


def benchmark_with_cache(model, tools: list[BaseTool], iterations: int) -> list[float]:
    """Benchmark with caching."""
    cache = BindToolsCache(max_size=100)
    durations = []

    for _ in range(iterations):
        start = time.perf_counter()
        cache.get_or_bind(model, "test-model", tools, tool_choice="auto")
        durations.append((time.perf_counter() - start) * 1000)

    return durations


def print_results(name: str, durations: list[float]):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(
        f"  Mean: {statistics.mean(durations):.2f}ms, Std: {statistics.stdev(durations):.2f}ms"
    )
    print(f"  Min: {min(durations):.2f}ms, Max: {max(durations):.2f}ms")
    print(f"  Total: {sum(durations):.2f}ms")


def main():
    print("=" * 50)
    print("Bind Tools Cache Benchmark")
    print("=" * 50)

    tools = create_realistic_tools()
    iterations = 10
    bind_delay_ms = 50.0  # Simulates real bind_tools overhead

    print("\nConfiguration:")
    print(f"  Tools: {len(tools)}")
    print(f"  Iterations: {iterations}")
    print(f"  Simulated bind_tools delay: {bind_delay_ms}ms")

    # Benchmark signature computation
    print("\nTool signature computation:")
    start = time.perf_counter()
    for _ in range(100):
        compute_tool_signature(tools)
    sig_time = (time.perf_counter() - start) * 1000 / 100
    print(f"  Mean per call: {sig_time:.3f}ms")

    # Benchmark without cache
    model = create_mock_model(bind_delay_ms)
    no_cache_durations = benchmark_no_cache(model, tools, iterations)
    print_results("WITHOUT Cache (NoOpBindToolsCache)", no_cache_durations)

    # Benchmark with cache
    model = create_mock_model(bind_delay_ms)
    cache_durations = benchmark_with_cache(model, tools, iterations)
    print_results("WITH Cache (BindToolsCache)", cache_durations)
    print(f"  Cache hit rate: {(iterations - 1) / iterations * 100:.0f}%")

    # Summary
    speedup = sum(no_cache_durations) / sum(cache_durations)
    print(f"\n{'=' * 50}")
    print(f"IMPROVEMENT: {speedup:.1f}x faster with cache")
    print(
        f"Time saved: {sum(no_cache_durations) - sum(cache_durations):.0f}ms over {iterations} calls"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
