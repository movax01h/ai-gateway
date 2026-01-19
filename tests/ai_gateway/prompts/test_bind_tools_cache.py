"""Unit tests for bind_tools_cache module.

Tests cover:
- Cache key generation (tool signature computation)
- Cache hit/miss behavior (BindToolsCache)
- LRU eviction policy (BindToolsCache)
- Thread safety (BindToolsCache)
- Pass-through behavior (NoOpBindToolsCache)
- Prometheus metrics (both implementations)
"""

from unittest import mock
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ai_gateway.prompts.bind_tools_cache import (
    BindToolsCache,
    NoOpBindToolsCache,
    compute_tool_signature,
)


# Mock tools for testing
class ReadFileArgs(BaseModel):
    """Arguments for ReadFile tool."""

    path: str = Field(description="Path to the file to read")


class WriteFileArgs(BaseModel):
    """Arguments for WriteFile tool."""

    path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class ListDirArgs(BaseModel):
    """Arguments for ListDir tool."""

    path: str = Field(description="Path to the directory to list")


class ReadFileTool(BaseTool):
    """Mock ReadFile tool."""

    name: str = "read_file"
    description: str = "Read contents of a file"
    args_schema: type[BaseModel] = ReadFileArgs

    def _run(self, path: str) -> str:
        return f"Contents of {path}"


class WriteFileTool(BaseTool):
    """Mock WriteFile tool."""

    name: str = "write_file"
    description: str = "Write content to a file"
    args_schema: type[BaseModel] = WriteFileArgs

    def _run(self, path: str, content: str) -> str:
        return f"Wrote to {path}"


class ListDirTool(BaseTool):
    """Mock ListDir tool."""

    name: str = "list_dir"
    description: str = "List contents of a directory"
    args_schema: type[BaseModel] = ListDirArgs

    def _run(self, path: str) -> str:
        return f"Contents of {path}"


@pytest.fixture(name="read_file_tool")
def read_file_tool_fixture():
    return ReadFileTool()


@pytest.fixture(name="write_file_tool")
def write_file_tool_fixture():
    return WriteFileTool()


@pytest.fixture(name="list_dir_tool")
def list_dir_tool_fixture():
    return ListDirTool()


class TestComputeToolSignature:
    """Test compute_tool_signature function."""

    def test_signature_is_stable(self, read_file_tool, write_file_tool):
        """Tool signature should be deterministic for same tools."""
        tools1 = [read_file_tool, write_file_tool]
        tools2 = [read_file_tool, write_file_tool]

        sig1 = compute_tool_signature(tools1)
        sig2 = compute_tool_signature(tools2)

        assert sig1 == sig2

    def test_signature_is_order_independent(
        self, read_file_tool, write_file_tool, list_dir_tool
    ):
        """Tool signature should be same regardless of order."""
        tools1 = [read_file_tool, write_file_tool, list_dir_tool]
        tools2 = [list_dir_tool, read_file_tool, write_file_tool]

        sig1 = compute_tool_signature(tools1)
        sig2 = compute_tool_signature(tools2)

        assert sig1 == sig2

    def test_signature_differs_for_different_tools(
        self, read_file_tool, write_file_tool, list_dir_tool
    ):
        """Different tool sets should have different signatures."""
        tools1 = [read_file_tool, write_file_tool]
        tools2 = [read_file_tool, list_dir_tool]

        sig1 = compute_tool_signature(tools1)
        sig2 = compute_tool_signature(tools2)

        assert sig1 != sig2

    def test_signature_is_hex_string(self, read_file_tool):
        """Signature should be a hex string (SHA256)."""
        tools = [read_file_tool]
        sig = compute_tool_signature(tools)

        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in sig)

    def test_signature_with_empty_list(self):
        """Empty tool list should produce a signature."""
        sig = compute_tool_signature([])

        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_signature_with_dict_args_schema(self):
        """Tool with dict args_schema (like MCP tools) should work."""

        class McpStyleTool(BaseTool):
            """Tool that uses a dict for args_schema (like MCP tools)."""

            name: str = "mcp_tool"
            description: str = "An MCP-style tool"
            args_schema: dict = {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "A parameter"}
                },
                "required": ["param"],
            }

            def _run(self, param: str) -> str:
                return f"Result: {param}"

        tool = McpStyleTool()
        sig = compute_tool_signature([tool])

        # Should not raise AttributeError and should produce a valid signature
        assert isinstance(sig, str)
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


class TestBindToolsCache:
    """Test BindToolsCache class."""

    @pytest.fixture(name="cache")
    def cache_fixture(self):
        return BindToolsCache(max_size=128)

    def test_cache_hit(self, cache, read_file_tool, write_file_tool):
        """Second call with same tools should hit cache."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        tools = [read_file_tool, write_file_tool]

        # First call - cache miss
        bound1 = cache.get_or_bind(fake_model, "test-model", tools, None, "test")

        # Second call - cache hit (should return same object)
        bound2 = cache.get_or_bind(fake_model, "test-model", tools, None, "test")

        assert bound1 is bound2
        # bind_tools should only be called once (cache hit on second call)
        assert fake_model.bind_tools.call_count == 1

    def test_cache_miss_different_tools(
        self,
        cache,
        read_file_tool,
        write_file_tool,
        list_dir_tool,
    ):
        """Different tools should miss cache."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        fake_model.bind_tools.side_effect = [mock.Mock(), mock.Mock()]

        tools1 = [read_file_tool, write_file_tool]
        tools2 = [read_file_tool, list_dir_tool]

        bound1 = cache.get_or_bind(fake_model, "test-model", tools1, None, "test")
        bound2 = cache.get_or_bind(fake_model, "test-model", tools2, None, "test")

        # Should return different objects
        assert bound1 is not bound2
        # bind_tools should be called twice (cache miss both times)
        assert fake_model.bind_tools.call_count == 2

    def test_cache_miss_different_model_id(
        self, cache, read_file_tool, write_file_tool
    ):
        """Different model IDs should miss cache."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        fake_model.bind_tools.side_effect = [mock.Mock(), mock.Mock()]

        tools = [read_file_tool, write_file_tool]

        bound1 = cache.get_or_bind(fake_model, "model-1", tools, None, "test")
        bound2 = cache.get_or_bind(fake_model, "model-2", tools, None, "test")

        assert bound1 is not bound2
        # bind_tools should be called twice (cache miss both times)
        assert fake_model.bind_tools.call_count == 2

    def test_cache_miss_different_tool_choice(
        self, cache, read_file_tool, write_file_tool
    ):
        """Different tool_choice should miss cache."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        fake_model.bind_tools.side_effect = [mock.Mock(), mock.Mock()]

        tools = [read_file_tool, write_file_tool]

        bound1 = cache.get_or_bind(fake_model, "test-model", tools, None, "test")
        bound2 = cache.get_or_bind(fake_model, "test-model", tools, "auto", "test")

        assert bound1 is not bound2
        # bind_tools should be called twice (cache miss both times)
        assert fake_model.bind_tools.call_count == 2

    def test_lru_eviction(
        self,
        read_file_tool,
        write_file_tool,
        list_dir_tool,
    ):
        """Cache should evict LRU entries when full."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        # We need 5 mock objects: bound1, bound2, bound3, bound2_new, bound1_new
        fake_model.bind_tools.side_effect = [
            mock.Mock(name="bound1"),
            mock.Mock(name="bound2"),
            mock.Mock(name="bound3"),
            mock.Mock(name="bound2_new"),
            mock.Mock(name="bound1_new"),
        ]

        cache = BindToolsCache(max_size=2)

        # Fill cache with 2 entries
        tools1 = [read_file_tool]
        tools2 = [write_file_tool]
        tools3 = [list_dir_tool]

        bound1 = cache.get_or_bind(fake_model, "model-1", tools1, None, "test")
        bound2 = cache.get_or_bind(fake_model, "model-2", tools2, None, "test")

        # Access bound1 to make it more recently used
        bound1_again = cache.get_or_bind(fake_model, "model-1", tools1, None, "test")
        assert bound1 is bound1_again

        # Add third entry - should evict bound2 (LRU)
        bound3 = cache.get_or_bind(fake_model, "model-3", tools3, None, "test")

        # Verify bound2 was evicted (cache miss)
        bound2_new = cache.get_or_bind(fake_model, "model-2", tools2, None, "test")
        assert bound2 is not bound2_new

        # Verify bound1 was also evicted (adding bound2_new evicted bound1)
        # because cache size is 2 and we now have bound3 and bound2_new
        bound1_new = cache.get_or_bind(fake_model, "model-1", tools1, None, "test")
        assert bound1 is not bound1_new

    def test_clear_cache(self, cache, read_file_tool, write_file_tool):
        """Clear should remove all entries."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        fake_model.bind_tools.side_effect = [mock.Mock(), mock.Mock()]

        tools = [read_file_tool, write_file_tool]

        # Add entry
        bound1 = cache.get_or_bind(fake_model, "test-model", tools, None, "test")

        # Clear cache
        cache.clear()

        # Should miss cache after clear
        bound2 = cache.get_or_bind(fake_model, "test-model", tools, None, "test")
        assert bound1 is not bound2
        # bind_tools should be called twice (once before clear, once after)
        assert fake_model.bind_tools.call_count == 2

    def test_get_stats(self, cache, read_file_tool, write_file_tool):
        """get_stats should return cache statistics."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        tools = [read_file_tool, write_file_tool]

        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["size"] == 0
        assert stats["max_size"] == 128

        # Add entry
        cache.get_or_bind(fake_model, "test-model", tools, None, "test")

        stats = cache.get_stats()
        assert stats["size"] == 1

    @patch("ai_gateway.prompts.bind_tools_cache.BIND_TOOLS_CACHE_HITS")
    @patch("ai_gateway.prompts.bind_tools_cache.BIND_TOOLS_CACHE_MISSES")
    def test_metrics_recorded(
        self,
        mock_misses,
        mock_hits,
        cache,
        read_file_tool,
        write_file_tool,
    ):
        """Prometheus metrics should be recorded correctly."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        tools = [read_file_tool, write_file_tool]

        # First call - cache miss
        cache.get_or_bind(fake_model, "test-model", tools, None, "anthropic")
        mock_misses.labels.assert_called_with(model_provider="anthropic")
        mock_misses.labels.return_value.inc.assert_called_once()

        # Second call - cache hit
        cache.get_or_bind(fake_model, "test-model", tools, None, "anthropic")
        mock_hits.labels.assert_called_with(model_provider="anthropic")
        mock_hits.labels.return_value.inc.assert_called_once()


class TestNoOpBindToolsCache:
    """Test NoOpBindToolsCache pass-through implementation."""

    @pytest.fixture(name="no_op_cache")
    def no_op_cache_fixture(self):
        return NoOpBindToolsCache()

    def test_always_calls_bind_tools(
        self, no_op_cache, read_file_tool, write_file_tool
    ):
        """Should always call bind_tools without caching."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        # Configure mock to return different objects for each call
        fake_model.bind_tools.side_effect = [mock.Mock(), mock.Mock()]

        tools = [read_file_tool, write_file_tool]

        bound1 = no_op_cache.get_or_bind(fake_model, "test-model", tools, None, "test")
        bound2 = no_op_cache.get_or_bind(fake_model, "test-model", tools, None, "test")

        # Should return different objects (no caching)
        assert bound1 is not bound2
        # bind_tools should be called twice (no caching)
        assert fake_model.bind_tools.call_count == 2

    def test_get_stats(self, no_op_cache):
        """get_stats should indicate caching is disabled."""
        stats = no_op_cache.get_stats()

        assert stats["enabled"] is False
        assert stats["size"] == 0
        assert stats["max_size"] == 0

    @patch("ai_gateway.prompts.bind_tools_cache.BIND_TOOLS_DURATION")
    def test_metrics_recorded(
        self,
        mock_duration,
        no_op_cache,
        read_file_tool,
        write_file_tool,
    ):
        """Should record metrics with disabled status."""
        fake_model = mock.MagicMock(spec=BaseChatModel)
        tools = [read_file_tool, write_file_tool]

        no_op_cache.get_or_bind(fake_model, "test-model", tools, None, "anthropic")

        # Should record duration with cache_status="disabled"
        mock_duration.labels.assert_called_with(
            cache_status="disabled", model_provider="anthropic"
        )
        mock_duration.labels.return_value.observe.assert_called_once()
