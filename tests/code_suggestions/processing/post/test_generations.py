from typing import AsyncIterator

import pytest

from ai_gateway.code_suggestions.processing.post.generations import (
    StreamingPostProcessor,
)


async def _stream(chunks: list[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


async def _collect(processor: StreamingPostProcessor, chunks: list[str]) -> list[str]:
    out: list[str] = []
    async for text in processor.process(_stream(chunks)):
        out.append(text)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_chunks", "expected_concat"),
    [
        # No fences — passes through unchanged
        (["hello ", "world!"], "hello world!"),
        # Leading fence in a single chunk (with language tag) — fully stripped
        (
            ["```cpp\n", "int main() { return 0; }\n", "```"],
            "int main() { return 0; }\n",
        ),
        # Both fences inline in one chunk
        (
            ["```\nint x = 1;\n```"],
            "int x = 1;\n",
        ),
        # No language identifier, fence in own chunk
        (
            ["```\n", "code\n", "```"],
            "code\n",
        ),
        # Trailing fence only (no leading)
        (
            ["int x = 1;\n", "```"],
            "int x = 1;\n",
        ),
        # Stream of just fences — emits nothing
        (
            ["```\n", "```"],
            "",
        ),
    ],
)
async def test_strips_fences(input_chunks: list[str], expected_concat: str):
    out = await _collect(StreamingPostProcessor(), input_chunks)
    assert "".join(out) == expected_concat


@pytest.mark.asyncio
async def test_handles_empty_stream():
    out = await _collect(StreamingPostProcessor(), [])
    assert out == []


@pytest.mark.asyncio
async def test_strips_split_leading_fence():
    """Line-buffering accumulates ``` + cpp + \\n into a complete fence line and strips it as a unit, even when the
    model emits each token separately."""
    out = await _collect(
        StreamingPostProcessor(),
        ["```", "cpp", "\n", "void f() {}\n", "```"],
    )
    assert "```" not in "".join(out)
    assert "cpp" not in "".join(out)
    assert "void f() {}" in "".join(out)


@pytest.mark.asyncio
async def test_flushes_buffer_on_upstream_exception():
    async def _failing_stream() -> AsyncIterator[str]:
        yield "int x = "
        raise RuntimeError("upstream failed")

    out: list[str] = []
    with pytest.raises(RuntimeError, match="upstream failed"):
        async for text in StreamingPostProcessor().process(_failing_stream()):
            out.append(text)

    assert "".join(out) == "int x = "
