from typing import Any, AsyncIterator, override

from ai_gateway.code_suggestions.processing.ops import strip_whitespaces
from ai_gateway.code_suggestions.processing.post.base import PostProcessorBase
from ai_gateway.code_suggestions.processing.post.ops import (
    clean_model_reflection,
    prepend_new_line,
    strip_code_block_markdown,
)

__all__ = ["PostProcessor", "PostProcessorAnthropic", "StreamingPostProcessor"]


class PostProcessor(PostProcessorBase):
    def __init__(self, code_context: str):
        self.code_context = code_context

    @override
    async def process(self, completion: str, **kwargs: Any) -> str:
        completion = strip_code_block_markdown(completion)
        completion = prepend_new_line(self.code_context, completion)

        # Note: `clean_model_reflection` joins code context and completion
        # we need to call the function right after prepending a new line
        completion = await clean_model_reflection(self.code_context, completion)
        completion = await strip_whitespaces(completion)

        return completion


class PostProcessorAnthropic(PostProcessor):
    @override
    async def process(self, completion: str, **kwargs: Any) -> str:
        completion = await strip_whitespaces(completion)

        return completion


class StreamingPostProcessor:
    """Post-processor for streaming code suggestions that strips markdown code blocks.

    Processes streaming text chunks line by line, accumulating partial chunks until complete lines are formed, then
    applies markdown fence stripping to each line. This ensures that LLM-generated markdown fences (```) are removed
    from the streaming output before reaching the editor.
    """

    async def process(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        buffer: str = ""
        try:
            async for text in chunks:
                buffer += text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield strip_code_block_markdown(line + "\n")
            if buffer:
                yield strip_code_block_markdown(buffer)
        except Exception:
            if buffer:
                yield strip_code_block_markdown(buffer)
            raise
