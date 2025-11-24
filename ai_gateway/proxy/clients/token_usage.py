from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Generic token usage model for all proxy clients.

    This model normalizes token usage across different providers:
    - OpenAI: uses prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details.cached_tokens
    - Anthropic: uses input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens
    - Vertex AI: uses metadata.tokenMetadata with inputTokenCount, outputTokenCount
    """

    input_tokens: int = Field(
        default=0,
        description="Number of tokens in the input/prompt",
    )
    output_tokens: int = Field(
        default=0,
        description="Number of tokens in the output/completion",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total number of tokens (input + output). Computed if not provided.",
    )
    cache_creation_input_tokens: int = Field(
        default=0,
        description="Number of input tokens used to create the cache (Anthropic)",
    )
    cache_read_input_tokens: int = Field(
        default=0,
        description="Number of input tokens read from the cache (Anthropic/OpenAI)",
    )
    reasoning_tokens: Optional[int] = Field(
        default=0,
        description="Number of tokens used for reasoning (OpenAI reasoning models)",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if data.get("total_tokens") is None:
            data["total_tokens"] = data.get("input_tokens", 0) + data.get(
                "output_tokens", 0
            )
        return data

    def to_billing_metadata(self) -> Dict[str, Any]:
        """Convert TokenUsage to billing metadata format.

        Returns a dictionary with token counts in the billing event format, using the original field names
        (prompt_tokens, completion_tokens). Only includes cache token fields if they're non-zero.
        """
        metadata = {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
            or (self.input_tokens + self.output_tokens),
        }

        # Only include cache tokens if they're non-zero
        if self.cache_creation_input_tokens > 0:
            metadata["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.cache_read_input_tokens > 0:
            metadata["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.reasoning_tokens:
            metadata["reasoning_tokens"] = self.reasoning_tokens

        return metadata
