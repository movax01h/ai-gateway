from litellm.llms.anthropic import common_utils as _anthropic_common_utils
from litellm.llms.anthropic.chat.handler import ModelResponseIterator
from litellm.types.utils import ModelResponseStream

# litellm reports an Anthropic `stop_reason: "refusal"` only as `content_filter`, the
# same value it uses for Gemini safety stops and Bedrock guardrails, and drops the
# refusal's `stop_details`. Carry both on the chunk's `provider_specific_fields`, where
# ChatLiteLLM picks them up, so a refusal keeps its category.
# litellm 1.109 does this itself through `stop_reason_provider_specific_fields`.
_original_chunk_parser = ModelResponseIterator.chunk_parser


def _chunk_parser(self: ModelResponseIterator, chunk: dict) -> ModelResponseStream:
    parsed = _original_chunk_parser(self, chunk)
    message_delta = chunk.get("delta") or {}
    if (
        chunk.get("type") == "message_delta"
        and message_delta.get("stop_reason") == "refusal"
        and parsed.choices
    ):
        delta = parsed.choices[0].delta
        delta.provider_specific_fields = {
            **(delta.provider_specific_fields or {}),
            "stop_reason": "refusal",
            "stop_details": message_delta.get("stop_details"),
        }
    return parsed


if not hasattr(_anthropic_common_utils, "stop_reason_provider_specific_fields"):
    ModelResponseIterator.chunk_parser = _chunk_parser
