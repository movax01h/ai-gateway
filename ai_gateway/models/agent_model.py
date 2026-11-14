from typing import Any, AsyncIterator, Optional, override

from langchain_core.messages import AIMessage

from ai_gateway.model_selection import LLMDefinition
from ai_gateway.models.base import ModelMetadata, TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.prompts.base import Prompt
from ai_gateway.safety_attributes import SafetyAttributes

__all__ = [
    "AgentModel",
]

AGENT = "agent"

# The frozen generation prompts prefill `<new_code>` and stop at `</new_code>`. Claude 4.6+
# rejects prefill, so the turn is re-roled (`remove_trailing_assistant_message`) and the model
# writes the opening tag itself; the stop sequence still removes the closing one.
_NEW_CODE_OPEN = "<new_code>"


class AgentModel(TextGenModelBase):
    def __init__(
        self,
        prompt: Prompt,
        llm_definition: Optional[LLMDefinition] = None,
    ):
        self.prompt = prompt
        self._llm_definition = llm_definition
        self._metadata = ModelMetadata(
            name=prompt.name,
            engine=AGENT,
            identifier=llm_definition.gitlab_identifier if llm_definition else None,
        )

    @property
    @override
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    @override
    def input_token_limit(self) -> int:
        if self._llm_definition:
            return self._llm_definition.max_context_tokens

        # Most modern models support at least 32k tokens
        return 32_768

    @override
    async def generate(  # type: ignore[override]
        self,
        params: dict[str, Any],
        stream: bool = False,
    ) -> (
        TextGenModelOutput | list[TextGenModelOutput] | AsyncIterator[TextGenModelChunk]
    ):
        if stream:
            return self._handle_stream(params)

        response = await self.prompt.ainvoke(params)

        # `.text` skips the thinking blocks Claude 5.x emits without being asked.
        text = response.text.removeprefix(_NEW_CODE_OPEN)
        response_content = self._format_response_content(text) or ""

        score = 10**5
        if response.response_metadata and "score" in response.response_metadata:
            metadata_score = response.response_metadata["score"]
            if metadata_score is not None:
                score = metadata_score

        metadata = None
        if isinstance(response, AIMessage) and isinstance(
            response.usage_metadata, dict
        ):
            output_tokens = int(response.usage_metadata.get("output_tokens") or 0)
            max_tokens = getattr(self.prompt.model, "max_tokens", None)
            metadata = TokensConsumptionMetadata(
                input_tokens=int(response.usage_metadata.get("input_tokens") or 0),
                output_tokens=output_tokens,
                max_output_tokens_used=isinstance(max_tokens, int)
                and output_tokens >= max_tokens,
            )

        return TextGenModelOutput(
            text=response_content,
            # Give a high value, the model doesn't return scores.
            score=score,
            safety_attributes=SafetyAttributes(),
            metadata=metadata,
        )

    async def _handle_stream(
        self,
        params: dict,
    ) -> AsyncIterator[TextGenModelChunk]:
        # Hold back the head of the stream until the opening tag is ruled in or out.
        head: Optional[str] = ""
        async for chunk in self.prompt.astream(params):
            text: str = chunk.text
            if head is not None:
                head += text
                if len(head) < len(_NEW_CODE_OPEN) and _NEW_CODE_OPEN.startswith(head):
                    continue
                text, head = head.removeprefix(_NEW_CODE_OPEN), None
            if text:
                yield TextGenModelChunk(text=self._format_response_content(text) or "")

        if head:
            yield TextGenModelChunk(text=self._format_response_content(head) or "")

    def _format_response_content(self, text: Optional[str]) -> Optional[str]:
        if text and self.prompt.model_name == "mixtral":
            return text.replace("\\_", "_")

        return text
