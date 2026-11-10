"""The OpenAI Decisions endpoint, for prompts with `operation_type: decision`.

`OpenAIDecisions` returns its answers from a chain run, so the usage callback `Prompt` attaches never sees them. This
subclass reports each call as an LLM run that ends in an `AIMessage` with the token usage, like `EmbeddingLiteLLM`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, override

from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableBinding, RunnableConfig
from langchain_openai.decisions import (
    DecisionRequest,
    DecisionResponse,
    OpenAIDecisions,
)
from pydantic import ConfigDict

__all__ = ["OpenAIDecisionsModel", "decision_response"]

_DECISION_KEY = "decision"


class OpenAIDecisionsModel(OpenAIDecisions):
    # The model entry is shared with chat, so ignore chat-only params such as `max_tokens`.
    model_config = ConfigDict(extra="ignore")

    # Prompt passes it to every model; the Decisions API doesn't stream.
    disable_streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "openai_decisions"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @override
    def bind(self, **kwargs: Any) -> RunnableBinding:
        return RunnableBinding(bound=self, kwargs=kwargs, config={})

    async def ainvoke(  # type: ignore[override]
        self,
        input: DecisionRequest,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = self._traced_config(config)
        callbacks = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        # One LLM run, without the input, so usage callbacks and traces see the call.
        [run] = await callbacks.on_llm_start(
            {}, [""], invocation_params=self._identifying_params, name=self._llm_type
        )

        try:
            response = await self._adecide(input)
        # CancelledError subclasses BaseException (not Exception), so a cancelled
        # call or an `asyncio.wait_for` deadline skips `except Exception`. Catch
        # BaseException so the run still ends in an error, then re-raise to
        # preserve cancellation semantics.
        except BaseException as e:
            await run.on_llm_error(e)
            raise

        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        message = AIMessage(
            content="",
            response_metadata={
                "model_name": response.model,
                _DECISION_KEY: response.model_dump(),
            },
            usage_metadata=UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        await run.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        return message


def decision_response(message: BaseMessage) -> DecisionResponse:
    """Return the answers from a decision prompt's message."""
    return DecisionResponse.model_validate(message.response_metadata[_DECISION_KEY])
