from typing import Any, AsyncIterator, Optional, Union

import structlog
from dependency_injector.providers import Factory
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.code_suggestions.base import (
    CodeSuggestionsChunk,
    CodeSuggestionsOutput,
    LanguageId,
    increment_lang_counter,
    resolve_lang_id,
    resolve_lang_name,
)
from ai_gateway.code_suggestions.processing import Prompt, TokenStrategyBase
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import PromptBuilderPrefixBased
from ai_gateway.code_suggestions.processing.typing import MetadataExtraInfo
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import ChatModelBase, Message, ModelAPICallError, ModelAPIError
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.amazon_q import AmazonQModel
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from lib.billing_events.client import BillingEventsClient

__all__ = ["CodeCompletions"]

log = structlog.stdlib.get_logger("codesuggestions")


class CodeCompletions:
    SUFFIX_RESERVED_PERCENT = 0.07

    def __init__(
        self,
        model: TextGenModelBase,
        tokenization_strategy: TokenStrategyBase,
        post_processor: Optional[Factory[PostProcessor]] = None,
        billing_event_client: Optional[BillingEventsClient] = None,
    ):
        self.model = model

        self.instrumentator = TextGenModelInstrumentator(
            model.metadata.engine, model.metadata.name
        )

        self.post_processor = post_processor
        self.billing_event_client = billing_event_client
        self.tokenization_strategy = tokenization_strategy

        self.prompt_builder = PromptBuilderPrefixBased(
            model.input_token_limit, tokenization_strategy
        )

    def _track_billing_event(
        self, user: Optional[CloudConnectorUser], output_tokens: int
    ) -> None:
        """Track billing event for code completions."""
        if self.billing_event_client and user:
            try:
                billing_metadata = {
                    "execution_environment": "code_completions",
                    "llm_operations": [
                        {
                            "model_id": self.model.metadata.identifier,
                            "completion_tokens": output_tokens,
                        }
                    ],
                }

                self.billing_event_client.track_billing_event(
                    user=user,
                    event_type="code_completions",
                    category=self.__class__.__name__,
                    unit_of_measure="request",
                    quantity=1,
                    metadata=billing_metadata,
                )
            except Exception as e:
                log.error(
                    "Failed to track billing event for code suggestions",
                    error=str(e),
                    output_tokens=output_tokens,
                )

    def _get_prompt(
        self,
        prefix: str,
        suffix: str,
        raw_prompt: Optional[str | list[Message]] = None,
        code_context: Optional[list] = None,
        context_max_percent: Optional[float] = None,
    ) -> Prompt:
        if raw_prompt:
            return self.prompt_builder.wrap(raw_prompt)

        self.prompt_builder.add_content(
            prefix,
            suffix=suffix,
            suffix_reserved_percent=self.SUFFIX_RESERVED_PERCENT,
            context_max_percent=context_max_percent,
            code_context=code_context,
        )

        prompt = self.prompt_builder.build()

        return prompt

    async def execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: Optional[str] = None,
        raw_prompt: Optional[str | list[Message]] = None,
        code_context: Optional[list] = None,
        stream: bool = False,
        user: Optional[CloudConnectorUser] = None,
        **kwargs: Any,
    ) -> Union[CodeSuggestionsOutput, AsyncIterator[CodeSuggestionsChunk]]:
        lang_id = resolve_lang_id(file_name, editor_lang)
        increment_lang_counter(file_name, lang_id, editor_lang)

        context_max_percent = kwargs.pop(
            "context_max_percent", 1.0
        )  # default is full context window
        prompt = self._get_prompt(
            prefix,
            suffix,
            raw_prompt=raw_prompt,
            code_context=code_context,
            context_max_percent=context_max_percent,
        )

        with self.instrumentator.watch(prompt) as watch_container:
            try:
                watch_container.register_lang(lang_id, editor_lang)

                if isinstance(self.model, AgentModel):
                    if lang := (editor_lang or resolve_lang_name(file_name)):
                        params = {
                            "prefix": prompt.prefix,
                            "suffix": prompt.suffix,
                            "file_name": file_name,
                            "language": lang.lower(),
                        }

                        res = await self.model.generate(params, stream)
                    else:
                        res = None
                elif isinstance(self.model, ChatModelBase):
                    res = await self.model.generate(
                        prompt.prefix, stream=stream, **kwargs
                    )
                elif isinstance(self.model, AmazonQModel):
                    if lang := (editor_lang or resolve_lang_name(file_name)):
                        res = await self.model.generate(
                            prompt.prefix,
                            prompt.suffix,
                            file_name,
                            lang.lower(),
                            stream,
                            **kwargs,
                        )
                    else:
                        res = None
                else:
                    res = await self.model.generate(
                        prompt.prefix, prompt.suffix, stream, **kwargs
                    )

                if res:
                    if isinstance(res, AsyncIterator):
                        return self._handle_stream(res, user)

                    return await self._handle_sync(
                        prompt, res, lang_id, watch_container, user
                    )
            except ModelAPICallError as ex:
                watch_container.register_model_exception(str(ex), ex.code)
                raise
            except ModelAPIError as ex:
                # TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/294
                watch_container.register_model_exception(str(ex), -1)
                raise

        return CodeSuggestionsOutput(
            text="",
            score=0,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                tokens_consumption_metadata=self._get_tokens_consumption_metadata(
                    prompt
                ),
            ),
        )

    async def _handle_stream(
        self,
        response: AsyncIterator[TextGenModelChunk],
        user: Optional[CloudConnectorUser] = None,
    ) -> AsyncIterator[CodeSuggestionsChunk]:
        chunks = []
        try:
            async for chunk in response:
                chunk_content = CodeSuggestionsChunk(text=chunk.text)
                chunks.append(chunk.text)
                yield chunk_content
        finally:
            # Track billing event for streaming response
            if chunks:
                estimated_tokens = sum(
                    self.tokenization_strategy.estimate_length(chunks)
                )
                self._track_billing_event(user, estimated_tokens)

    async def _handle_sync(
        self,
        prompt: Prompt,
        response: TextGenModelOutput,
        lang_id: Optional[LanguageId],
        watch_container: TextGenModelInstrumentator.WatchContainer,
        user: Optional[CloudConnectorUser] = None,
    ) -> CodeSuggestionsOutput:
        watch_container.register_model_output_length(response.text)
        watch_container.register_model_score(response.score)
        watch_container.register_safety_attributes(response.safety_attributes)

        tokens_consumption_metadata = self._get_tokens_consumption_metadata(
            prompt, response
        )

        response_text = await self._get_response_text(
            response_text=response.text,
            prompt=prompt,
            lang_id=lang_id,
            score=response.score,
            max_output_tokens_used=tokens_consumption_metadata.max_output_tokens_used,
        )

        watch_container.register_model_post_processed_output_length(response_text)

        self._track_billing_event(user, tokens_consumption_metadata.output_tokens)

        return CodeSuggestionsOutput(
            text=response_text,
            score=response.score,
            model=self.model.metadata,
            lang_id=lang_id,
            metadata=CodeSuggestionsOutput.Metadata(
                tokens_consumption_metadata=tokens_consumption_metadata,
            ),
        )

    async def _get_response_text(
        self,
        response_text: str,
        prompt: Prompt,
        lang_id: LanguageId,
        score: float,
        max_output_tokens_used: bool,
    ):
        if self.post_processor:
            return await self.post_processor(
                prompt.prefix, suffix=prompt.suffix, lang_id=lang_id
            ).process(
                response_text,
                score=score,
                max_output_tokens_used=max_output_tokens_used,
                model_name=self.model.metadata.name,
            )

        return response_text

    def _get_tokens_consumption_metadata(
        self, prompt: Prompt, response: Optional[TextGenModelOutput] = None
    ) -> TokensConsumptionMetadata:
        input_tokens = sum(
            component.length_tokens for component in prompt.metadata.components.values()
        )

        max_output_tokens_used = False

        if response:
            output_tokens = (
                response.metadata.output_tokens
                if response.metadata and hasattr(response.metadata, "output_tokens")
                else 0
            )

            if response.metadata and hasattr(
                response.metadata, "max_output_tokens_used"
            ):
                max_output_tokens_used = response.metadata.max_output_tokens_used
        else:
            output_tokens = 0

        context_tokens_sent = 0
        context_tokens_used = 0

        if prompt.metadata.code_context and isinstance(
            prompt.metadata.code_context, MetadataExtraInfo
        ):
            context_tokens_sent = prompt.metadata.code_context.pre.length_tokens
            context_tokens_used = prompt.metadata.code_context.post.length_tokens

        return TokensConsumptionMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_output_tokens_used=max_output_tokens_used,
            context_tokens_sent=context_tokens_sent,
            context_tokens_used=context_tokens_used,
        )
