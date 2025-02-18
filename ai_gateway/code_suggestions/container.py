import anthropic
from dependency_injector import containers, providers
from transformers import PreTrainedTokenizerFast

from ai_gateway.code_suggestions.completions import (
    CodeCompletions,
    CodeCompletionsLegacy,
)
from ai_gateway.code_suggestions.generations import CodeGenerations
from ai_gateway.code_suggestions.processing import ModelEngineCompletions
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor as PostProcessorCompletions,
)
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessorOperation,
)
from ai_gateway.code_suggestions.processing.pre import TokenizerTokenStrategy
from ai_gateway.experimentation import experiment_registry_provider
from ai_gateway.models import KindAnthropicModel, KindVertexTextModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.base_chat import ChatModelBase
from ai_gateway.models.base_text import TextGenModelBase
from ai_gateway.models.litellm import KindLiteLlmModel
from ai_gateway.tokenizer import init_tokenizer
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator

__all__ = [
    "ContainerCodeSuggestions",
]


class ContainerCodeGenerations(containers.DeclarativeContainer):
    tokenizer = providers.Dependency(instance_of=PreTrainedTokenizerFast)
    vertex_code_bison = providers.Dependency(instance_of=TextGenModelBase)
    anthropic_claude = providers.Dependency(instance_of=TextGenModelBase)
    anthropic_claude_chat = providers.Dependency(instance_of=ChatModelBase)
    amazon_q_model = providers.Dependency(instance_of=TextGenModelBase)
    litellm_chat = providers.Dependency(instance_of=ChatModelBase)
    agent_model = providers.Dependency(instance_of=TextGenModelBase)

    snowplow_instrumentator = providers.Dependency(instance_of=SnowplowInstrumentator)

    vertex = providers.Factory(
        CodeGenerations,
        model=providers.Factory(
            vertex_code_bison, name=KindVertexTextModel.CODE_BISON_002
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    # We need to resolve the model based on model name provided in request payload
    # Hence, CodeGenerations is only partially applied here.
    anthropic_factory = providers.Factory(
        CodeGenerations,
        model=providers.Factory(
            anthropic_claude,
            stop_sequences=["</new_code>", anthropic.HUMAN_PROMPT],
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    anthropic_chat_factory = providers.Factory(
        CodeGenerations,
        model=providers.Factory(anthropic_claude_chat),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    litellm_factory = providers.Factory(
        CodeGenerations,
        model=providers.Factory(litellm_chat),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    amazon_q_factory = providers.Factory(
        CodeGenerations,
        model=providers.Factory(amazon_q_model),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    agent_factory = providers.Factory(
        CodeGenerations,
        model=providers.Factory(agent_model),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        snowplow_instrumentator=snowplow_instrumentator,
    )

    # Default use case with claude.2.1
    anthropic_default = providers.Factory(
        anthropic_factory,
        model__name=KindAnthropicModel.CLAUDE_2_1,
    )


class ContainerCodeCompletions(containers.DeclarativeContainer):
    tokenizer = providers.Dependency(instance_of=PreTrainedTokenizerFast)
    vertex_code_gecko = providers.Dependency(instance_of=TextGenModelBase)
    anthropic_claude = providers.Dependency(instance_of=TextGenModelBase)
    anthropic_claude_chat = providers.Dependency(instance_of=ChatModelBase)
    litellm = providers.Dependency(instance_of=TextGenModelBase)
    agent_model = providers.Dependency(instance_of=TextGenModelBase)
    amazon_q_model = providers.Dependency(instance_of=TextGenModelBase)
    snowplow_instrumentator = providers.Dependency(instance_of=SnowplowInstrumentator)

    config = providers.Configuration(strict=True)

    vertex_legacy = providers.Factory(
        CodeCompletionsLegacy,
        engine=providers.Factory(
            ModelEngineCompletions,
            model=providers.Factory(
                vertex_code_gecko, name=KindVertexTextModel.CODE_GECKO_002
            ),
            tokenization_strategy=providers.Factory(
                TokenizerTokenStrategy, tokenizer=tokenizer
            ),
            experiment_registry=experiment_registry_provider(),
        ),
        post_processor=providers.Factory(
            PostProcessorCompletions,
            overrides={
                PostProcessorOperation.FIX_END_BLOCK_ERRORS: PostProcessorOperation.FIX_END_BLOCK_ERRORS_LEGACY,
            },
            exclude=config.excl_post_process,
        ).provider,
        snowplow_instrumentator=snowplow_instrumentator,
    )

    anthropic = providers.Factory(
        CodeCompletions,
        model=providers.Factory(anthropic_claude_chat),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    litellm_factory = providers.Factory(
        CodeCompletions,
        model=providers.Factory(litellm),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    fireworks_qwen_factory = providers.Factory(
        CodeCompletions,
        model=providers.Factory(
            litellm,
            name=KindLiteLlmModel.QWEN_2_5,
            provider=KindModelProvider.FIREWORKS,
        ),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
        post_processor=providers.Factory(
            PostProcessorCompletions,
            exclude=config.excl_post_process,
            extras=[
                PostProcessorOperation.FILTER_SCORE,
            ],
            score_threshold=config.fireworks_qwen_score_threshold,
        ).provider,
    )

    agent_factory = providers.Factory(
        CodeCompletions,
        model=providers.Factory(agent_model),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )

    amazon_q_factory = providers.Factory(
        CodeCompletions,
        model=providers.Factory(amazon_q_model),
        tokenization_strategy=providers.Factory(
            TokenizerTokenStrategy, tokenizer=tokenizer
        ),
    )


class ContainerCodeSuggestions(containers.DeclarativeContainer):
    models = providers.DependenciesContainer()

    config = providers.Configuration(strict=True)

    tokenizer = providers.Singleton(init_tokenizer)

    snowplow = providers.DependenciesContainer()

    generations = providers.Container(
        ContainerCodeGenerations,
        tokenizer=tokenizer,
        vertex_code_bison=models.vertex_code_bison,
        anthropic_claude=models.anthropic_claude,
        anthropic_claude_chat=models.anthropic_claude_chat,
        litellm_chat=models.litellm_chat,
        agent_model=models.agent_model,
        amazon_q_model=models.amazon_q_model,
        snowplow_instrumentator=snowplow.instrumentator,
    )

    completions = providers.Container(
        ContainerCodeCompletions,
        tokenizer=tokenizer,
        vertex_code_gecko=models.vertex_code_gecko,
        anthropic_claude=models.anthropic_claude,
        anthropic_claude_chat=models.anthropic_claude_chat,
        litellm=models.litellm,
        agent_model=models.agent_model,
        amazon_q_model=models.amazon_q_model,
        config=config,
        snowplow_instrumentator=snowplow.instrumentator,
    )
