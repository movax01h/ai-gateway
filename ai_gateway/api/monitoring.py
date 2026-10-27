import asyncio
import functools
from typing import Annotated, Any, Awaitable, Callable

from dependency_injector.providers import Configuration, Factory
from fastapi import APIRouter, Depends, Request
from fastapi_health import health
from gitlab_cloud_connector import cloud_connector_ready
from langsmith import tracing_context

from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_completions_agent_factory_provider,
    get_config,
    get_prompt_registry,
    get_search_factory_provider,
)
from ai_gateway.code_suggestions.completions import CodeCompletions
from ai_gateway.code_suggestions.processing.post.completions import (
    create_post_processor_for_model_metadata,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers.treesitter import CodeParser
from ai_gateway.model_metadata import (
    completion_context_max_percent_for_model_metadata,
    resolve_provider_aware_metadata,
)
from ai_gateway.model_selection import LLMDefinition, ModelSelectionConfig
from ai_gateway.prompts.base import BasePromptRegistry
from ai_gateway.searches import Searcher

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)

# Avoid calling out to the models multiple times from this public, unauthenticated endpoint.
# this is not threadsafe, but that should be fine, we aren't issuing multiple of
# these calls in parallel. When the instance is marked as ready, we won't be modifying
# the list anymore.
validated: set[str] = set()


def single_validation(
    key: str,
):
    def _decorator(
        func: Callable[[Any], Awaitable[bool]],
    ) -> Callable[[Any, Any], Awaitable[bool]]:

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs) -> bool:
            if key in validated:
                return True

            result = await func(*args, **kwargs)
            validated.add(key)

            return result

        return _wrapper

    return _decorator


async def validate_default_models_available(
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
) -> bool:
    return await prompt_registry.validate_default_models()


_CONTENT_ABOVE_CURSOR = "def hello_world():"


async def execute_default_code_completions(
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    config: Configuration,
) -> None:
    """Run a code completion with each default model, the way `/v4/code/suggestions` does when no model is picked.

    Requests without a model get a weighted pick among the defaults, so each one is checked. Skips user authorization,
    event tracking, billing and the rate limit fallback.
    """
    model_selection = ModelSelectionConfig.instance()
    completions_config = model_selection.get_resolved_unit_primitive_config_map()[
        "code_completions"
    ]

    with tracing_context(enabled=False):
        await asyncio.gather(
            *(
                _execute_code_completion(
                    model_selection.get_model(identifier),
                    prompt_registry,
                    completions_agent_factory,
                    config,
                )
                for identifier in completions_config.default_model_identifiers
            )
        )

    # Post-processing skips the parser when a model returns an empty completion, so run it directly
    await CodeParser.from_language_id(_CONTENT_ABOVE_CURSOR, LanguageId.PYTHON)


async def _execute_code_completion(
    llm_definition: LLMDefinition,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    config: Configuration,
) -> None:
    # Mirrors `code_completion` in ai_gateway/code_suggestions/handler.py, without the user authorization and tracking
    model_metadata = resolve_provider_aware_metadata(
        llm_definition,
        provider_keys=config.model_keys(),
        fireworks_api_base_url=config.fireworks_api_base_url(),
        mock_model_responses=config.mock_model_responses(),
    )
    prompt = prompt_registry.get(
        "code_suggestions/completions", None, model_metadata=model_metadata
    )
    post_processor = create_post_processor_for_model_metadata(
        model_metadata,
        config.feature_flags.excl_post_process(),
        config.feature_flags.fireworks_score_threshold(),
    )
    code_completions = completions_agent_factory(
        model__prompt=prompt,
        post_processor=post_processor,
        model_metadata=model_metadata,
    )

    kwargs: dict[str, Any] = {}
    context_max_percent = completion_context_max_percent_for_model_metadata(
        model_metadata
    )
    if context_max_percent is not None:
        kwargs["context_max_percent"] = context_max_percent

    await code_completions.execute(
        prefix=_CONTENT_ABOVE_CURSOR,
        suffix="",
        file_name="monitoring.py",
        editor_lang="python",
        **kwargs,
    )


@single_validation("default_code_completions")
async def validate_default_code_completions_available(
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    completions_agent_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_agent_factory_provider),
    ],
    config: Annotated[Configuration, Depends(get_config)],
) -> bool:
    """Exercise the code path of a `/v4/code/suggestions` completion that doesn't pick a model.

    `validate_default_models_available` only checks the models through a generic prompt, so it doesn't catch failures
    in the completions prompt, provider defaults or the tree-sitter parser used by post-processing.
    """
    await execute_default_code_completions(
        prompt_registry, completions_agent_factory, config
    )
    return True


async def validate_cloud_connector_ready(
    config: Annotated[Configuration, Depends(get_config)],
    request: Request,
) -> bool:
    """Always pass for Self-Hosted-Models.

    This is temporary. With the current CC <-> AI GW interface, we can't easily skip CDot sync just for SHM only. At the
    same time, we can't require SHM to always connect to CustomersDot - as CustomersDot connection/sync is not needed
    for AI GW to work in SHM context. So we shouldn't fail the probe for customer setups that can't or don't want to
    reach CustomersDot. As soon as
    https://gitlab.com/gitlab-org/gitlab/-/issues/517088
    is complete, we should stop passing CustomersDot
    as a provider for SHM setups. With that, we can drop the check for SHM in this file.
    """
    if config.custom_models.enabled():
        return True

    provider = request.app.state.cloud_connector_auth_provider
    return cloud_connector_ready(provider)


async def validate_doc_search_available(
    search_factory: Annotated[Factory[Searcher], Depends(get_search_factory_provider)],
    config: Annotated[Configuration, Depends(get_config)],
) -> bool:
    try:
        searcher = search_factory()
        fallback_version = config.vertex_search.fallback_datastore_version()

        await searcher.search(
            query="can I upload images to GitLab repo?",
            page_size=1,
            gl_version=fallback_version,
        )

        return True
    except Exception:
        # Returning False to allow the health check to return 503 instead of 500.
        return False


router.add_api_route("/healthz", health([]))
router.add_api_route(
    "/ready",
    health(
        [
            validate_default_models_available,
            validate_default_code_completions_available,
            validate_cloud_connector_ready,
            validate_doc_search_available,
        ]
    ),
)
