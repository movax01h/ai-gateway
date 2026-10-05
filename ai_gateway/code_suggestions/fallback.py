from typing import Any, Dict, Optional

from ai_gateway.code_suggestions.completions import FallbackFactory, FallbackModel
from ai_gateway.code_suggestions.processing.post.completions import (
    create_post_processor_for_model_metadata,
)
from ai_gateway.model_metadata import (
    TypeModelMetadata,
    build_fallback_code_completions_metadata,
)
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.prompts import BasePromptRegistry
from lib.context import StarletteUser


def build_rate_limit_fallback(
    model_metadata: TypeModelMetadata,
    prompt_registry: BasePromptRegistry,
    current_user: StarletteUser,
    fireworks_api_base_url: str,
    model_keys: Dict[str, Any],
    using_cache: bool,
    mock_model_responses: bool,
    excl_post_process: list[str],
    fireworks_score_thresholds: dict[str, float],
) -> Optional[FallbackFactory]:
    """Return a builder for the fallback model, or ``None`` when no fallback applies.

    The builder runs only after a rate limit, so a request that never hits one builds no second prompt. The fallback
    prompt skips the request event, because the primary prompt already counted this request.
    """
    fallback_metadata = build_fallback_code_completions_metadata(
        model_metadata,
        fireworks_api_base_url=fireworks_api_base_url,
        model_keys=model_keys,
        user=current_user,
        using_cache=using_cache,
        mock_model_responses=mock_model_responses,
    )
    if fallback_metadata is None:
        return None

    def build() -> FallbackModel:
        prompt = prompt_registry.get_on_behalf(
            current_user,
            "code_suggestions/completions",
            model_metadata=fallback_metadata,
            track_event=False,
        )

        return FallbackModel(
            model=AgentModel(prompt, fallback_metadata.llm_definition),
            model_metadata=fallback_metadata,
            post_processor=create_post_processor_for_model_metadata(
                fallback_metadata, excl_post_process, fireworks_score_thresholds
            ),
        )

    return build
