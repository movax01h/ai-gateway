from unittest.mock import MagicMock, patch

from ai_gateway.code_suggestions import fallback as fallback_module
from ai_gateway.code_suggestions.fallback import build_rate_limit_fallback
from ai_gateway.models.agent_model import AgentModel


def test_builder_creates_the_fallback_model_only_when_called():
    fallback_metadata = MagicMock()
    prompt_registry = MagicMock()
    post_processor = MagicMock()

    with (
        patch.object(
            fallback_module,
            "build_fallback_code_completions_metadata",
            return_value=fallback_metadata,
        ),
        patch.object(
            fallback_module,
            "create_post_processor_for_model_metadata",
            return_value=post_processor,
        ),
    ):
        build = build_rate_limit_fallback(
            MagicMock(),
            prompt_registry,
            MagicMock(),
            fireworks_api_base_url="https://api.fireworks.ai/inference/v1",
            model_keys={},
            using_cache=True,
            mock_model_responses=False,
            excl_post_process=[],
            fireworks_score_thresholds={},
        )
        prompt_registry.get_on_behalf.assert_not_called()

        fallback = build()

    call_kwargs = prompt_registry.get_on_behalf.call_args.kwargs
    assert call_kwargs["model_metadata"] is fallback_metadata
    assert call_kwargs["track_event"] is False
    assert isinstance(fallback.model, AgentModel)
    assert fallback.model.prompt is prompt_registry.get_on_behalf.return_value
    assert fallback.model_metadata is fallback_metadata
    assert fallback.post_processor is post_processor
