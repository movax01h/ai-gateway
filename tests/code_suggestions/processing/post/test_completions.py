"""Tests for v4 completion post-processor factory selection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from dependency_injector.providers import Factory

from ai_gateway.code_suggestions.processing.post.completions import (
    ORDERED_POST_PROCESSORS,
    PostProcessor,
    PostProcessorOperation,
    create_post_processor_for_model_metadata,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId


def _model_metadata(custom_llm_provider, model, family, is_custom_model=False):
    params = SimpleNamespace(custom_llm_provider=custom_llm_provider, model=model)
    llm_definition = SimpleNamespace(params=params, family=family)
    return SimpleNamespace(
        llm_definition=llm_definition, is_custom_model=is_custom_model
    )


class TestCreatePostProcessorForModelMetadata:
    def test_fireworks_applies_filter_score_and_fix_truncation(self):
        metadata = _model_metadata(
            "fireworks_ai", "codestral-2508", ["completion_fim", "codestral"]
        )

        result = create_post_processor_for_model_metadata(
            metadata, [], {"codestral-2508": 0.5}
        )

        assert isinstance(result, Factory)
        instance = result(code_context="ctx")
        assert isinstance(instance, PostProcessor)
        assert PostProcessorOperation.FILTER_SCORE in instance.extras
        assert PostProcessorOperation.FIX_TRUNCATION in instance.extras
        assert instance.score_threshold == 0.5

    @pytest.mark.parametrize(
        ("model_name", "thresholds", "expected_threshold"),
        [
            ("codestral-2508", {"codestral-2508": 0.5}, 0.5),
            ("unknown-model", {"codestral-2508": 0.5}, None),
            (None, {}, None),
        ],
    )
    def test_fireworks_score_threshold_resolution(
        self, model_name, thresholds, expected_threshold
    ):
        metadata = _model_metadata("fireworks_ai", model_name, ["completion_fim"])

        result = create_post_processor_for_model_metadata(metadata, [], thresholds)

        assert result(code_context="ctx").score_threshold == expected_threshold

    def test_vertex_codestral_applies_strip_asterisks(self):
        metadata = _model_metadata(
            "vertex_ai", "codestral-2", ["completion_text", "codestral"]
        )

        result = create_post_processor_for_model_metadata(metadata, [], {})

        assert isinstance(result, Factory)
        instance = result(code_context="ctx")
        assert PostProcessorOperation.STRIP_ASTERISKS in instance.extras

    def test_vertex_non_codestral_returns_none(self):
        metadata = _model_metadata("vertex_ai", "gemini-2", ["completion_text"])

        assert create_post_processor_for_model_metadata(metadata, [], {}) is None

    def test_other_provider_returns_none(self):
        metadata = _model_metadata("anthropic", "claude", ["chat"])

        assert create_post_processor_for_model_metadata(metadata, [], {}) is None

    def test_self_hosted_extracts_fenced_code(self):
        metadata = _model_metadata(None, "mistral", ["mistral"], is_custom_model=True)

        result = create_post_processor_for_model_metadata(metadata, [], {})

        assert isinstance(result, Factory)
        instance = result(code_context="ctx")
        assert instance.pre_extras == [PostProcessorOperation.EXTRACT_FENCED_CODE]
        assert instance.extras == [PostProcessorOperation.FIX_TRUNCATION]


@pytest.mark.asyncio
async def test_process_keeps_fences_without_lang_id():
    processor = PostProcessor(
        "context",
        lang_id=None,
        pre_extras=[PostProcessorOperation.EXTRACT_FENCED_CODE],
        exclude=[str(op) for op in ORDERED_POST_PROCESSORS],
    )

    text = "```python\ncode\n```"

    assert await processor.process(text) == text


@pytest.mark.asyncio
async def test_process_extracts_fences_with_lang_id():
    processor = PostProcessor(
        "context",
        lang_id=LanguageId.PYTHON,
        pre_extras=[PostProcessorOperation.EXTRACT_FENCED_CODE],
        exclude=[str(op) for op in ORDERED_POST_PROCESSORS],
    )

    assert await processor.process("```python\ncode\n```") == "code"


@pytest.mark.asyncio
async def test_pre_extras_rebaseline_lets_truncation_fire_on_fenced_output():
    fix_truncation = AsyncMock(
        side_effect=lambda code_context, completion, **kwargs: completion
    )

    processor = PostProcessor(
        "context",
        lang_id=LanguageId.PYTHON,
        pre_extras=[PostProcessorOperation.EXTRACT_FENCED_CODE],
        extras=[PostProcessorOperation.FIX_TRUNCATION],
        exclude=[str(op) for op in ORDERED_POST_PROCESSORS],
    )

    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_truncation",
        fix_truncation,
    ):
        await processor.process("```python\nreturn a +", max_output_tokens_used=True)

    assert fix_truncation.call_args.args[1] == "return a +"
    assert fix_truncation.call_args.kwargs["raw_completion"] == "return a +"
