"""Tests for v4 completion post-processor factory selection."""

from types import SimpleNamespace

import pytest
from dependency_injector.providers import Factory

from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor,
    PostProcessorOperation,
    create_post_processor_for_model_metadata,
)


def _model_metadata(custom_llm_provider, model, family):
    params = SimpleNamespace(custom_llm_provider=custom_llm_provider, model=model)
    llm_definition = SimpleNamespace(params=params, family=family)
    return SimpleNamespace(llm_definition=llm_definition)


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
