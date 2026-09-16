# pylint: disable=file-naming-for-tests,unsubscriptable-object,unsupported-membership-test
"""Tests for post processor creation in code completions."""

import pytest
from dependency_injector.providers import Factory

from ai_gateway.api.v2.code.completions import _create_post_processor_for_model
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor,
    PostProcessorOperation,
)
from ai_gateway.config import Config
from ai_gateway.models import KindModelProvider


@pytest.fixture(name="mock_config")
def mock_config_fixture():
    """Create a mock config with feature flags."""
    config = Config()
    config.feature_flags.excl_post_process = lambda: []
    config.feature_flags.fireworks_score_threshold = lambda: {
        "codestral-2501": 0.5,
        "qwen2p5-coder-7b": 0.3,
    }
    return config


class TestCreatePostProcessorForModel:
    """Tests for _create_post_processor_for_model function."""

    @pytest.mark.parametrize("model_metadata_provider", [KindModelProvider.VERTEX_AI])
    def test_vertex_codestral_creates_post_processor_with_strip_asterisks(
        self, mock_config, model_metadata
    ):
        """Test that Vertex Codestral creates post processor with STRIP_ASTERISKS."""
        result = _create_post_processor_for_model(
            model_name="codestral-2501",
            config=mock_config,
            model_metadata=model_metadata,
        )

        assert isinstance(result, Factory)
        instance = result(code_context="test context")
        assert isinstance(instance, PostProcessor)
        assert PostProcessorOperation.STRIP_ASTERISKS in instance.extras
        assert instance.exclude == mock_config.feature_flags.excl_post_process()

    @pytest.mark.parametrize("model_metadata_provider", [KindModelProvider.FIREWORKS])
    def test_fireworks_creates_post_processor_with_filter_score_and_fix_truncation(
        self, mock_config, model_metadata
    ):
        """Test that Fireworks creates post processor with FILTER_SCORE and FIX_TRUNCATION."""
        result = _create_post_processor_for_model(
            model_name="codestral-2501",
            config=mock_config,
            model_metadata=model_metadata,
        )

        assert isinstance(result, Factory)
        instance = result(code_context="test context")
        assert isinstance(instance, PostProcessor)
        assert PostProcessorOperation.FILTER_SCORE in instance.extras
        assert PostProcessorOperation.FIX_TRUNCATION in instance.extras
        assert instance.exclude == mock_config.feature_flags.excl_post_process()
        assert "score_threshold" in result.kwargs
        assert result.kwargs["score_threshold"] == 0.5
        assert isinstance(result.kwargs["score_threshold"], float)
        assert instance.score_threshold == 0.5
        assert isinstance(instance.score_threshold, float)

    @pytest.mark.parametrize("model_metadata_provider", [KindModelProvider.FIREWORKS])
    def test_fireworks_different_model_uses_correct_threshold(
        self, mock_config, model_metadata
    ):
        """Test that different Fireworks models use their specific thresholds."""
        result = _create_post_processor_for_model(
            model_name="qwen2p5-coder-7b",
            config=mock_config,
            model_metadata=model_metadata,
        )

        assert isinstance(result, Factory)
        instance = result(code_context="test context")
        assert instance.score_threshold == 0.3

    @pytest.mark.parametrize("model_metadata_provider", [KindModelProvider.ANTHROPIC])
    def test_other_providers_return_none(self, mock_config, model_metadata):
        """Test that other providers return None."""
        result = _create_post_processor_for_model(
            model_name="claude-sonnet-4-5-20250929",
            config=mock_config,
            model_metadata=model_metadata,
        )
        assert result is None

    @pytest.mark.parametrize(
        ("model_metadata_provider", "is_custom_model"), [("openai", True)]
    )
    def test_self_hosted_metadata_gets_cleanup_chain(self, mock_config, model_metadata):
        """Test that self-hosted model metadata gets the completion cleanup chain."""
        result = _create_post_processor_for_model(
            model_name="some-model",
            config=mock_config,
            model_metadata=model_metadata,
        )

        assert isinstance(result, Factory)
        instance = result(code_context="test context")
        assert isinstance(instance, PostProcessor)
        assert instance.pre_extras == [PostProcessorOperation.EXTRACT_FENCED_CODE]
        assert instance.extras == [PostProcessorOperation.FIX_TRUNCATION]
        assert instance.exclude == mock_config.feature_flags.excl_post_process()

    @pytest.mark.parametrize("model_metadata_provider", [KindModelProvider.VERTEX_AI])
    def test_vertex_non_codestral_returns_none(self, mock_config, model_metadata):
        """Test that Vertex AI with non-codestral model returns None."""
        result = _create_post_processor_for_model(
            model_name="code-bison",
            config=mock_config,
            model_metadata=model_metadata,
        )
        assert result is None
