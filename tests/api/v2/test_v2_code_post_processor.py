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
    config.feature_flags.excl_post_process = []
    config.feature_flags.fireworks_score_threshold = lambda: {
        "codestral-2501": 0.5,
        "qwen2p5-coder-7b": 0.3,
    }
    return config


class TestCreatePostProcessorForModel:
    """Tests for _create_post_processor_for_model function."""

    def test_vertex_codestral_creates_post_processor_with_strip_asterisks(
        self, mock_config
    ):
        """Test that Vertex Codestral creates post processor with STRIP_ASTERISKS."""
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.VERTEX_AI,
            model_name="codestral-2501",
            config=mock_config,
        )

        assert isinstance(result, Factory)
        # Create an instance to verify configuration
        instance = result(code_context="test context")
        assert isinstance(instance, PostProcessor)
        assert PostProcessorOperation.STRIP_ASTERISKS in instance.extras
        assert instance.exclude == mock_config.feature_flags.excl_post_process

    def test_fireworks_creates_post_processor_with_filter_score_and_fix_truncation(
        self, mock_config
    ):
        """Test that Fireworks creates post processor with FILTER_SCORE and FIX_TRUNCATION."""
        model_name = "codestral-2501"
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.FIREWORKS,
            model_name=model_name,
            config=mock_config,
        )

        assert isinstance(result, Factory)
        # Create an instance to verify configuration
        instance = result(code_context="test context")
        assert isinstance(instance, PostProcessor)
        assert PostProcessorOperation.FILTER_SCORE in instance.extras
        assert PostProcessorOperation.FIX_TRUNCATION in instance.extras
        assert instance.exclude == mock_config.feature_flags.excl_post_process
        # Verify score_threshold is set correctly as a float, not a dict
        assert "score_threshold" in result.kwargs
        assert result.kwargs["score_threshold"] == 0.5
        assert isinstance(result.kwargs["score_threshold"], float)
        assert instance.score_threshold == 0.5
        assert isinstance(instance.score_threshold, float)

    def test_fireworks_different_model_uses_correct_threshold(self, mock_config):
        """Test that different Fireworks models use their specific thresholds."""
        model_name = "qwen2p5-coder-7b"
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.FIREWORKS,
            model_name=model_name,
            config=mock_config,
        )

        assert isinstance(result, Factory)
        instance = result(code_context="test context")
        assert instance.score_threshold == 0.3

    def test_other_providers_return_none(self, mock_config):
        """Test that other providers return None."""
        # Test Anthropic
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            config=mock_config,
        )
        assert result is None

        # Test LiteLLM
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.LITELLM,
            model_name="some-model",
            config=mock_config,
        )
        assert result is None

    def test_vertex_non_codestral_returns_none(self, mock_config):
        """Test that Vertex AI with non-codestral model returns None."""
        result = _create_post_processor_for_model(
            model_provider=KindModelProvider.VERTEX_AI,
            model_name="code-bison",
            config=mock_config,
        )
        assert result is None
