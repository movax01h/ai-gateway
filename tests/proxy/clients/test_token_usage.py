"""Tests for token usage models."""

import pytest

from ai_gateway.proxy.clients.token_usage import TokenUsage


class TestTokenUsage:
    """Test cases for TokenUsage model."""

    def test_token_usage_with_all_fields(self):
        """Test TokenUsage with all fields provided."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_auto_compute_total(self):
        """Test TokenUsage auto-computes total_tokens when not provided."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_with_zero_tokens(self):
        """Test TokenUsage with zero tokens."""
        usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
        )

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_defaults(self):
        """Test TokenUsage with default values."""
        usage = TokenUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_model_dump(self):
        """Test TokenUsage model_dump includes computed total."""
        usage = TokenUsage(
            input_tokens=75,
            output_tokens=25,
        )

        dumped = usage.model_dump()

        assert dumped["input_tokens"] == 75
        assert dumped["output_tokens"] == 25
        assert dumped["total_tokens"] == 100

    def test_token_usage_model_dump_with_explicit_total(self):
        """Test TokenUsage model_dump preserves explicit total_tokens."""
        usage = TokenUsage(
            input_tokens=75,
            output_tokens=25,
            total_tokens=150,  # Explicitly set different value
        )

        dumped = usage.model_dump()

        assert dumped["input_tokens"] == 75
        assert dumped["output_tokens"] == 25
        assert dumped["total_tokens"] == 150

    def test_token_usage_partial_tokens(self):
        """Test TokenUsage with only input tokens."""
        usage = TokenUsage(input_tokens=100)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 0
        assert usage.total_tokens == 100

    def test_token_usage_only_output_tokens(self):
        """Test TokenUsage with only output tokens."""
        usage = TokenUsage(output_tokens=50)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 50
        assert usage.total_tokens == 50

    @pytest.mark.parametrize(
        "input_tokens,output_tokens,expected_total",
        [
            (10, 20, 30),
            (0, 100, 100),
            (100, 0, 100),
            (1, 1, 2),
            (999, 1, 1000),
        ],
    )
    def test_token_usage_various_combinations(
        self, input_tokens, output_tokens, expected_total
    ):
        """Test TokenUsage with various token combinations."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        assert usage.total_tokens == expected_total

    def test_token_usage_with_cache_metrics(self):
        """Test TokenUsage with cache metrics."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_creation_input_tokens == 20
        assert usage.cache_read_input_tokens == 30

    def test_token_usage_cache_defaults(self):
        """Test TokenUsage cache metrics default to zero."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_token_usage_only_cache_creation(self):
        """Test TokenUsage with only cache creation tokens."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=25,
        )

        assert usage.cache_creation_input_tokens == 25
        assert usage.cache_read_input_tokens == 0

    def test_token_usage_only_cache_read(self):
        """Test TokenUsage with only cache read tokens."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=40,
        )

        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 40

    def test_to_billing_metadata_basic(self):
        """Test to_billing_metadata with basic token usage."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
        )

        metadata = usage.to_billing_metadata()

        assert metadata == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_to_billing_metadata_with_cache_read(self):
        """Test to_billing_metadata includes cache_read_input_tokens when non-zero."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
        )

        metadata = usage.to_billing_metadata()

        assert metadata == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_read_input_tokens": 30,
        }

    def test_to_billing_metadata_with_cache_creation(self):
        """Test to_billing_metadata includes cache_creation_input_tokens when non-zero."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
        )

        metadata = usage.to_billing_metadata()

        assert metadata == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_creation_input_tokens": 20,
        }

    def test_to_billing_metadata_with_all_cache_tokens(self):
        """Test to_billing_metadata includes both cache token types when non-zero."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )

        metadata = usage.to_billing_metadata()

        assert metadata == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
        }

    def test_to_billing_metadata_excludes_zero_cache_tokens(self):
        """Test to_billing_metadata excludes cache tokens when they're zero."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        metadata = usage.to_billing_metadata()

        assert metadata == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        assert "cache_creation_input_tokens" not in metadata
        assert "cache_read_input_tokens" not in metadata
