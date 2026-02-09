"""Tests for lib/context/llm_operations module."""

import unittest

from lib.context.llm_operations import (
    get_llm_operations,
    get_token_usage,
    init_llm_operations,
    init_token_usage,
    llm_operations,
    token_usage,
)


class TestTokenUsage(unittest.TestCase):
    """Tests for token usage context variable and functions."""

    def setUp(self):
        # Reset context var before each test
        token_usage.set(None)

    def tearDown(self):
        # Clean up after each test
        token_usage.set(None)

    def test_token_usage_default(self):
        self.assertIsNone(token_usage.get())

    def test_init_token_usage(self):
        init_token_usage()
        self.assertEqual(token_usage.get(), {})

    def test_get_token_usage_returns_and_resets(self):
        init_token_usage()
        # Manually set some usage
        current = token_usage.get()
        current["model1"] = {"input_tokens": 100, "output_tokens": 50}
        token_usage.set(current)

        # Get should return the usage
        result = get_token_usage()
        self.assertEqual(result, {"model1": {"input_tokens": 100, "output_tokens": 50}})

        # And reset to None
        self.assertIsNone(token_usage.get())

    def test_get_token_usage_when_not_initialized(self):
        result = get_token_usage()
        self.assertIsNone(result)


class TestLlmOperations(unittest.TestCase):
    """Tests for LLM operations context variable and functions."""

    def setUp(self):
        # Reset context var before each test
        llm_operations.set(None)

    def tearDown(self):
        # Clean up after each test
        llm_operations.set(None)

    def test_llm_operations_default(self):
        self.assertIsNone(llm_operations.get())

    def test_init_llm_operations(self):
        init_llm_operations()
        self.assertEqual(llm_operations.get(), [])

    def test_get_llm_operations_returns_and_resets(self):
        init_llm_operations()
        # Manually add an operation
        current = llm_operations.get()
        current.append(
            {
                "token_count": 150,
                "model_id": "claude-3",
                "model_engine": "anthropic",
                "model_provider": "anthropic",
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        )
        llm_operations.set(current)

        # Get should return the operations
        result = get_llm_operations()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_id"], "claude-3")

        # And reset to None
        self.assertIsNone(llm_operations.get())

    def test_get_llm_operations_when_not_initialized(self):
        result = get_llm_operations()
        self.assertIsNone(result)
