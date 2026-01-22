from unittest.mock import Mock

import pytest

from ai_gateway.api.v2.code.model_provider_handlers import (
    AnthropicHandler,
    FireworksHandler,
    LegacyHandler,
    LiteLlmHandler,
    VertexHandler,
)


class TestAnthropicHandler:
    @pytest.mark.parametrize(
        ("prompt_version", "prompt", "want_completion_params"),
        [
            (
                1,
                "foo",
                {},
            ),
            (
                3,
                "foo",
                {"raw_prompt": "foo"},
            ),
        ],
    )
    def test_update_completion_params(
        self, prompt_version, prompt, want_completion_params
    ):
        payload = Mock(prompt_version=prompt_version, prompt=prompt)
        request = Mock()
        completion_params = {}

        AnthropicHandler(payload, request, completion_params).update_completion_params()

        assert completion_params == want_completion_params


class TestLiteLlmHandler:
    @pytest.mark.parametrize(
        ("context", "want_completion_params"),
        [
            (
                [],
                {},
            ),
            (
                [
                    Mock(
                        type="file",
                        name="foo.py",
                        content="from typing import List",
                    ),
                    Mock(
                        type="file",
                        name="bar.py",
                        content="from typing import Any",
                    ),
                ],
                {
                    "code_context": [
                        "from typing import List",
                        "from typing import Any",
                    ]
                },
            ),
        ],
    )
    def test_update_completion_params(self, context, want_completion_params):
        payload = Mock(context=context)
        request = Mock()
        completion_params = {}

        LiteLlmHandler(payload, request, completion_params).update_completion_params()

        assert completion_params == want_completion_params


class TestFireworksHandler:
    @pytest.mark.parametrize(
        ("context", "want_completion_params"),
        [
            (
                [],
                {"max_output_tokens": 48, "context_max_percent": 0.3},
            ),
            (
                [
                    Mock(
                        type="file",
                        name="foo.py",
                        content="from typing import List",
                    ),
                    Mock(
                        type="file",
                        name="bar.py",
                        content="from typing import Any",
                    ),
                ],
                {
                    "max_output_tokens": 48,
                    "context_max_percent": 0.3,
                    "code_context": [
                        "from typing import List",
                        "from typing import Any",
                    ],
                },
            ),
        ],
    )
    def test_update_completion_params(self, context, want_completion_params):
        payload = Mock(context=context)
        request = Mock()
        completion_params = {}

        FireworksHandler(payload, request, completion_params).update_completion_params()

        assert completion_params == want_completion_params


class TestVertexHandler:
    @pytest.mark.parametrize(
        ("context", "initial_model_name", "want_completion_params", "want_model_name"),
        [
            (
                [],
                "claude-sonnet-4-5@20250929",
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                },
                "claude-sonnet-4-5@20250929",
            ),
            (
                [
                    Mock(
                        type="file",
                        name="foo.py",
                        content="from typing import List",
                    ),
                    Mock(
                        type="file",
                        name="bar.py",
                        content="from typing import Any",
                    ),
                ],
                "claude-sonnet-4-5@20250929",
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                    "code_context": [
                        "from typing import List",
                        "from typing import Any",
                    ],
                },
                "claude-sonnet-4-5@20250929",
            ),
            (
                [],
                None,
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                },
                "codestral-2508",
            ),
        ],
    )
    def test_update_completion_params_preserves_model_name(
        self, context, initial_model_name, want_completion_params, want_model_name
    ):
        """Test that VertexHandler preserves the model_name when provided, and only defaults to CODESTRAL_2501 when not
        provided."""
        payload = Mock(context=context, model_name=initial_model_name)
        request = Mock()
        completion_params = {}

        VertexHandler(payload, request, completion_params).update_completion_params()

        assert completion_params == want_completion_params
        # Ensure model_name is preserved when provided, or defaults to CODESTRAL_2501 when not
        assert payload.model_name == want_model_name


class TestLegacyHandler:
    @pytest.mark.parametrize(
        ("choices_count", "context", "headers", "want_completion_params"),
        [
            (
                0,
                [],
                {},
                {},
            ),
            (
                1,
                [],
                {},
                {"candidate_count": 1},
            ),
            (
                0,
                [
                    Mock(
                        type="file",
                        name="foo.py",
                        content="from typing import List",
                    ),
                    Mock(
                        type="file",
                        name="bar.py",
                        content="from typing import Any",
                    ),
                ],
                {"X-Gitlab-Language-Server-Version": "7.17.1"},
                {
                    "code_context": [
                        "from typing import List",
                        "from typing import Any",
                    ],
                },
            ),
            (
                0,
                [],
                {"X-Gitlab-Language-Server-Version": "7.17.1"},
                {},
            ),
            (
                0,
                [
                    Mock(
                        type="file",
                        name="foo.py",
                        content="from typing import List",
                    ),
                    Mock(
                        type="file",
                        name="bar.py",
                        content="from typing import Any",
                    ),
                ],
                {"X-Gitlab-Language-Server-Version": "4.15.0"},
                {},
            ),
        ],
    )
    def test_update_completion_params(
        self, choices_count, context, headers, want_completion_params
    ):
        payload = Mock(choices_count=choices_count, context=context)
        request = Mock(headers=headers)
        completion_params = {}

        LegacyHandler(payload, request, completion_params).update_completion_params()

        assert completion_params == want_completion_params
