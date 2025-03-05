from unittest.mock import Mock

import pytest

from ai_gateway.api.v2.code.model_provider_handlers import (
    AnthropicHandler,
    FireworksHandler,
    LegacyHandler,
    LiteLlmHandler,
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
