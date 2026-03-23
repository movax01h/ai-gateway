import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.model_selection import PromptParams
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.caching import (
    CACHE_CONTROL_INJECTION_POINTS_KEY,
    REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST,
    CacheControlInjectionPointsConverter,
    default_cache_control_injection_points,
    filter_cache_control_injection_points,
)
from lib.prompts.caching import set_prompt_caching_enabled_to_current_request


class TestFilterCacheControlInjectionPoints:
    @pytest.mark.parametrize(
        "model_kwargs,prompt_cache_enabled,expected_cache_control_injection_points",
        [
            (
                {
                    CACHE_CONTROL_INJECTION_POINTS_KEY: [
                        {
                            "location": "message",
                            "index": 0,
                        },
                        {
                            "location": "message",
                            "index": -1,
                            REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST: "true",
                        },
                    ]
                },
                "true",
                [
                    {
                        "location": "message",
                        "index": 0,
                    },
                    {
                        "location": "message",
                        "index": -1,
                    },
                ],
            ),
            (
                {
                    CACHE_CONTROL_INJECTION_POINTS_KEY: [
                        {
                            "location": "message",
                            "index": 0,
                        },
                        {
                            "location": "message",
                            "index": -1,
                            REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST: "true",
                        },
                    ]
                },
                "false",
                [
                    {
                        "location": "message",
                        "index": 0,
                    },
                ],
            ),
            ({"other_key": "other_value"}, "true", None),
        ],
    )
    def test_filter(
        self,
        model_kwargs,
        prompt_cache_enabled,
        expected_cache_control_injection_points,
    ):
        set_prompt_caching_enabled_to_current_request(prompt_cache_enabled)

        filter_cache_control_injection_points(model_kwargs)

        if expected_cache_control_injection_points is None:
            assert len(model_kwargs) > 0
        else:
            assert (
                model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY]
                == expected_cache_control_injection_points
            )


class TestDefaultCacheControlInjectionPoints:
    def test_single_turn_prompt(self):
        prompt_template = {
            "system": "You are a helpful assistant.",
            "user": "Hello",
        }

        result = default_cache_control_injection_points(prompt_template)

        assert result == [{"location": "message", "index": 0}]

    def test_multi_turn_prompt(self):
        prompt_template = {
            "system": "You are a helpful assistant.",
            "user": "Hello",
            "placeholder": "history",
        }

        result = default_cache_control_injection_points(prompt_template)

        assert result == [
            {"location": "message", "index": 0},
            {
                "location": "message",
                "index": -1,
                REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST: "true",
            },
        ]


class TestDefaultCacheControlAppliedToPrompts:
    """Verify that prompts without explicit cache_control_injection_points in their YAML config automatically get
    caching defaults — the core cost-saving behaviour this feature provides."""

    def test_litellm_single_turn_prompt_gets_system_message_cached(self):
        """LiteLLM single-turn prompt: system message is always cached
        (passed through to LiteLLM which handles it natively)."""
        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template={"system": "You are helpful.", "user": "Hi"},
            model_class_provider=ModelClassProvider.LITE_LLM,
        )

        assert model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] == [
            {"location": "message", "index": 0},
        ]

    def test_litellm_multi_turn_prompt_caches_history_when_header_enabled(self):
        """LiteLLM multi-turn prompt: both system message and conversation
        history are cached when the request header enables caching."""
        set_prompt_caching_enabled_to_current_request("true")

        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template={
                "system": "You are helpful.",
                "user": "Hi",
                "placeholder": "history",
            },
            model_class_provider=ModelClassProvider.LITE_LLM,
        )

        assert model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] == [
            {"location": "message", "index": 0},
            {"location": "message", "index": -1},
        ]

    def test_litellm_multi_turn_prompt_only_caches_system_when_header_disabled(self):
        """LiteLLM multi-turn prompt: only the system message is cached
        when the request header does not enable caching."""

        set_prompt_caching_enabled_to_current_request("false")

        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template={
                "system": "You are helpful.",
                "user": "Hi",
                "placeholder": "history",
            },
            model_class_provider=ModelClassProvider.LITE_LLM,
        )

        assert model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] == [
            {"location": "message", "index": 0},
        ]

    @pytest.mark.parametrize(
        "provider",
        [
            ModelClassProvider.AMAZON_Q,
            ModelClassProvider.LITE_LLM_COMPLETION,
            ModelClassProvider.OPENAI,
            ModelClassProvider.GOOGLE_GENAI,
        ],
    )
    def test_unsupported_provider_gets_no_defaults(self, provider):
        """Providers not in the allowlist should not get any cache_control_injection_points."""
        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template={"system": "You are helpful.", "user": "Hi"},
            model_class_provider=provider,
        )

        assert CACHE_CONTROL_INJECTION_POINTS_KEY not in model_kwargs

    @pytest.mark.parametrize(
        "provider",
        [
            ModelClassProvider.LITE_LLM,
            ModelClassProvider.ANTHROPIC,
        ],
    )
    def test_supported_provider_without_prompt_template_gets_no_defaults(
        self, provider
    ):
        """Supported providers should work without caching when no prompt_template is provided."""
        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template=None,
            model_class_provider=provider,
        )

        assert CACHE_CONTROL_INJECTION_POINTS_KEY not in model_kwargs

    @pytest.mark.parametrize(
        "provider",
        [
            ModelClassProvider.LITE_LLM,
            ModelClassProvider.ANTHROPIC,
        ],
    )
    def test_supported_provider_with_prompt_template_gets_defaults(self, provider):
        """Supported providers should get caching defaults when prompt_template is provided."""
        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(timeout=60),
            model_metadata=None,
            prompt_template={"system": "You are helpful.", "user": "Hi"},
            model_class_provider=provider,
        )

        assert model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] == [
            {"location": "message", "index": 0},
        ]

    def test_explicit_yaml_config_is_not_overridden(self):
        """When a prompt YAML already defines cache_control_injection_points, the defaults must NOT override it."""
        explicit_points = [{"location": "message", "index": 0}]

        model_kwargs = Prompt._build_model_kwargs(
            params=PromptParams(cache_control_injection_points=explicit_points),
            model_metadata=None,
            prompt_template={
                "system": "You are helpful.",
                "user": "Hi",
                "placeholder": "history",
            },
            model_class_provider=ModelClassProvider.LITE_LLM,
        )

        # Should keep the explicit single-point config, NOT add the multi-turn default
        assert model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] == [
            {"location": "message", "index": 0},
        ]


class TestCacheControlInjectionPointsConverter:
    @pytest.fixture(name="model_class_provider")
    def model_class_provider_fixture(self):
        return ModelClassProvider.ANTHROPIC

    @pytest.fixture(name="cache_control_injection_points")
    def cache_control_injection_points_fixture(self):
        return [{"location": "message", "index": 0}]

    @pytest.fixture(name="converter")
    def converter_fixture(self, model_class_provider, cache_control_injection_points):
        return CacheControlInjectionPointsConverter().bind(
            model_class_provider=model_class_provider,
            cache_control_injection_points=cache_control_injection_points,
        )

    @pytest.mark.parametrize(
        "messages",
        [
            [SystemMessage(content="Hi, I'm Duo"), HumanMessage(content="What's up?")],
            [
                SystemMessage(content=["hi", "there"]),
                HumanMessage(content="What's up?"),
            ],
            [
                SystemMessage(
                    content=[
                        {"text": "hi", "type": "text"},
                        {"text": "there", "type": "text"},
                    ]
                ),
                HumanMessage(content="What's up?"),
            ],
        ],
    )
    def test_invoke(self, messages, converter: CacheControlInjectionPointsConverter):
        prompt_value = ChatPromptValue(messages=messages)
        original_value = prompt_value.model_copy(deep=True)

        response = converter.invoke(prompt_value)

        first_message = response.to_messages()[0]

        assert isinstance(first_message.content, list)
        assert isinstance(first_message.content[-1], dict)
        assert first_message.content[-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

        assert (
            original_value.to_messages() == prompt_value.to_messages()
        ), "Original list has been modified"

    @pytest.mark.parametrize(
        "model_class_provider",
        [
            ModelClassProvider.AMAZON_Q,
            ModelClassProvider.LITE_LLM_COMPLETION,
            ModelClassProvider.OPENAI,
            ModelClassProvider.GOOGLE_GENAI,
        ],
    )
    def test_invoke_with_unsupported_client(
        self, converter: CacheControlInjectionPointsConverter
    ):
        prompt_value = ChatPromptValue(messages=[])

        with pytest.raises(NotImplementedError) as ex:
            converter.invoke(prompt_value)
            assert (
                str(ex)
                == "cache_control_injection_points is specified but conversion method is not defined"
            )
