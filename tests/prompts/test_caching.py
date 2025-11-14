import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.prompts.caching import (
    CACHE_CONTROL_INJECTION_POINTS_KEY,
    REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST,
    CacheControlInjectionPointsConverter,
    filter_cache_control_injection_points,
)
from ai_gateway.prompts.config.models import ModelClassProvider
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context
from lib.prompts.caching import set_prompt_caching_enabled_to_current_request


class TestFilterCacheControlInjectionPoints:
    @pytest.mark.parametrize(
        "model_kwargs,prompt_cache_enabled,feature_flag,expected_cache_control_injection_points",
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
                True,
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
                True,
                [
                    {
                        "location": "message",
                        "index": 0,
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
                "true",
                False,
                [
                    {
                        "location": "message",
                        "index": 0,
                    },
                ],
            ),
            ({"other_key": "other_value"}, "true", True, None),
        ],
    )
    def test_filter(
        self,
        model_kwargs,
        prompt_cache_enabled,
        feature_flag,
        expected_cache_control_injection_points,
    ):
        if feature_flag:
            current_feature_flag_context.set(
                {FeatureFlag.AI_GATEWAY_ALLOW_CONVERSATION_CACHING}
            )

        set_prompt_caching_enabled_to_current_request(prompt_cache_enabled)

        filter_cache_control_injection_points(model_kwargs)

        if expected_cache_control_injection_points is None:
            assert len(model_kwargs) > 0
        else:
            assert (
                model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY]
                == expected_cache_control_injection_points
            )


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
        [ModelClassProvider.AMAZON_Q],
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
