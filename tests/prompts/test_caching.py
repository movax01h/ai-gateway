import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.prompts.caching import CacheControlInjectionPointsConverter
from ai_gateway.prompts.config.models import ModelClassProvider


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

        response = converter.invoke(prompt_value)

        first_message = response.to_messages()[0]

        assert isinstance(first_message.content, list)
        assert isinstance(first_message.content[-1], dict)
        assert first_message.content[-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

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
