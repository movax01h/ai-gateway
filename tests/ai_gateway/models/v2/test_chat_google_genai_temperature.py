# pylint: disable=file-naming-for-tests,protected-access
import pytest
from google.genai import Client

from ai_gateway.models.v2.chat_google_genai import ChatGoogleGenerativeAI

# Gemini 3.6 Flash onwards no longer accept the sampling parameters, and Google's
# guidance is to leave them out of the request rather than send a value. Gemini 3.1
# Pro Preview is included because its models.yml entry no longer sets one either.
MODELS_WITHOUT_SAMPLING_PARAMS = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]


class TestChatGoogleGenerativeAITemperature:
    @pytest.fixture(name="google_client")
    def google_client_fixture(self):
        return Client(api_key="test-key")

    def _build(self, google_client, model, **kwargs):
        return ChatGoogleGenerativeAI(
            model=model, google_api_key="dummy", client=google_client, **kwargs
        )

    @pytest.mark.parametrize("model", MODELS_WITHOUT_SAMPLING_PARAMS)
    def test_temperature_absent_from_request_when_not_configured(
        self, google_client, model
    ):
        """An unset temperature must not reach the request.

        LangChain would otherwise default it to 0.7, or rewrite it to 1.0 for Gemini 3 and later.
        """
        llm = self._build(google_client, model)

        assert llm.temperature is None

        params = llm._prepare_params(stop=None)

        assert "temperature" not in params.model_dump(exclude_none=True)

    @pytest.mark.parametrize("model", MODELS_WITHOUT_SAMPLING_PARAMS)
    def test_explicitly_configured_temperature_is_preserved(self, google_client, model):
        """models.yml stays authoritative when it does set a temperature."""
        llm = self._build(google_client, model, temperature=0.5)

        assert llm.temperature == 0.5
        assert params_temperature(llm) == 0.5

    def test_temperature_out_of_range_is_rejected(self, google_client):
        with pytest.raises(ValueError, match="temperature must be in the range"):
            self._build(google_client, "gemini-3.7-flash", temperature=2.5)


def params_temperature(llm):
    return llm._prepare_params(stop=None).model_dump(exclude_none=True)["temperature"]
