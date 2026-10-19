import pytest

from ai_gateway.model_selection import (
    FeatureSettingRequiredError,
    ModelNotAllowedForFeatureError,
    ModelSelectionError,
    ProviderNotServedError,
)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            ModelNotAllowedForFeatureError("duo_chat", "claude_fable_5"),
            "Model 'claude_fable_5' is not available for 'duo_chat'",
        ),
        (
            FeatureSettingRequiredError("claude_fable_5"),
            "feature_setting is required to request model 'claude_fable_5'",
        ),
        (
            ProviderNotServedError("openai"),
            "This gateway does not serve models from provider 'openai'",
        ),
    ],
)
def test_error_message(error, message):
    assert str(error) == message


@pytest.mark.parametrize(
    "error",
    [
        ModelNotAllowedForFeatureError("duo_chat", "claude_fable_5"),
        FeatureSettingRequiredError("claude_fable_5"),
        ProviderNotServedError("openai"),
    ],
)
def test_is_a_model_selection_error_but_not_a_value_error(error):
    """Callers that catch ValueError fall back to a default model.

    A denial must not be swallowed that way.
    """
    assert isinstance(error, ModelSelectionError)
    assert not isinstance(error, ValueError)


def test_model_not_allowed_error_exposes_feature_and_model():
    error = ModelNotAllowedForFeatureError("duo_chat", "claude_fable_5")

    assert error.feature_setting == "duo_chat"
    assert error.gitlab_identifier == "claude_fable_5"
