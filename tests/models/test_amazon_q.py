from unittest.mock import MagicMock

import pytest

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.models.amazon_q import AmazonQModel, KindAmazonQModel
from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.safety_attributes import SafetyAttributes


def test_amazon_q_model_init():
    mock_user = MagicMock(spec=StarletteUser)
    mock_factory = MagicMock(spec=AmazonQClientFactory)

    model = AmazonQModel(mock_user, "test-role", mock_factory)

    assert model._current_user == mock_user
    assert model._role_arn == "test-role"
    assert model.metadata.name == KindAmazonQModel.AMAZON_Q
    assert model.metadata.engine == KindAmazonQModel.AMAZON_Q


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "expected_suffix"),
    [
        ("suffix", "suffix"),
        (None, ""),
    ],
)
async def test_amazon_q_model_generate(suffix, expected_suffix):
    mock_user = MagicMock(spec=StarletteUser)
    mock_factory = MagicMock(spec=AmazonQClientFactory)
    mock_client = MagicMock()
    mock_factory.get_client.return_value = mock_client

    model = AmazonQModel(mock_user, "test-role", mock_factory)

    mock_client.generate_code_recommendations.return_value = {
        "CodeRecommendations": [{"content": "Generated Code"}]
    }

    output = await model.generate("prefix", suffix, "file.py", "Python")

    assert isinstance(output, TextGenModelOutput)
    assert output.text == "Generated Code"
    assert output.score == 10**5
    assert isinstance(output.safety_attributes, SafetyAttributes)
    mock_client.generate_code_recommendations.assert_called_once_with(
        {
            "fileContext": {
                "leftFileContent": "prefix",
                "rightFileContent": expected_suffix,
                "filename": "file.py",
                "programmingLanguage": {"languageName": "Python"},
            },
            "maxResults": 1,
        }
    )
