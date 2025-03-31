# Import your model classes
import pytest

from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    create_model_metadata,
)


def test_create_amazon_q_model_metadata():
    # Arrange
    data = {
        "provider": "amazon_q",
        "name": "amazon_q",
        "role_arn": "arn:aws:iam::123456789012:role/example-role",
    }

    # Act
    result = create_model_metadata(data)

    # Assert
    assert isinstance(result, AmazonQModelMetadata)
    assert result.provider == "amazon_q"
    assert result.name == "amazon_q"
    assert result.role_arn == "arn:aws:iam::123456789012:role/example-role"


def test_create_regular_model_metadata():
    # Arrange
    data = {
        "name": "gpt-4",
        "provider": "openai",
        "endpoint": "https://api.openai.com/v1",
        "api_key": "test-key",
        "identifier": "openai/gpt-4",
    }

    # Act
    result = create_model_metadata(data)

    # Assert
    assert isinstance(result, ModelMetadata)
    assert result.name == "gpt-4"
    assert result.provider == "openai"
    assert str(result.endpoint) == "https://api.openai.com/v1"
    assert result.api_key == "test-key"
    assert result.identifier == "openai/gpt-4"


def test_create_model_metadata_invalid_data():
    # Arrange
    invalid_data = {
        "provider": "amazon_q",
        "name": "amazon_q",
        # missing required role_arn
    }

    # Act & Assert
    with pytest.raises(ValueError):
        create_model_metadata(invalid_data)
