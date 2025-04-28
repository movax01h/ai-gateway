# Import your model classes
import pytest
from pydantic import HttpUrl

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


class TestModelMetadataToParams:
    def test_without_identifier(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier=None,
        )

        params = model_metadata.to_params()

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_family",
            "custom_llm_provider": "provider",
        }

    def test_with_identifier_no_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="model_identifier",
        )

        params = model_metadata.to_params()

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_identifier",
            "custom_llm_provider": "custom_openai",
        }

    def test_with_identifier_with_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="custom_provider/model/identifier",
        )

        params = model_metadata.to_params()

        assert params == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model/identifier",
            "custom_llm_provider": "custom_provider",
        }

    def test_with_identifier_with_bedrock_provider(self):
        model_metadata = ModelMetadata(
            name="model_family",
            provider="provider",
            endpoint=HttpUrl("https://api.example.com"),
            api_key="abcde",
            identifier="bedrock/model/identifier",
        )

        params = model_metadata.to_params()

        assert params == {
            "model": "model/identifier",
            "api_key": "abcde",
            "custom_llm_provider": "bedrock",
        }
