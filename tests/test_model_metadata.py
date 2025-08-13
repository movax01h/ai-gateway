# Import your model classes
from unittest import mock
from unittest.mock import patch

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import HttpUrl

from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    create_model_metadata,
)
from ai_gateway.model_selection import (
    LLMDefinition,
    ModelSelectionConfig,
    UnitPrimitiveConfig,
)


@pytest.fixture(autouse=True)
def get_llm_definitions():
    mock_models = {
        "gitlab_model1": LLMDefinition(
            gitlab_identifier="gitlab_model1",
            name="gitlab_model",
            family=["mixtral"],
            params={
                "model_class_provider": "provider",
                "model": "model_family",
            },
        ),
        "amazon_q": LLMDefinition(
            gitlab_identifier="amazon_q",
            name="amazon_q",
            family=["amazon_q"],
            params={"model": "amazon_q"},
        ),
    }

    mock_definitions = {
        "duo_chat": UnitPrimitiveConfig(
            feature_setting="duo_chat",
            unit_primitives=[GitLabUnitPrimitive.DUO_CHAT],
            default_model="gitlab_model1",
        )
    }

    with patch.multiple(
        ModelSelectionConfig,
        get_llm_definitions=mock.Mock(return_value=mock_models),
        get_unit_primitive_config_map=mock.Mock(return_value=mock_definitions),
    ) as mock_method:
        yield mock_method


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
        "name": "gitlab_model1",
        "provider": "openai",
        "endpoint": "https://api.openai.com/v1",
        "api_key": "test-key",
        "identifier": "openai/gpt-4",
    }

    # Act
    result = create_model_metadata(data)

    # Assert
    assert isinstance(result, ModelMetadata)
    assert result.name == "gitlab_model1"
    assert result.provider == "openai"
    assert str(result.endpoint) == "https://api.openai.com/v1"
    assert result.api_key == "test-key"
    assert result.identifier == "openai/gpt-4"


class TestCreateModelMetadata:
    def test_create_gitlab_model_metadata_with_identifier(self):
        data = {
            "provider": "gitlab",
            "identifier": "gitlab_model1",
        }

        result = create_model_metadata(data)

        assert result.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

    def test_create_gitlab_model_metadata_with_feature_setting(self):
        data = {
            "provider": "gitlab",
            "feature_setting": "duo_chat",
        }

        result = create_model_metadata(data)

        assert result.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

    def test_required_parameters(self):
        data = {
            "provider": "gitlab",
        }

        with pytest.raises(
            ValueError,
            match=r"Argument error: either identifier or feature_setting must be present.",
        ):
            create_model_metadata(data)

    def test_create_gitlab_model_metadata_non_existing(self):
        data = {
            "provider": "gitlab",
            "identifier": "non_existing_gitlab_model",
        }

        with pytest.raises(ValueError):
            create_model_metadata(data)


def test_create_model_metadata_invalid_data():
    # Arrange
    invalid_data = {
        "provider": "amazon_q",
        "name": "amazon_q",
        # missing required role_arn
    }

    with pytest.raises(ValueError):
        create_model_metadata(invalid_data)


class TestModelMetadataToParams:
    def test_without_identifier(self):
        model_metadata = create_model_metadata(
            {
                "name": "gitlab_model1",
                "provider": "provider",
                "endpoint": HttpUrl("https://api.example.com"),
                "api_key": "abcde",
                "identifier": None,
            }
        )

        assert model_metadata.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
        }

    def test_with_identifier_no_provider(self):
        model_metadata = create_model_metadata(
            {
                "name": "gitlab_model1",
                "provider": "provider",
                "endpoint": HttpUrl("https://api.example.com"),
                "api_key": "abcde",
                "identifier": "model_identifier",
            }
        )

        assert model_metadata.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_identifier",
            "custom_llm_provider": "custom_openai",
        }

    def test_with_identifier_with_provider(self):
        model_metadata = create_model_metadata(
            {
                "name": "gitlab_model1",
                "provider": "provider",
                "endpoint": HttpUrl("https://api.example.com"),
                "api_key": "abcde",
                "identifier": "custom_provider/model/identifier",
            }
        )

        assert model_metadata.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model/identifier",
            "custom_llm_provider": "custom_provider",
        }

    def test_with_identifier_with_bedrock_provider(self):
        model_metadata = create_model_metadata(
            {
                "name": "gitlab_model1",
                "provider": "provider",
                "endpoint": HttpUrl("https://api.example.com"),
                "api_key": "abcde",
                "identifier": "bedrock/model/identifier",
            }
        )

        assert model_metadata.llm_definition_params == {
            "model_class_provider": "provider",
            "model": "model_family",
        }

        assert model_metadata.to_params() == {
            "model": "model/identifier",
            "api_key": "abcde",
            "custom_llm_provider": "bedrock",
        }


def test_create_model_metadata_with_none_data():
    result = create_model_metadata(None)
    assert result is None


def test_create_model_metadata_without_provider():
    result = create_model_metadata({"name": "test"})
    assert result is None
