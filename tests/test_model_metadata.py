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


@pytest.fixture(name="gitlab_model1")
def gitlab_model1_fixture():
    return LLMDefinition(
        gitlab_identifier="gitlab_model1",
        name="gitlab_model",
        max_context_tokens=200000,
        family=["mixtral"],
        params={
            "model_class_provider": "provider",
            "model": "model_family",
        },
        prompt_params={"timeout": 10},
    )


@pytest.fixture(name="gitlab_model2")
def gitlab_model2_fixture():
    return LLMDefinition(
        gitlab_identifier="gitlab_model2",
        name="gitlab_model2",
        max_context_tokens=200000,
        params={
            "model_class_provider": "provider2",
            "model": "model_family2",
        },
    )


@pytest.fixture(name="amazon_q_model")
def amazon_q_model_fixture():
    return LLMDefinition(
        gitlab_identifier="amazon_q",
        name="amazon_q",
        max_context_tokens=200000,
        family=["amazon_q"],
        params={"model": "amazon_q"},
    )


@pytest.fixture(autouse=True)
def get_llm_definitions(
    gitlab_model1, gitlab_model2, amazon_q_model
):  # pylint: disable=unused-argument
    mock_models = {
        "gitlab_model1": LLMDefinition(
            gitlab_identifier="gitlab_model1",
            name="gitlab_model",
            max_context_tokens=200000,
            family=["mixtral"],
            params={
                "model_class_provider": "provider",
                "model": "model_family",
            },
            prompt_params={"timeout": 10},
        ),
        "gitlab_model2": LLMDefinition(
            gitlab_identifier="gitlab_model2",
            name="gitlab_model2",
            max_context_tokens=200000,
            params={
                "model_class_provider": "provider2",
                "model": "model_family2",
            },
        ),
        "amazon_q": LLMDefinition(
            gitlab_identifier="amazon_q",
            name="amazon_q",
            max_context_tokens=200000,
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
    def test_create_gitlab_model_metadata_with_identifier(self, gitlab_model1):
        data = {
            "provider": "gitlab",
            "identifier": "gitlab_model1",
        }

        result = create_model_metadata(data)

        assert result.llm_definition == gitlab_model1

    def test_create_gitlab_model_metadata_with_feature_setting(self, gitlab_model1):
        data = {
            "provider": "gitlab",
            "feature_setting": "duo_chat",
        }

        result = create_model_metadata(data)

        assert result.llm_definition == gitlab_model1

    def test_create_gitlab_model_metadata_with_identifier_and_feature_setting(
        self, gitlab_model2
    ):
        data = {
            "provider": "gitlab",
            "identifier": "gitlab_model2",
            "feature_setting": "duo_chat",
        }

        result = create_model_metadata(data)

        assert result.llm_definition == gitlab_model2

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

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "timeout": 10,
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

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model_identifier",
            "custom_llm_provider": "custom_openai",
            "timeout": 10,
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

        assert model_metadata.to_params() == {
            "api_base": "https://api.example.com",
            "api_key": "abcde",
            "model": "model/identifier",
            "custom_llm_provider": "custom_provider",
            "timeout": 10,
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

        assert model_metadata.to_params() == {
            "model": "model/identifier",
            "api_key": "abcde",
            "custom_llm_provider": "bedrock",
            "timeout": 10,
        }


def test_create_model_metadata_with_none_data():
    result = create_model_metadata(None)
    assert result is None


def test_create_model_metadata_without_provider():
    result = create_model_metadata({"name": "test"})
    assert result is None


class TestFriendlyName:
    """Test friendly_name functionality in ModelMetadata."""

    def test_create_model_metadata_with_friendly_name_from_models_yml(self):
        """Test that create_model_metadata populates friendly_name from models.yml."""
        data = {
            "name": "gitlab_model1",
            "provider": "provider",
            "endpoint": "https://api.test.com/v1",
            "api_key": "test-key",
        }

        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="gitlab_model1",
                name="GitLab Model One",  # This becomes friendly_name
                max_context_tokens=200000,
                family=["mixtral"],
                params={
                    "model_class_provider": "provider",
                    "model": "model_family",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.friendly_name == "GitLab Model One"

    def test_create_model_metadata_with_real_model_friendly_name(self):
        """Test friendly_name with real data using mock."""
        data = {
            "name": "claude_sonnet_4_5_20250929",
            "provider": "gitlab",
            "identifier": "anthropic/claude-sonnet-4-5-20250929",
        }

        # Mock the model selection config to simulate real models.yml lookup
        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="claude_sonnet_4_5_20250929",
                name="Claude Sonnet 4.5 - Anthropic",
                max_context_tokens=200000,
                family=["claude_4"],
                params={
                    "model_class_provider": "anthropic",
                    "model": "claude-sonnet-4-5-20250929",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.name == "claude_sonnet_4_5_20250929"
            assert result.friendly_name == "Claude Sonnet 4.5 - Anthropic"

    def test_create_model_metadata_with_self_hosted_model_friendly_name(self):
        """Test friendly_name with self-hosted model using mocked models.yml definitions."""
        data = {
            "name": "llama3",  # Maps to gitlab_identifier in models.yml
            "provider": "custom_openai",
            "endpoint": "http://custom-endpoint.com/v1",
            "api_key": "custom-key",
            "identifier": "custom_openai/Llama-3.1-70B-Instruct",
        }

        # Mock the model selection config to simulate llama3 lookup
        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="llama3",
                name="Llama3",
                max_context_tokens=200000,
                family=["llama3"],
                params={
                    "model_class_provider": "custom_openai",
                    "model": "llama3",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.name == "llama3"
            assert result.provider == "custom_openai"
            assert result.friendly_name == "Llama3"

    def test_create_model_metadata_with_gpt_model_friendly_name(self):
        """Test friendly_name with GPT model using mock."""
        data = {
            "name": "gpt_5",
            "provider": "gitlab",
            "identifier": "openai/gpt-5-2025-08-07",
        }

        # Mock the model selection config to simulate gpt_5 lookup
        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="gpt_5",
                name="OpenAI GPT-5",
                max_context_tokens=200000,
                family=["gpt_5"],
                params={
                    "model_class_provider": "openai",
                    "model": "gpt-5-2025-08-07",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.name == "gpt_5"
            assert result.friendly_name == "OpenAI GPT-5"

    def test_create_amazon_q_model_metadata_with_friendly_name(self):
        """Test that create_model_metadata populates friendly_name for AmazonQ."""
        data = {
            "provider": "amazon_q",
            "name": "amazon_q",  # AmazonQ requires this literal value
            "role_arn": "arn:aws:iam::123456789012:role/AmazonQRole",
        }

        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="amazon_q",
                name="Amazon Q",  # This becomes friendly_name
                max_context_tokens=200000,
                family=["amazon_q"],
                params={
                    "model_class_provider": "amazon_q",
                    "model": "amazon_q",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, AmazonQModelMetadata)
            assert result.friendly_name == "Amazon Q"
            assert result.role_arn == "arn:aws:iam::123456789012:role/AmazonQRole"

    def test_model_metadata_friendly_name_optional(self, llm_definition):
        """Test that friendly_name can be None."""
        metadata = ModelMetadata(
            name="test_model",
            provider="test_provider",
            max_context_tokens=200000,
            friendly_name=None,
            llm_definition=llm_definition,
        )

        assert metadata.friendly_name is None

    def test_create_model_metadata_no_name_uses_identifier_for_friendly_name(self):
        """Test that when no name is provided, identifier lookup works for friendly_name."""
        data = {
            "provider": "provider",
            "identifier": "gitlab_model1",
        }

        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = LLMDefinition(
                gitlab_identifier="gitlab_model1",
                name="GitLab Identifier Model",
                max_context_tokens=200000,
                family=["mixtral"],
                params={
                    "model_class_provider": "provider",
                    "model": "model_family",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.friendly_name == "GitLab Identifier Model"
