# Import your model classes
from unittest import mock
from unittest.mock import patch

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import HttpUrl

from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    FireworksModelMetadata,
    ModelMetadata,
    create_model_metadata,
)
from ai_gateway.model_selection import ModelSelectionConfig, UnitPrimitiveConfig
from ai_gateway.model_selection.model_selection_config import (
    ChatAmazonQDefinition,
    ChatAnthropicDefinition,
    ChatLiteLLMDefinition,
    ChatOpenAIDefinition,
)


@pytest.fixture(name="gitlab_model1")
def gitlab_model1_fixture():
    return ChatLiteLLMDefinition(
        gitlab_identifier="gitlab_model1",
        name="gitlab_model",
        max_context_tokens=200000,
        family=["mixtral"],
        params={
            "model": "model_family",
        },
        prompt_params={"timeout": 10},
    )


@pytest.fixture(name="gitlab_model2")
def gitlab_model2_fixture():
    return ChatAnthropicDefinition(
        gitlab_identifier="gitlab_model2",
        name="gitlab_model2",
        max_context_tokens=200000,
        params={
            "model": "model_family2",
        },
    )


@pytest.fixture(name="amazon_q_model")
def amazon_q_model_fixture():
    return ChatAmazonQDefinition(
        gitlab_identifier="amazon_q",
        name="amazon_q",
        max_context_tokens=200000,
        family=["amazon_q"],
        params={"model": "amazon_q"},
    )


@pytest.fixture(name="fireworks_model")
def fireworks_model_fixture():
    return ChatLiteLLMDefinition(
        gitlab_identifier="fireworks_ai",
        name="fireworks_ai",
        max_context_tokens=200000,
        family=["codestral"],
        params={"model": "test_model"},
    )


@pytest.fixture(autouse=True)
def get_llm_definitions(gitlab_model1, gitlab_model2, amazon_q_model, fireworks_model):
    mock_models = {
        "gitlab_model1": gitlab_model1,
        "gitlab_model2": gitlab_model2,
        "amazon_q": amazon_q_model,
        "test_model": fireworks_model,
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

    def test_with_identifier_with_vertex_ai_provider(self):
        model_metadata = create_model_metadata(
            {
                "name": "gitlab_model1",
                "provider": "provider",
                "endpoint": HttpUrl("https://api.example.com"),
                "api_key": "abcde",
                "identifier": "vertex_ai/model/identifier",
            }
        )

        assert model_metadata.to_params() == {
            "model": "model/identifier",
            "api_key": "abcde",
            "custom_llm_provider": "vertex_ai",
            "timeout": 10,
        }


def test_create_model_metadata_with_none_data():
    with pytest.raises(ValueError, match="provider must be present"):
        create_model_metadata(None)


def test_create_model_metadata_without_provider():
    with pytest.raises(ValueError, match="provider must be present"):
        create_model_metadata({"name": "test"})


class TestFireworksModelMetadata:
    """Test Fireworks model metadata creation and validation."""

    def test_create_fireworks_model_metadata_missing_identifier(self):
        """Test that missing model identifier raises ValueError."""
        data = {
            "provider": "fireworks_ai",
            "name": "test_model",
            "provider_keys": {"fireworks_api_key": "test-key"},
            "model_endpoints": {
                "fireworks_current_region_endpoint": {
                    "test_model": {
                        "endpoint": "https://api.fireworks.ai/inference/v1"
                        # Missing "identifier" key
                    }
                }
            },
        }

        with pytest.raises(
            ValueError,
            match=r"Fireworks model identifier is missing for model test_model\.",
        ):
            create_model_metadata(data)

    def test_create_fireworks_model_metadata_empty_string_identifier(self):
        """Test that empty string model identifier raises ValueError."""
        data = {
            "provider": "fireworks_ai",
            "name": "test_model",
            "provider_keys": {"fireworks_api_key": "test-key"},
            "model_endpoints": {
                "fireworks_current_region_endpoint": {
                    "test_model": {
                        "endpoint": "https://api.fireworks.ai/inference/v1",
                        "identifier": "",  # Empty string identifier
                    }
                }
            },
        }

        with pytest.raises(
            ValueError,
            match=r"Fireworks model identifier is missing for model test_model\.",
        ):
            create_model_metadata(data)

    def test_create_fireworks_model_metadata_none_identifier(self):
        """Test that None model identifier raises ValueError."""
        data = {
            "provider": "fireworks_ai",
            "name": "test_model",
            "provider_keys": {"fireworks_api_key": "test-key"},
            "model_endpoints": {
                "fireworks_current_region_endpoint": {
                    "test_model": {
                        "endpoint": "https://api.fireworks.ai/inference/v1",
                        "identifier": None,  # None identifier
                    }
                }
            },
        }

        with pytest.raises(
            ValueError,
            match=r"Fireworks model identifier is missing for model test_model\.",
        ):
            create_model_metadata(data)

    def test_create_fireworks_model_metadata_valid_identifier(self, fireworks_model):
        """Test that valid model identifier creates FireworksModelMetadata successfully."""
        data = {
            "provider": "fireworks_ai",
            "name": "test_model",
            "provider_keys": {"fireworks_api_key": "test-key"},
            "model_endpoints": {
                "fireworks_current_region_endpoint": {
                    "test_model": {
                        "endpoint": "https://api.fireworks.ai/inference/v1",
                        "identifier": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                    }
                }
            },
            "llm_definition": fireworks_model,
        }

        result = create_model_metadata(data)

        assert isinstance(result, FireworksModelMetadata)
        assert result.provider == "fireworks_ai"
        assert result.name == "test_model"
        assert (
            result.model_identifier
            == "accounts/fireworks/models/llama-v3p1-70b-instruct"
        )
        assert result.api_key == "test-key"
        assert str(result.endpoint) == "https://api.fireworks.ai/inference/v1"

    def test_create_fireworks_model_metadata_empty_identifier_with_mock_enabled(self):
        """Test that empty model identifier is allowed when mock_model_responses is True."""
        data = {
            "provider": "fireworks_ai",
            "name": "test_model",
            "provider_keys": {"fireworks_api_key": "test-key"},
            "model_endpoints": {
                "fireworks_current_region_endpoint": {
                    "test_model": {
                        "endpoint": "https://api.fireworks.ai/inference/v1",
                        "identifier": "",  # Empty string identifier
                    }
                }
            },
        }

        result = create_model_metadata(data, mock_model_responses=True)

        assert isinstance(result, FireworksModelMetadata)
        assert result.provider == "fireworks_ai"
        assert result.name == "test_model"
        assert result.model_identifier == ""
        assert result.api_key == "test-key"
        assert str(result.endpoint) == "https://api.fireworks.ai/inference/v1"

    def test_fireworks_to_params_with_all_fields(self, fireworks_model):
        """Test that to_params includes all fields when provided."""
        metadata = FireworksModelMetadata(
            provider="fireworks_ai",
            name="test_model",
            endpoint="https://api.fireworks.ai/v1",
            api_key="test_key",
            model_identifier="test_identifier",
            using_cache="True",
            session_id="test_session_id",
            llm_definition=fireworks_model,
            family=["codestral"],
        )
        params = metadata.to_params()
        assert params["model"] == "test_identifier"
        assert params["api_key"] == "test_key"
        assert params["api_base"] == "https://api.fireworks.ai/v1"
        assert params["using_cache"] == "True"
        assert params["session_id"] == "test_session_id"


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
            mock_definition = ChatLiteLLMDefinition(
                gitlab_identifier="gitlab_model1",
                name="GitLab Model One",  # This becomes friendly_name
                max_context_tokens=200000,
                family=["mixtral"],
                params={
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
            mock_definition = ChatAnthropicDefinition(
                gitlab_identifier="claude_sonnet_4_5_20250929",
                name="Claude Sonnet 4.5 - Anthropic",
                max_context_tokens=200000,
                family=["claude_4"],
                params={
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
            "provider": "litellm",
            "endpoint": "http://custom-endpoint.com/v1",
            "api_key": "custom-key",
            "identifier": "custom_openai/Llama-3.1-70B-Instruct",
        }

        # Mock the model selection config to simulate llama3 lookup
        with patch.object(ModelSelectionConfig, "get_model") as mock_get_model:
            mock_definition = ChatLiteLLMDefinition(
                gitlab_identifier="llama3",
                name="Llama3",
                max_context_tokens=200000,
                family=["llama3"],
                params={
                    "model": "llama3",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.name == "llama3"
            assert result.provider == "litellm"
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
            mock_definition = ChatOpenAIDefinition(
                gitlab_identifier="gpt_5",
                name="OpenAI GPT-5",
                max_context_tokens=200000,
                family=["gpt_5"],
                params={
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
            mock_definition = ChatAmazonQDefinition(
                gitlab_identifier="amazon_q",
                name="Amazon Q",  # This becomes friendly_name
                max_context_tokens=200000,
                family=["amazon_q"],
                params={
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
            mock_definition = ChatLiteLLMDefinition(
                gitlab_identifier="gitlab_model1",
                name="GitLab Identifier Model",
                max_context_tokens=200000,
                family=["mixtral"],
                params={
                    "model": "model_family",
                },
            )
            mock_get_model.return_value = mock_definition

            result = create_model_metadata(data)

            assert isinstance(result, ModelMetadata)
            assert result.friendly_name == "GitLab Identifier Model"
