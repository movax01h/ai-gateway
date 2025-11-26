from unittest.mock import Mock

import pytest

from ai_gateway.prompts.registry import LocalPromptRegistry


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    """Fixture for mock prompt registry."""
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_prompt = Mock()
    mock_prompt.model = Mock()
    mock_prompt.model.model_name = "claude-3-sonnet"
    mock_registry.get_on_behalf.return_value = mock_prompt
    return mock_registry
