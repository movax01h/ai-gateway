from unittest.mock import Mock

import pytest

from ai_gateway.prompts.registry import LocalPromptRegistry


@pytest.fixture(name="mock_local_prompt_registry")
def mock_local_prompt_registry_fixture(prompt):
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_registry.get_on_behalf.return_value = prompt
    return mock_registry
