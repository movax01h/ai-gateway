import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from duo_workflow_service.components.base import _process_agent_user_environment
from duo_workflow_service.workflows.type_definitions import (
    AdditionalContext,
    OsInformationContext,
)


class TestProcessAgentUserEnvironment:
    """Test cases for _process_agent_user_environment function."""

    @pytest.fixture
    def mock_os_information_context_data(self) -> Dict[str, Any]:
        """Valid data for OsInformationContext."""
        return {"platform": "Linux", "architecture": "x86_64"}

    @pytest.fixture
    def create_additional_context(self):
        """Factory fixture to create AdditionalContext objects."""

        def _create(category: str, content: Optional[str] = None) -> AdditionalContext:
            context = Mock(spec=AdditionalContext)
            context.category = category
            context.content = content
            return context

        return _create

    @pytest.mark.parametrize(
        "additional_contexts,expected_result",
        [
            # Test Case 1: None input
            (None, {}),
            # Test Case 2: Empty list
            ([], {}),
            # Test Case 3: Wrong category
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="wrong_category",
                        content='{"os_name": "Linux"}',
                    )
                ],
                {},
            ),
            # Test Case 4: Empty content
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content=None,
                    )
                ],
                {},
            ),
            # Test Case 5: Empty string content
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content="",
                    )
                ],
                {},
            ),
            # Test Case 6: Invalid JSON
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content="invalid json",
                    )
                ],
                {},
            ),
            # Test Case 7: Valid JSON but not a dict
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content='["not", "a", "dict"]',
                    )
                ],
                {},
            ),
            # Test Case 8: Valid JSON number
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content="42",
                    )
                ],
                {},
            ),
            # Test Case 9: Valid JSON string
            (
                [
                    Mock(
                        spec=AdditionalContext,
                        category="agent_user_environment",
                        content='"just a string"',
                    )
                ],
                {},
            ),
        ],
    )
    def test_invalid_inputs(self, additional_contexts, expected_result):
        """Test various invalid input scenarios."""
        result = _process_agent_user_environment(additional_contexts)
        assert result == expected_result

    @pytest.mark.parametrize(
        "content_data,should_validate",
        [
            # Valid OsInformationContext data
            ({"platform": "Linux", "architecture": "x86_64"}, True),
            # Missing required field
            ({"architecture": "x86_64"}, False),
            # Extra fields that don't match signature
            (
                {"platform": "Linux", "architecture": "x86_64", "extra_field": "value"},
                False,
            ),
            # Wrong field types
            ({"platform": 123, "architecture": "x86_64"}, False),
        ],
    )
    def test_validation_scenarios(
        self, create_additional_context, content_data, should_validate
    ):
        """Test various validation scenarios for OsInformationContext."""
        context = create_additional_context(
            category="agent_user_environment", content=json.dumps(content_data)
        )

        result = _process_agent_user_environment([context])

        if should_validate:
            assert "os_information_context" in result
            assert isinstance(result["os_information_context"], OsInformationContext)
        else:
            assert result == {}

    def test_multiple_contexts_same_type(
        self, create_additional_context, mock_os_information_context_data
    ):
        """Test multiple contexts of the same type - last one should win."""
        context1 = create_additional_context(
            category="agent_user_environment",
            content=json.dumps(
                {**mock_os_information_context_data, "platform": "Windows"}
            ),
        )
        context2 = create_additional_context(
            category="agent_user_environment",
            content=json.dumps(
                {**mock_os_information_context_data, "platform": "Linux"}
            ),
        )

        result = _process_agent_user_environment([context1, context2])

        assert "os_information_context" in result
        assert result["os_information_context"].platform == "Linux"

    def test_with_real_additional_context(self):
        """Test with actual AdditionalContext instances."""
        valid_os_data = {"platform": "Ubuntu", "architecture": "amd64"}

        context = AdditionalContext(
            category="agent_user_environment", content=json.dumps(valid_os_data)
        )

        result = _process_agent_user_environment([context])

        assert "os_information_context" in result
        assert isinstance(result["os_information_context"], OsInformationContext)
        assert result["os_information_context"].platform == "Ubuntu"
        assert result["os_information_context"].architecture == "amd64"
