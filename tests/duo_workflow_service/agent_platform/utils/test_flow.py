import pytest

from duo_workflow_service.agent_platform.utils.flow import (
    _MAX_WORKFLOW_DEFINITION_LENGTH,
    parse_deprecated_workflow_definition,
)


class TestParseDeprecatedWorkflowDefinition:
    def test_valid_v1(self):
        assert parse_deprecated_workflow_definition("developer/v1") == (
            "v1",
            "developer",
        )

    def test_valid_experimental(self):
        assert parse_deprecated_workflow_definition("developer/experimental") == (
            "experimental",
            "developer",
        )

    def test_single_segment_rejected(self):
        with pytest.raises(ValueError, match="Invalid workflow_definition format"):
            parse_deprecated_workflow_definition("developer")

    def test_three_segments_rejected(self):
        with pytest.raises(ValueError, match="Invalid workflow_definition format"):
            parse_deprecated_workflow_definition("developer/v1/1.0.0")

    def test_multi_segment_traversal_rejected(self):
        # Path traversal attempts with extra segments are caught by the 2-segment check.
        for value in ["../etc/v1", "./developer/v1", "developer/../v1"]:
            with pytest.raises(ValueError, match="Invalid workflow_definition format"):
                parse_deprecated_workflow_definition(value)

    def test_trailing_slash_rejected(self):
        with pytest.raises(ValueError, match="Invalid workflow_definition format"):
            parse_deprecated_workflow_definition("developer/v1/")

    def test_empty_flow_name_rejected(self):
        with pytest.raises(ValueError, match="Invalid workflow_definition format"):
            parse_deprecated_workflow_definition("/v1")

    def test_invalid_api_version(self):
        with pytest.raises(ValueError, match="Invalid API version"):
            parse_deprecated_workflow_definition("developer/v2")

    def test_invalid_api_version_stable(self):
        with pytest.raises(ValueError, match="Invalid API version"):
            parse_deprecated_workflow_definition("developer/stable")

    def test_length_cap_exceeded(self):
        long_value = "a" * (_MAX_WORKFLOW_DEFINITION_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            parse_deprecated_workflow_definition(long_value)

    def test_length_cap_at_limit(self):
        # Exactly at the limit with a valid definition should not raise.
        flow_name = "a" * (_MAX_WORKFLOW_DEFINITION_LENGTH - len("/v1"))
        value = f"{flow_name}/v1"
        assert len(value) == _MAX_WORKFLOW_DEFINITION_LENGTH
        assert parse_deprecated_workflow_definition(value) == ("v1", flow_name)
