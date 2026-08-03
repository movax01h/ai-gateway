# pylint: disable=file-naming-for-tests,unsubscriptable-object
"""Tests for the ASCP business context and security guidelines system prompt.

These assert on the content of
``business_context_security_guidelines/1.0.0.yml`` to lock in the structured
field contract (boolean / enum) and the removals introduced alongside it.
"""

import pytest

from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)

FLOW_NAME = "business_context_security_guidelines"


@pytest.fixture(scope="module", name="system_prompt")
def system_prompt_fixture() -> str:
    """The rendered system prompt template for the ASCP flow config."""
    config = FlowConfig.from_yaml_config(FLOW_NAME, None)
    prompts = config.prompts
    assert prompts is not None
    return prompts[0]["prompt_template"]["system"]


def _field_line(system_prompt: str, field: str) -> str:
    """Return the prompt line describing the given field."""
    marker = f"`{field}`:"
    for line in system_prompt.splitlines():
        if marker in line:
            return line.lower()
    raise AssertionError(f"No line describing `{field}` found in prompt")


class TestStructuredFields:
    """The three enrichment fields must describe their structured types.

    These assert on contract keywords rather than exact wording so the prompt can be reworded without breaking the test.
    """

    @pytest.mark.parametrize("field", ["authentication_model", "data_sensitivity"])
    def test_boolean_field_describes_string_values(self, system_prompt, field):
        line = _field_line(system_prompt, field)
        assert '"true"' in line
        assert '"false"' in line

    def test_authorization_model_is_enum(self, system_prompt):
        line = _field_line(system_prompt, "authorization_model")
        assert "elevated" in line
        assert "standard" in line


class TestRemovedContent:
    """Content removed in this update must not reappear."""

    @pytest.mark.parametrize(
        "removed",
        [
            "<output_format>",
            "</output_format>",
            "legitimate_use",
            "business_context",
            "Created ASCP scan",
            "Registered component:",
            "Created security context for",
            "Skip components that",
        ],
    )
    def test_removed_snippet_absent(self, system_prompt, removed):
        assert removed not in system_prompt
