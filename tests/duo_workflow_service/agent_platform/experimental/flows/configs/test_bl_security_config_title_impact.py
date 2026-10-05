# pylint: disable=file-naming-for-tests
"""Tests for the customer-facing title and impact the bl_security scan prompts ask for."""

import pytest

from ai_gateway.response_schemas.converter import json_schema_to_pydantic
from tests.duo_workflow_service.agent_platform.experimental.flows.configs.test_bl_security_config import (
    DETECT_PROMPT_ID,
    FINDING_KEYS,
    UNIT_SCHEMA_ID,
    _prompt_system,
    _schema,
)

SIBLING_PROMPT_ID = "security_scan_sibling"

# Customer-facing text the report writer prefers when present; optional, so a
# finding without them is still a valid answer.
OPTIONAL_FINDING_KEYS = ("title", "impact")


def _model():
    return json_schema_to_pydantic(
        _schema(UNIT_SCHEMA_ID), title_fallback=UNIT_SCHEMA_ID
    )


def test_title_and_impact_are_optional_strings_that_survive_the_answer_tool():
    """Undeclared keys are dropped by the generated answer model, so the two must be declared to reach the report;
    optional, so a finding without them is still accepted."""
    item = _schema(UNIT_SCHEMA_ID)["properties"]["findings"]["items"]
    for key in OPTIONAL_FINDING_KEYS:
        assert item["properties"][key]["type"] == "string"
        assert item["properties"][key]["description"]
        assert key not in item["required"]
    finding = dict.fromkeys(FINDING_KEYS, "x") | {"new_line": 3, "tier": 1}
    summary = "One handler, one finding."

    bare = _model()(summary=summary, findings=[finding]).to_output()
    assert bare["findings"][0]["title"] is None
    given = {"title": "Checkout accepts another user's basket", "impact": "Any."}
    out = _model()(summary=summary, findings=[finding | given]).to_output()
    assert out["findings"][0] | given == out["findings"][0]


@pytest.mark.parametrize("prompt_id", [DETECT_PROMPT_ID, SIBLING_PROMPT_ID])
def test_the_scan_prompts_ask_for_a_customer_facing_title_and_impact(prompt_id):
    system = _prompt_system(prompt_id)
    assert '"title": "<under 80 characters' in system
    assert '"impact": "<one plain sentence' in system
    assert "customer" in system
