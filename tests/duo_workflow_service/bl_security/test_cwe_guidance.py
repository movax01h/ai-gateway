"""Tests for the fixed per-CWE text on BL findings."""

import pytest

from duo_workflow_service.bl_security.cwe_guidance import (
    OWASP_2021_OF_CWE,
    OWASP_2025_OF_CWE,
    SOLUTIONS,
    owasp_identifiers,
)
from duo_workflow_service.tools.bl_report import IN_SCOPE_CWES


def test_every_in_scope_cwe_has_a_solution_and_nothing_else_does():
    assert set(SOLUTIONS) == set(IN_SCOPE_CWES)


@pytest.mark.parametrize("cwe", sorted(IN_SCOPE_CWES))
def test_each_solution_is_short_plain_text(cwe):
    text = SOLUTIONS[cwe]
    sentences = [s for s in text.split(". ") if s]
    assert 1 <= len(sentences) <= 3
    assert len(text) <= 300
    assert "\n" not in text
    assert text.endswith(".")


@pytest.mark.parametrize(
    ("of_cwe", "unmapped"),
    [
        (OWASP_2021_OF_CWE, {"362", "367", "459"}),
        (OWASP_2025_OF_CWE, {"459", "840"}),
    ],
)
def test_every_in_scope_cwe_is_mapped_or_explicitly_unmapped(of_cwe, unmapped):
    assert set(of_cwe) | unmapped == set(IN_SCOPE_CWES)
    assert not set(of_cwe) & unmapped


def test_the_owasp_identifiers_are_schema_shaped():
    assert owasp_identifiers("639") == [
        {
            "type": "owasp",
            "name": "A01:2021 - Broken Access Control",
            "value": "A01:2021",
            "url": "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
        },
        {
            "type": "owasp",
            "name": "A01:2025 - Broken Access Control",
            "value": "A01:2025",
            "url": "https://owasp.org/Top10/2025/A01_2025-Broken_Access_Control/",
        },
    ]
    assert [i["value"] for i in owasp_identifiers("287")] == ["A07:2021", "A07:2025"]
    assert [i["value"] for i in owasp_identifiers("362")] == ["A06:2025"]


@pytest.mark.parametrize("cwe", ["459", "79", ""])
def test_a_cwe_outside_the_top_10_gets_no_owasp_identifier(cwe):
    assert owasp_identifiers(cwe) == []
