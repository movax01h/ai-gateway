"""Tests for the fixed per-CWE text on BL findings."""

import pytest

from duo_workflow_service.bl_security.cwe_guidance import (
    OWASP_2021_OF_CWE,
    SOLUTIONS,
    owasp_identifier,
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


def test_owasp_categories_only_cover_in_scope_cwes():
    assert set(OWASP_2021_OF_CWE) <= set(IN_SCOPE_CWES)


def test_the_owasp_identifier_is_schema_shaped():
    assert owasp_identifier("639") == {
        "type": "owasp",
        "name": "A01:2021 - Broken Access Control",
        "value": "A01:2021",
        "url": "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
    }
    assert owasp_identifier("287")["value"] == "A07:2021"


@pytest.mark.parametrize("cwe", ["362", "367", "459", "79", ""])
def test_a_cwe_outside_the_top_10_gets_no_owasp_identifier(cwe):
    assert owasp_identifier(cwe) is None


@pytest.mark.parametrize(
    ("cwe", "category"),
    [
        ("200", "A01:2021"),
        ("284", "A01:2021"),
        ("285", "A01:2021"),
        ("639", "A01:2021"),
        ("862", "A01:2021"),
        ("863", "A01:2021"),
        ("840", "A04:2021"),
        ("287", "A07:2021"),
        ("915", "A08:2021"),
        ("362", None),
        ("367", None),
        ("459", None),
    ],
)
def test_each_in_scope_cwe_maps_to_its_owasp_category(cwe, category):
    identifier = owasp_identifier(cwe)

    assert (identifier or {}).get("value") == category


@pytest.mark.parametrize("cwe", ["639", "863"])
def test_the_access_control_fixes_do_not_assume_one_language_or_single_user_ownership(
    cwe,
):
    text = SOLUTIONS[cwe]

    assert "`" not in text
    assert "UserId" not in text
    assert "signed-in user" not in text
