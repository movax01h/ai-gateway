# pylint: disable=file-naming-for-tests
"""Tests for the fixed title and the What / Where / Details description of BL findings in the SAST report."""

import hashlib

import pytest

import duo_workflow_service.tools.bl_write_sast_report as bl
from duo_workflow_service.bl_security.finding_text import IMPACTS
from tests.duo_workflow_service.tools.test_bl_write_sast_report import _writer


def test_no_cwe_and_no_file_has_the_generic_name():
    report = _writer()._build_report([{}])
    assert report["vulnerabilities"][0]["name"] == "Business-logic finding"


def test_the_title_names_the_weakness_and_the_file():
    (v,) = _writer()._build_report(
        [{"cwe": "CWE-639", "file": "a.ts", "body": "Any user can refund any order."}]
    )["vulnerabilities"]

    assert v["name"] == "Authorization bypass through user-controlled key in a.ts"


def test_the_description_is_structured_and_keeps_the_model_text():
    finding = {
        "cwe": "CWE-639",
        "file": "a.ts",
        "new_line": 4,
        "code_excerpt": "find(id)",
        "body": "Any user can read any basket. The id is not scoped.",
        "anchor_status": bl.ANCHOR_VERIFIED,
    }
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["description"] == (
        f"**What**\n\n{IMPACTS['639']}\n\n"
        "**Where**\n\n`a.ts:4`\n\n```\nfind(id)\n```\n\n"
        "**Details**\n\n- Any user can read any basket.\n- The id is not scoped."
    )


def test_the_id_still_comes_from_the_raw_model_text():
    """The description is rebuilt, but identity is keyed on the excerpt or the raw body, as before."""
    finding = {"cwe": "CWE-639", "file": "a.ts", "body": "raw body"}
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["id"] == hashlib.sha256(b"a.ts|raw body").hexdigest()


def test_the_model_title_and_impact_lead_the_structured_text():
    finding = {
        "cwe": "CWE-639",
        "file": "routes/order.ts",
        "body": "Proof.",
        "title": "Checkout accepts another user's basket",
        "impact": "Any signed-in user can check out another user's basket.",
        "anchor_status": bl.ANCHOR_VERIFIED,
    }
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["name"] == "Checkout accepts another user's basket"
    assert v["description"] == (
        "**What**\n\nAny signed-in user can check out another user's basket.\n\n"
        "**Where**\n\n`routes/order.ts`\n\n"
        "**Details**\n\n- Proof."
    )


@pytest.mark.parametrize(
    "title,impact",
    [
        (None, None),
        ("", "   "),
        ("Two\nlines here", 42),
        ("Too short", "x" * 401),
        ("x" * 121, None),
    ],
    ids=[
        "missing",
        "blank",
        "multi-line-or-not-text",
        "too-short-or-long",
        "too-long",
    ],
)
def test_the_fixed_text_is_used_when_the_model_text_is_not_sane(title, impact):
    finding = {"cwe": "CWE-862", "file": "a.ts", "title": title, "impact": impact}
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["name"] == "Missing authorization in a.ts"
    assert v["description"].startswith(f"**What**\n\n{IMPACTS['862']}")
