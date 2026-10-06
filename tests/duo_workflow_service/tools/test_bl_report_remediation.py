# pylint: disable=file-naming-for-tests
"""Tests for the fix guidance and OWASP category on BL findings in the SAST report."""

import hashlib

import pytest

import duo_workflow_service.tools.bl_write_sast_report as bl
from duo_workflow_service.bl_security.cwe_guidance import SOLUTIONS
from tests.duo_workflow_service.tools.test_bl_write_sast_report import _writer


def test_the_cwe_stays_the_primary_identifier_and_owasp_follows():
    findings = [{"cwe": "CWE-639", "file": "a.ts", "body": "x", "code_excerpt": "y"}]
    (v,) = _writer()._build_report(findings)["vulnerabilities"]

    assert [i["value"] for i in v["identifiers"]] == ["639", "A01:2021", "A01:2025"]


def test_the_owasp_identifier_changes_neither_the_id_nor_tracking():
    finding = {
        "cwe": "CWE-639",
        "file": "a.ts",
        "new_line": 2,
        "code_excerpt": "find(id)",
        "anchor_status": bl.ANCHOR_VERIFIED,
    }
    sources = {"a.ts": "function getBasket(id) {\n  find(id)\n}\n"}
    (v,) = _writer()._build_report([finding], sources=sources)["vulnerabilities"]

    assert v["id"] == hashlib.sha256(b"a.ts|find(id)").hexdigest()
    assert v["tracking"]["items"][0]["signatures"] == [
        {"algorithm": "scope_offset", "value": "a.ts|getBasket[0]:639"}
    ]


def test_a_cwe_with_no_owasp_category_has_the_cwe_alone():
    (v,) = _writer()._build_report([{"cwe": "CWE-459", "file": "a.ts"}])[
        "vulnerabilities"
    ]

    assert [i["name"] for i in v["identifiers"]] == ["CWE-459"]


def test_the_solution_is_the_fixed_text_for_the_cwe():
    (v,) = _writer()._build_report([{"cwe": "CWE-639", "file": "a.ts"}])[
        "vulnerabilities"
    ]

    assert v["solution"] == SOLUTIONS["639"]


@pytest.mark.parametrize("cwe", ["", "CWE-79"])
def test_no_solution_when_the_cwe_has_none(cwe):
    (v,) = _writer()._build_report([{"cwe": cwe, "file": "a.ts"}])["vulnerabilities"]

    assert "solution" not in v
