# pylint: disable=file-naming-for-tests
"""Tests that BL pipeline internals stay out of the SAST report and an unmatched line is said on the finding."""

import json

import duo_workflow_service.tools.bl_write_sast_report as bl
from tests.duo_workflow_service.tools.test_bl_write_sast_report import _audit, _writer


def test_no_anchor_state_reaches_the_report():
    report = _writer()._build_report(
        [
            {
                "file": "a.go",
                "new_line": 3,
                "anchor_status": bl.ANCHOR_CORRECTED,
                "anchor_claimed_line": 4,
                "claimed_file": "x/a.go",
            }
        ],
    )

    assert "details" not in report["vulnerabilities"][0]
    assert "x/a.go" not in json.dumps(report)


def test_an_unverified_line_is_stated_in_that_findings_description():
    report = _writer()._build_report(
        [
            {
                "file": "a.go",
                "new_line": 3,
                "body": "Leaks the token.",
                "anchor_status": bl.ANCHOR_VERIFIED,
            },
            {
                "file": "b.go",
                "new_line": 4,
                "body": "Leaks the key.",
                "anchor_status": bl.ANCHOR_CORRECTED,
            },
            {
                "file": "c.go",
                "new_line": 5,
                "body": "Leaks the id.",
                "anchor_status": bl.ANCHOR_UNVERIFIED,
            },
            {"file": "d.go", "new_line": 6, "anchor_status": bl.ANCHOR_UNVERIFIED},
        ],
    )
    descriptions = [v["description"] for v in report["vulnerabilities"]]

    assert [bl.UNVERIFIED_LINE_NOTE in d for d in descriptions] == [
        False,
        False,
        True,
        True,
    ]
    # The last paragraph, after the finding's own text.
    assert descriptions[2].endswith(f"Leaks the id.\n\n{bl.UNVERIFIED_LINE_NOTE}")
    assert descriptions[3].endswith(bl.UNVERIFIED_LINE_NOTE)
    # The plain sentence, never the internal state name.
    assert bl.ANCHOR_UNVERIFIED not in json.dumps(report["vulnerabilities"])


def test_a_finding_the_check_never_saw_also_carries_the_note():
    report = _writer()._build_report([{"file": "a.go", "new_line": 3, "body": "x"}])

    assert report["vulnerabilities"][0]["description"].endswith(bl.UNVERIFIED_LINE_NOTE)


def test_a_secret_in_the_triage_clause_is_redacted_in_the_audit_log():
    token = "glpat-abcdefghijklmnopqrst12"
    (audit,) = _audit(
        [{"file": "a.go", "new_line": 3, "verdict": "KEEP", "clause": f"KEEP {token}"}]
    )
    clause = audit[bl.TRIAGE_CLAUSE_DETAIL_KEY]

    assert token not in clause
    assert "[REDACTED]" in clause
