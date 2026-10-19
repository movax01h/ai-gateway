"""Tests for the pure BL findings helpers shared by the dedup and report-writer tools."""

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.bl_security.findings import (
    audit_clause_of,
    coerce_to_list,
    cwe_digits,
    dedup_key,
    excerpt_of,
    findings_input,
    json_or_none,
    line_of,
    norm_excerpt,
    verdict_of,
)


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"line": "42"}, 42),
        ({"line": "not-a-number"}, 0),
        ({"new_line": 7}, 7),
        ({}, 0),
    ],
)
def test_line_of(finding, expected):
    assert line_of(finding) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ([1, 2], [1, 2]),
        ("[1, 2]", [1, 2]),
        ({"a": 1}, [{"a": 1}]),
    ],
)
def test_coerce_to_list(raw, expected):
    assert coerce_to_list(raw) == expected


@pytest.mark.parametrize("text", ["[1, 2] -- done", "```json\n[1, 2]\n```", "x"])
def test_coerce_to_list_reads_strict_json_only(text):
    # Every input was written by an earlier step with json.dumps, so text
    # around a JSON value is a defect, not prose to salvage the value from.
    assert coerce_to_list(text) == []


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"cwe": "CWE-639"}, "639"),
        ({"CWE": 862}, "862"),
        ({"cwe": "none"}, ""),
        ({}, ""),
    ],
)
def test_cwe_digits(finding, expected):
    assert cwe_digits(finding) == expected


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"code_excerpt": "a", "excerpt": "b"}, "a"),
        ({"excerpt": "b"}, "b"),
        ({}, ""),
    ],
)
def test_excerpt_of(finding, expected):
    assert excerpt_of(finding) == expected


def test_norm_excerpt_collapses_whitespace_and_case():
    assert norm_excerpt({"code_excerpt": "  Foo(\n\t X )  "}) == "foo( x )"


_KEYED = {"cwe": "CWE-639", "path": "a.py", "line": 9, "excerpt": "Get(Id)"}


def test_dedup_key_is_cwe_file_line_bucket_and_normalized_excerpt():
    assert dedup_key(_KEYED) == ("639", "a.py", 2, "get(id)")


def test_dedup_key_uses_a_precomputed_cwe():
    assert dedup_key(_KEYED, cwe="862")[0] == "862"


@pytest.mark.parametrize(
    ("change", "same_key"),
    [
        ({"line": 8}, True),  # same line bucket (8 // 4 == 9 // 4)
        ({"line": 12}, False),  # next line bucket
        ({"excerpt": "other"}, False),  # same bucket, different statement
    ],
)
def test_dedup_key_buckets_lines_and_keeps_the_excerpt(change, same_key):
    assert (dedup_key({**_KEYED, **change}) == dedup_key(_KEYED)) is same_key


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("not json", None),
    ],
)
def test_json_or_none(text, expected):
    assert json_or_none(text) == expected


@pytest.mark.parametrize("raw", [[1], "[1]", " [1] "])
def test_findings_input_accepts_a_list(raw):
    assert findings_input(raw, "findings") == [1]


@pytest.mark.parametrize("raw", [None, "Error: x [0]", "{}", {}, 42])
def test_findings_input_refuses_anything_else(raw):
    with pytest.raises(ToolException, match="`findings` is not a findings list"):
        findings_input(raw, "findings")


def test_findings_input_redacts_secrets_in_the_error():
    token = "glpat-abcdefghijklmnopqrst12"
    with pytest.raises(ToolException) as exc:
        findings_input(f"Error: failed reading key = '{token}'", "findings")
    assert token not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"verdict": " drop "}, "DROP"),
        ({}, ""),
    ],
)
def test_verdict_of(finding, expected):
    assert verdict_of(finding) == expected


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"clause": " KEEP-x "}, "KEEP-x"),
        ({}, ""),
    ],
)
def test_audit_clause_of(finding, expected):
    assert audit_clause_of(finding) == expected
