# pylint: disable=file-naming-for-tests
"""Tests for the model's title and impact in the BL SAST report."""

import asyncio

import pytest

from duo_workflow_service.tools.bl_collect_and_flatten import BlCollectAndFlatten
from tests.duo_workflow_service.tools.test_bl_write_sast_report import (
    _dedup,
    _finding,
    _writer,
)


def test_the_model_title_and_impact_are_used_when_sane():
    finding = {
        "cwe": "CWE-639",
        "file": "routes/order.ts",
        "body": "Proof.",
        "title": "CWE-639: Checkout accepts another user\\'s basket on POST /rest/basket/:id/checkout.",
        "impact": "Any signed-in user can check out\n another user's basket.",
        "anchor_status": "verified",
    }
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["name"] == (
        "Checkout accepts another user's basket on POST /rest/basket/:id/checkout"
    )
    assert v["description"].startswith(
        "**What**\n\nAny signed-in user can check out another user's basket.\n\n"
    )


def test_the_impact_leads_the_description_when_there_is_no_body():
    finding = {
        "cwe": "CWE-639",
        "file": "a.ts",
        "impact": "Any user reads it.",
        "anchor_status": "verified",
    }
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["description"] == "**What**\n\nAny user reads it.\n\n**Where**\n\n`a.ts`"


def test_the_model_impact_is_escaped_as_markdown_text():
    finding = {
        "cwe": "CWE-639",
        "file": "a.ts",
        "body": "Proof.",
        "impact": "# @admin can read <token>",
        "anchor_status": "verified",
    }
    (v,) = _writer()._build_report([finding])["vulnerabilities"]

    assert v["description"].startswith(
        "**What**\n\n\\# `@admin` can read \\<token>\n\n"
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
def test_the_text_is_unchanged_when_the_model_text_is_not_sane(title, impact):
    base = {"cwe": "CWE-862", "file": "a.ts", "body": "No check here. More."}
    (want,) = _writer()._build_report([base])["vulnerabilities"]
    (v,) = _writer()._build_report([base | {"title": title, "impact": impact}])[
        "vulnerabilities"
    ]

    assert v["name"] == want["name"]
    assert v["description"] == want["description"]


def test_title_and_impact_do_not_change_identity():
    base = _finding()
    (want,) = _writer()._build_report([base])["vulnerabilities"]
    (v,) = _writer()._build_report(
        [base | {"title": "Webhook secret returned", "impact": "Any member."}]
    )["vulnerabilities"]

    assert (v["id"], v["identifiers"], v.get("tracking")) == (
        want["id"],
        want["identifiers"],
        want.get("tracking"),
    )


def test_title_and_impact_survive_dedup_triage_and_report():
    """The real stage chain after the scan: dedup, triage's verdict merge, dedup again, then the writer."""
    finding = _finding(
        title="Webhook secret returned to a non-admin caller",
        impact="Any project member can read the webhook secret.",
    )
    deduped = asyncio.run(_dedup()._execute([[finding]]))
    verdict = {
        "reasoning": "x" * 20,
        "verdict": "KEEP",
        "clause": "KEEP-1",
        "evidence": "a",
    }
    collected = asyncio.run(
        BlCollectAndFlatten(metadata=None)._execute(
            results=[{"final_answer": verdict}], verdict_items=deduped
        )
    )
    kept = asyncio.run(_dedup()._execute([collected["final_answer"]]))
    (v,) = _writer()._build_report(kept)["vulnerabilities"]

    assert v["name"] == "Webhook secret returned to a non-admin caller"
    assert v["description"].startswith(
        "**What**\n\nAny project member can read the webhook secret.\n\n"
    )
