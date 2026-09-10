import json
import re
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.code_review.post_duo_code_review_findings import (
    PostDuoCodeReviewFindings,
    PostDuoCodeReviewFindingsInput,
    build_summary,
    render_previous_findings,
    select_findings,
)

FINDING = {
    "file": "app/models/user.rb",
    "new_line": 42,
    "target_code": "  return true",
    "severity": "critical",
    "category": "fail-open",
    "message": "Fails open when the check errors.",
    "confidence": 9,
}


def finding(**overrides):
    return {**FINDING, **overrides}


def success_response():
    return GitLabHttpResponse(
        status_code=200, body=json.dumps({"message": "Comments added successfully"})
    )


def posted_review(gitlab_client_mock):
    body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    return body, json.loads(body["review_output"])


class TestSelectFindings:
    def test_zero_threshold_publishes_everything(self):
        published, suppressed = select_findings(
            [finding(confidence=0), finding(new_line=50)], min_confidence=0
        )

        assert len(published) == 2
        assert suppressed == 0

    def test_suppresses_below_the_threshold_and_counts_them(self):
        published, suppressed = select_findings(
            [
                finding(new_line=1, confidence=3),
                finding(new_line=2, confidence=7),
                finding(new_line=3, confidence=10),
            ],
            min_confidence=7,
        )

        assert [f["new_line"] for f in published] == [2, 3]
        assert suppressed == 1

    def test_finding_without_a_confidence_is_never_suppressed(self):
        unscored = {k: v for k, v in FINDING.items() if k != "confidence"}

        published, suppressed = select_findings([unscored], min_confidence=8)

        assert published == [unscored]
        assert suppressed == 0

    def test_orders_by_severity_and_keeps_reviewer_order_within_a_severity(self):
        published, _ = select_findings(
            [
                finding(new_line=1, severity="minor"),
                finding(new_line=2, severity="critical"),
                finding(new_line=3, severity="major"),
                finding(new_line=4, severity="critical"),
            ]
        )

        assert [(f["severity"], f["new_line"]) for f in published] == [
            ("critical", 2),
            ("critical", 4),
            ("major", 3),
            ("minor", 1),
        ]

    def test_unknown_severity_sorts_last(self):
        published, _ = select_findings(
            [finding(new_line=1, severity="odd"), finding(new_line=2, severity="minor")]
        )

        assert [f["new_line"] for f in published] == [2, 1]

    def test_findings_pass_through_verbatim(self):
        original = finding(suggestion="  return false", old_line=40)

        published, _ = select_findings([original])

        assert published[0] is original

    @pytest.mark.parametrize("path", ['we"ird.rb', "we>ird.rb", "we\nird.rb"])
    def test_any_file_path_is_publishable(self, path):
        """JSON carries any path, so nothing is dropped for the sake of the transport."""
        published, suppressed = select_findings([finding(file=path)])

        assert [f["file"] for f in published] == [path]
        assert suppressed == 0


class TestBuildSummary:
    def test_reports_counts_and_file_spread(self):
        summary = build_summary(
            [
                finding(new_line=1, severity="critical"),
                finding(file="b.rb", new_line=2, severity="minor"),
            ],
            None,
        )

        assert "2 findings" in summary
        assert "1 critical" in summary
        assert "1 minor" in summary
        assert "2 files" in summary

    def test_states_plainly_when_there_are_no_findings(self):
        assert build_summary([], None) == "No issues were raised in this review."

    def test_breaks_findings_down_by_severity(self):
        summary = build_summary(
            [
                finding(new_line=1, severity="critical", category="fail-open"),
                finding(new_line=2, severity="minor", category="missing-test"),
            ],
            None,
        )

        assert summary.index("**Critical**") < summary.index("**Minor**")
        assert f"- fail-open: `{FINDING['file']}:1`" in summary
        assert f"- missing-test: `{FINDING['file']}:2`" in summary
        assert "**Major**" not in summary

    def test_unknown_severity_gets_its_own_bucket(self):
        summary = build_summary([finding(severity="odd", category="x")], None)

        assert "**Other**" in summary
        assert f"- x: `{FINDING['file']}:42`" in summary

    def test_reviewer_narrative_leads_and_counts_stay_computed(self):
        summary = build_summary([finding()], "I focused on the auth changes.")

        assert summary.startswith("I focused on the auth changes.")
        assert "1 finding (1 critical) across 1 file:" in summary

    def test_reviewer_narrative_is_the_whole_summary_when_clean(self):
        summary = build_summary([], "- Well tested\n- Follows conventions")

        assert summary == "- Well tested\n- Follows conventions"

    def test_blank_narrative_falls_back_to_the_default(self):
        assert build_summary([], "   ") == "No issues were raised in this review."


def previous(status, file="app/models/user.rb", note="Nil check on `user`."):
    return {"file": file, "status": status, "note": note}


class TestRenderPreviousFindings:
    def test_nothing_to_render_without_items(self):
        assert render_previous_findings([]) is None

    @pytest.mark.parametrize(
        "statuses", [["fixed"], ["verified"], ["fixed", "verified"]]
    )
    def test_withheld_when_nothing_needs_attention(self, statuses):
        """A list of closed points reads as if there were something to act on, so the review is published as clean."""
        assert render_previous_findings([previous(s) for s in statuses]) is None

    def test_orders_closed_items_first_and_labels_each_status(self):
        rendered = render_previous_findings(
            [
                previous("still_outstanding", file="c.rb", note="Still nil."),
                previous("partially_fixed", file="b.rb", note="One branch left."),
                previous("fixed", file="a.rb", note="Guard added."),
                previous("verified", file="d.rb", note="Fix confirmed."),
            ]
        )

        assert rendered == (
            "- **Fixed:** `a.rb`: Guard added.\n"
            "- **Verified:** `d.rb`: Fix confirmed.\n"
            "- **Partially fixed:** `b.rb`: One branch left.\n"
            "- **Still outstanding:** `c.rb`: Still nil."
        )

    def test_unknown_status_sorts_last_and_is_shown_as_is(self):
        rendered = render_previous_findings(
            [previous("mystery"), previous("still_outstanding")]
        )

        assert rendered.splitlines()[-1].startswith("- **mystery:**")


class TestInputCoercion:
    @pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("7", 7), (5, 5)])
    def test_min_confidence_accepts_flow_config_strings(self, raw, expected):
        args = PostDuoCodeReviewFindingsInput(
            project_id=1, merge_request_iid=2, min_confidence=raw
        )

        assert args.min_confidence == expected

    def test_summary_may_be_absent(self):
        args = PostDuoCodeReviewFindingsInput(project_id=1, merge_request_iid=2)

        assert args.summary is None


class TestBuildPayload:
    @pytest.fixture(name="tool")
    def tool_fixture(self, metadata):
        return PostDuoCodeReviewFindings(metadata=metadata)

    def test_no_findings_sends_an_empty_list_with_the_summary(self, tool):
        assert tool._build_payload([], "All clean.") == {
            "findings": [],
            "summary": "All clean.",
        }

    def test_previous_findings_join_the_summary_when_there_are_comments(self, tool):
        payload = tool._build_payload(
            [FINDING], "1 finding.", "- **Still outstanding:** `a.rb`: x"
        )

        assert payload["summary"] == (
            "1 finding.\n\n**Previous findings**\n- **Still outstanding:** `a.rb`: x"
        )
        assert "previous_findings" not in payload

    def test_previous_findings_travel_alone_when_there_are_no_comments(self, tool):
        """The endpoint introduces the list as the outcome of a re-review, not as a clean first review."""
        payload = tool._build_payload(
            [], "- Looks fine", "- **Still outstanding:** `a.rb`: x"
        )

        assert payload == {
            "findings": [],
            "summary": "- Looks fine",
            "previous_findings": "- **Still outstanding:** `a.rb`: x",
        }

    def test_finding_carries_only_what_the_endpoint_anchors_and_renders(self, tool):
        payload = tool._build_payload([FINDING], "1 finding.")

        assert payload["summary"] == "1 finding."
        assert payload["findings"] == [
            {
                "file": "app/models/user.rb",
                "new_line": 42,
                "target_code": "  return true",
                "message": "**[Critical] fail-open**\n\nFails open when the check errors.",
                "severity": "critical",
                "confidence": 9,
            }
        ]

    def test_old_line_and_suggestion_are_sent_only_when_present(self, tool):
        [rendered] = tool._build_payload(
            [finding(old_line=40, suggestion="  return false")], "s"
        )["findings"]

        assert rendered["old_line"] == 40
        assert rendered["suggestion"] == "  return false"

    def test_end_line_is_sent_with_a_suggestion_that_spans_lines(self, tool):
        [rendered] = tool._build_payload(
            [finding(suggestion="  return false", end_line=44)], "s"
        )["findings"]

        assert rendered["end_line"] == 44

    @pytest.mark.parametrize(
        "overrides",
        [
            {"end_line": 44},  # no suggestion, so nothing to span
            {"suggestion": "  x", "end_line": 42},  # single line, the default
            {"suggestion": "  x", "end_line": 41},  # behind the anchor
            {"suggestion": "  x", "end_line": "44"},  # schema slip
        ],
    )
    def test_end_line_is_withheld_unless_it_widens_a_suggestion(self, tool, overrides):
        [rendered] = tool._build_payload([finding(**overrides)], "s")["findings"]

        assert "end_line" not in rendered

    def test_an_empty_suggestion_is_withheld(self, tool):
        """The monolith turns an empty replacement into a suggestion that deletes the line."""
        [rendered] = tool._build_payload([finding(suggestion="")], "s")["findings"]

        assert "suggestion" not in rendered

    def test_custom_instruction_ref_is_attributed(self, tool):
        [rendered] = tool._build_payload(
            [finding(custom_instruction_ref="No fail-open guards")], "s"
        )["findings"]

        assert (
            "According to custom instructions in 'No fail-open guards': "
            f"{FINDING['message']}" in rendered["message"]
        )
        # The monolith counts attributed comments with this regex, anchored per line.
        assert re.search(
            r"^According to custom instructions in .+?:",
            rendered["message"],
            re.MULTILINE,
        )

    def test_markup_in_messages_and_code_travels_verbatim(self, tool):
        """Nothing is escaped or defused: JSON has no structural tags to protect."""
        message = 'Do not write </comment> or <comment file="x.rb" new_line="1">.'
        [rendered] = tool._build_payload(
            [
                finding(
                    message=message,
                    target_code="  <summary>Old</summary>",
                    suggestion="  y = '</to>'",
                )
            ],
            "Quoting </comments_summary> and <review>.",
        )["findings"]

        assert rendered["message"].endswith(message)
        assert rendered["target_code"] == "  <summary>Old</summary>"
        assert rendered["suggestion"] == "  y = '</to>'"


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_success(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    # ainvoke, not _arun: the flow passes min_confidence as a string literal and the
    # args schema is what coerces it.
    response = await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [finding(new_line=1, severity="minor"), finding(new_line=2)],
            "summary": "I focused on the auth changes.",
            "min_confidence": "0",
        }
    )

    assert json.loads(response) == {
        "status": "success",
        "message": "Review posted to MR !45",
        "published": 2,
        "suppressed_below_threshold": 0,
    }
    body, review = posted_review(gitlab_client_mock)
    assert body["project_id"] == 123
    assert body["merge_request_iid"] == 45
    assert body["workflow_id"] == "test-workflow-123"
    # Critical publishes before minor, and the summary carries narrative plus computed counts.
    assert [f["new_line"] for f in review["findings"]] == [2, 1]
    assert review["summary"].startswith("I focused on the auth changes.")
    assert "2 findings (1 critical, 1 minor) across 1 file:" in review["summary"]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_applies_the_confidence_gate(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    response = await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [
                finding(new_line=1, confidence=2),
                finding(new_line=2, confidence=9),
            ],
            "summary": "",
            "min_confidence": "5",
        }
    )

    assert json.loads(response)["published"] == 1
    assert json.loads(response)["suppressed_below_threshold"] == 1
    _, review = posted_review(gitlab_client_mock)
    assert [f["new_line"] for f in review["findings"]] == [2]
    assert "1 finding (1 critical) across 1 file:" in review["summary"]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_clean_review_posts_summary_only(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123, merge_request_iid=45, findings=[], summary="- Well tested"
    )

    _, review = posted_review(gitlab_client_mock)
    assert review == {"findings": [], "summary": "- Well tested"}


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_accepts_a_null_previous_findings(
    gitlab_client_mock, metadata
):
    """A first review has no previous findings, and the optional flow input arrives as None."""
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [],
            "summary": "- Well tested",
            "previous_findings": None,
        }
    )

    _, review = posted_review(gitlab_client_mock)
    assert review == {"findings": [], "summary": "- Well tested"}


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_re_review_with_outstanding_threads(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123,
        merge_request_iid=45,
        findings=[],
        summary="- Nothing new",
        previous_findings=[
            previous("fixed"),
            previous("still_outstanding", file="b.rb"),
        ],
    )

    _, review = posted_review(gitlab_client_mock)
    assert review["findings"] == []
    assert review["previous_findings"].splitlines() == [
        "- **Fixed:** `app/models/user.rb`: Nil check on `user`.",
        "- **Still outstanding:** `b.rb`: Nil check on `user`.",
    ]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_failure_raises(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=422, body=json.dumps({"message": "Validation failed"})
        )
    )
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    with pytest.raises(ToolException, match="Failed to post review"):
        await tool._arun(
            project_id=123, merge_request_iid=45, findings=[FINDING], summary="s"
        )


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_non_json_response_raises(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=502, body="<html>Bad Gateway</html>"
        )
    )
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    with pytest.raises(ToolException, match="unreadable response"):
        await tool._arun(
            project_id=123, merge_request_iid=45, findings=[FINDING], summary="s"
        )


class TestDisplayMessage:
    def test_format_display_message(self, metadata):
        tool = PostDuoCodeReviewFindings(metadata=metadata)
        args = PostDuoCodeReviewFindingsInput(project_id=123, merge_request_iid=45)

        message = tool.format_display_message(args)

        assert "!45" in message
        assert "123" in message
