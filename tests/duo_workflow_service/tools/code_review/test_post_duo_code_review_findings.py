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
    def test_finding_with_an_unrenderable_file_path_is_dropped_alone(self, path):
        published, suppressed = select_findings(
            [finding(file=path), finding(file="fine.rb")]
        )

        assert [f["file"] for f in published] == ["fine.rb"]
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


class TestRenderReviewOutput:
    @pytest.fixture(name="tool")
    def tool_fixture(self, metadata):
        return PostDuoCodeReviewFindings(metadata=metadata)

    def test_no_findings_renders_summary_only(self, tool):
        output = tool._render_review_output([], "All clean.")

        assert output == "<summary>\nAll clean.\n</summary>"
        assert "<review>" not in output

    def test_finding_renders_comment_with_severity_header(self, tool):
        output = tool._render_review_output([FINDING], "1 finding.")

        assert '<comment file="app/models/user.rb" new_line="42">' in output
        assert "**[Critical] fail-open**" in output
        assert "Fails open when the check errors." in output
        assert output.endswith("<comments_summary>\n1 finding.\n</comments_summary>")

    def test_old_line_attribute_is_rendered_only_when_present(self, tool):
        output = tool._render_review_output([finding(old_line=40)], "s")

        assert 'old_line="40" new_line="42"' in output

    def test_suggestion_needs_both_suggestion_and_target_code(self, tool):
        with_both = tool._render_review_output(
            [finding(suggestion="  return false")], "s"
        )
        without_target = tool._render_review_output(
            [finding(target_code="", suggestion="  return false")], "s"
        )

        assert "<from>\n  return true\n</from>" in with_both
        assert "<to>\n  return false\n</to>" in with_both
        assert "<from>" not in without_target

    def test_custom_instruction_ref_is_attributed(self, tool):
        output = tool._render_review_output(
            [finding(custom_instruction_ref="No fail-open guards")], "s"
        )

        assert (
            "According to custom instructions in 'No fail-open guards': "
            f"{FINDING['message']}" in output
        )
        # The monolith counts attributed comments with this regex, anchored per line.
        assert re.search(
            r"^According to custom instructions in .+?:", output, re.MULTILINE
        )

    def test_comment_tags_anchor_at_line_boundaries(self, tool):
        """The monolith parses comments with ^<comment ...>...</comment>$ regexes, so tags must own their lines."""
        output = tool._render_review_output([FINDING, FINDING], "s")

        for line in output.splitlines():
            if "<comment " in line:
                assert line.startswith("<comment ")
            if "</comment>" in line:
                assert line == "</comment>"

    def test_structural_tags_in_a_message_cannot_end_the_comment(self, tool):
        """A message quoting the review format must not truncate its own comment or open a code suggestion."""
        output = tool._render_review_output(
            [finding(message="Do not write </comment> or <from> here.")],
            "1 finding.",
        )

        assert "</comment>\nor" not in output
        assert "< /comment>" in output
        assert "< from>" in output
        assert output.count("</comment>") == 1

    def test_attributed_comment_opener_in_a_message_is_defused(self, tool):
        """A fake <comment file=...> opener in free text must not be able to impersonate a comment anchor."""
        output = tool._render_review_output(
            [finding(message='Beware <comment file="x.rb" new_line="1"> here.')],
            "1 finding.",
        )

        assert '< comment file="x.rb" new_line="1">' in output
        assert output.count("<comment ") == 1

    def test_review_and_summary_tags_in_free_text_are_defused(self, tool):
        """The monolith scans <review>...</review> and the summary blocks lazily, so a quoted closer would truncate
        them."""
        output = tool._render_review_output(
            [finding(message="Do not emit </review> or <comments_summary>.")],
            "Narrative quoting </comments_summary> and <review>.",
        )

        assert output.count("<review>") == 1
        assert output.count("</review>") == 1
        assert output.count("</comments_summary>") == 1
        assert "< /review>" in output
        assert "< review>" in output

    def test_summary_only_output_defuses_its_own_closer(self, tool):
        output = tool._render_review_output([], "Clean, but never write </summary>.")

        assert output.count("</summary>") == 1
        assert "< /summary>" in output

    @pytest.mark.parametrize(
        ("target_code", "suggestion"),
        [
            ("  x = 1", "  y = '</to>'"),
            ("  <summary>Old</summary>", "  <summary>New</summary>"),
            ("  <summary>Old</summary>", "  x = 1"),
        ],
    )
    def test_suggestion_whose_code_carries_a_structural_tag_is_withheld(
        self, tool, target_code, suggestion
    ):
        """Defusing code would publish a one-click patch containing the inserted space, so the message ships alone."""
        output = tool._render_review_output(
            [
                finding(
                    message="Fix this.", target_code=target_code, suggestion=suggestion
                )
            ],
            "1 finding.",
        )

        assert "<from>" not in output
        assert "<to>" not in output
        assert "< " not in output
        assert "Fix this." in output

    def test_suggestion_without_a_structural_tag_is_published_byte_identical(
        self, tool
    ):
        output = tool._render_review_output(
            [finding(target_code="  x = 1", suggestion="  y = a < b")],
            "1 finding.",
        )

        assert "<from>\n  x = 1\n</from>" in output
        assert "<to>\n  y = a < b\n</to>" in output


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
    body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert body["project_id"] == 123
    assert body["merge_request_iid"] == 45
    assert body["workflow_id"] == "test-workflow-123"
    review_output = body["review_output"]
    # Critical renders before minor, and the summary carries narrative plus computed counts.
    assert review_output.index('new_line="2"') < review_output.index('new_line="1"')
    assert "I focused on the auth changes." in review_output
    assert "2 findings (1 critical, 1 minor) across 1 file:" in review_output


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
    review_output = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])[
        "review_output"
    ]
    assert 'new_line="2"' in review_output
    assert 'new_line="1"' not in review_output
    assert "1 finding (1 critical) across 1 file:" in review_output


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_clean_review_posts_summary_only(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123, merge_request_iid=45, findings=[], summary="- Well tested"
    )

    body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert body["review_output"] == "<summary>\n- Well tested\n</summary>"


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
