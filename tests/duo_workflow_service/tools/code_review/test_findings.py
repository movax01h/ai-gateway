import re

import pytest

from duo_workflow_service.tools.code_review.findings import (
    anchor_fields,
    attribute_message,
    build_review_payload,
    build_summary,
    render_posted_finding,
    render_previous_findings,
    render_structured_finding,
    select_findings,
    severity_counts,
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


class TestSeverityCounts:
    def test_counts_every_known_severity_even_when_absent(self):
        counts = severity_counts(
            [finding(severity="critical"), finding(severity="minor"), finding()]
        )

        assert counts == {"critical": 2, "major": 0, "minor": 1}

    def test_unknown_severities_are_not_counted(self):
        assert severity_counts([finding(severity="odd")]) == {
            "critical": 0,
            "major": 0,
            "minor": 0,
        }


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


class TestAttributeMessage:
    def test_message_is_untouched_without_a_custom_instruction(self):
        assert attribute_message(FINDING) == FINDING["message"]

    def test_custom_instruction_ref_is_attributed_in_the_monolith_wording(self):
        message = attribute_message(
            finding(custom_instruction_ref="No fail-open guards")
        )

        assert message == (
            "According to custom instructions in 'No fail-open guards': "
            f"{FINDING['message']}"
        )
        # The monolith counts attributed comments with this regex, anchored per line.
        assert re.search(
            r"^According to custom instructions in .+?:", message, re.MULTILINE
        )


class TestAnchorFields:
    def test_old_line_and_suggestion_are_carried_only_when_present(self):
        assert anchor_fields(FINDING) == {}
        assert anchor_fields(finding(old_line=40, suggestion="  return false")) == {
            "old_line": 40,
            "suggestion": "  return false",
        }

    def test_end_line_is_carried_with_a_suggestion_that_spans_lines(self):
        anchors = anchor_fields(finding(suggestion="  return false", end_line=44))

        assert anchors["end_line"] == 44

    @pytest.mark.parametrize(
        "overrides",
        [
            {"end_line": 44},  # no suggestion, so nothing to span
            {"suggestion": "  x", "end_line": 42},  # single line, the default
            {"suggestion": "  x", "end_line": 41},  # behind the anchor
            {"suggestion": "  x", "end_line": "44"},  # schema slip
        ],
    )
    def test_end_line_is_withheld_unless_it_widens_a_suggestion(self, overrides):
        assert "end_line" not in anchor_fields(finding(**overrides))

    def test_an_empty_suggestion_is_withheld(self):
        """The monolith turns an empty replacement into a suggestion that deletes the line."""
        assert "suggestion" not in anchor_fields(finding(suggestion=""))


class TestRenderLocalFinding:
    def test_schema_fields_travel_verbatim_without_a_rendered_header(self):
        rendered = render_structured_finding(finding(end_line=42))

        assert rendered == {
            "file": "app/models/user.rb",
            "new_line": 42,
            "target_code": "  return true",
            "severity": "critical",
            "category": "fail-open",
            "message": "Fails open when the check errors.",
            "confidence": 9,
        }
        assert "**[" not in rendered["message"]

    def test_attribution_is_rendered_into_the_message_and_the_ref_is_kept(self):
        rendered = render_structured_finding(
            finding(custom_instruction_ref="No fail-open guards")
        )

        assert rendered["message"] == (
            "According to custom instructions in 'No fail-open guards': "
            f"{FINDING['message']}"
        )
        assert rendered["custom_instruction_ref"] == "No fail-open guards"

    def test_anchor_rules_match_the_posting_path(self):
        source = finding(old_line=40, suggestion="  return false", end_line=44)

        local = render_structured_finding(source)
        posted = render_posted_finding(source)

        assert {k: local[k] for k in ("old_line", "suggestion", "end_line")} == {
            k: posted[k] for k in ("old_line", "suggestion", "end_line")
        }

    @pytest.mark.parametrize(
        "overrides",
        [{"suggestion": ""}, {"end_line": 44}, {"old_line": None}],
    )
    def test_withheld_anchors_are_dropped_rather_than_nulled(self, overrides):
        rendered = render_structured_finding(finding(**overrides))

        assert not set(overrides) & set(rendered)

    def test_the_source_finding_is_not_mutated(self):
        source = finding(custom_instruction_ref="Security", suggestion="")

        render_structured_finding(source)

        assert source == finding(custom_instruction_ref="Security", suggestion="")


class TestBuildPayload:
    def test_no_findings_sends_an_empty_list_with_the_summary(self):
        assert build_review_payload([], "All clean.") == {
            "findings": [],
            "summary": "All clean.",
        }

    def test_previous_findings_join_the_summary_when_there_are_comments(self):
        payload = build_review_payload(
            [FINDING], "1 finding.", "- **Still outstanding:** `a.rb`: x"
        )

        assert payload["summary"] == (
            "1 finding.\n\n**Previous findings**\n- **Still outstanding:** `a.rb`: x"
        )
        assert "previous_findings" not in payload

    def test_previous_findings_travel_alone_when_there_are_no_comments(self):
        """The endpoint introduces the list as the outcome of a re-review, not as a clean first review."""
        payload = build_review_payload(
            [], "- Looks fine", "- **Still outstanding:** `a.rb`: x"
        )

        assert payload == {
            "findings": [],
            "summary": "- Looks fine",
            "previous_findings": "- **Still outstanding:** `a.rb`: x",
        }

    def test_finding_carries_only_what_the_endpoint_anchors_and_renders(self):
        payload = build_review_payload([FINDING], "1 finding.")

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

    def test_old_line_and_suggestion_are_sent_only_when_present(self):
        [rendered] = build_review_payload(
            [finding(old_line=40, suggestion="  return false")], "s"
        )["findings"]

        assert rendered["old_line"] == 40
        assert rendered["suggestion"] == "  return false"

    def test_end_line_is_sent_with_a_suggestion_that_spans_lines(self):
        [rendered] = build_review_payload(
            [finding(suggestion="  return false", end_line=44)], "s"
        )["findings"]

        assert rendered["end_line"] == 44

    def test_an_empty_suggestion_is_withheld(self):
        [rendered] = build_review_payload([finding(suggestion="")], "s")["findings"]

        assert "suggestion" not in rendered

    def test_custom_instruction_ref_is_attributed(self):
        [rendered] = build_review_payload(
            [finding(custom_instruction_ref="No fail-open guards")], "s"
        )["findings"]

        assert (
            "According to custom instructions in 'No fail-open guards': "
            f"{FINDING['message']}" in rendered["message"]
        )
        assert re.search(
            r"^According to custom instructions in .+?:",
            rendered["message"],
            re.MULTILINE,
        )

    def test_markup_in_messages_and_code_travels_verbatim(self):
        """Nothing is escaped or defused: JSON has no structural tags to protect."""
        message = 'Do not write </comment> or <comment file="x.rb" new_line="1">.'
        [rendered] = build_review_payload(
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
