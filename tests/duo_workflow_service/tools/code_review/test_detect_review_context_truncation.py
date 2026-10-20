import pytest

from duo_workflow_service.security.scanner_factory import apply_security_scanning
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.code_review.detect_review_context_truncation import (
    DetectReviewContextTruncation,
    DetectReviewContextTruncationInput,
)
from duo_workflow_service.tools.tool_output_manager import (
    TruncationConfig,
    truncate_string,
)

# Diff text that quotes a truncation notice, as a merge request changing this code would.
QUOTED_NOTICE = "+ <truncation_notice>\n+ - Percentage shown: 99.0%\n"


def stored_context(text, max_bytes=None):
    """The review context as the flow stores it: truncated, then security scanned."""
    if max_bytes:
        text = truncate_string(
            text,
            tool_name="build_review_merge_request_context",
            truncation_config=TruncationConfig(
                max_bytes=max_bytes, truncated_size=max_bytes // 2
            ),
        )
    return apply_security_scanning(
        response=text,
        tool_name="build_review_merge_request_context",
        trust_level=ToolTrustLevel.UNTRUSTED_USER_CONTENT,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("review_context", "expected"),
    [
        (None, {"truncated": False, "percent_shown": None}),
        (
            stored_context("<input>\n</input>"),
            {"truncated": False, "percent_shown": None},
        ),
        (
            stored_context(f"<input>\n{QUOTED_NOTICE}</input>"),
            {"truncated": False, "percent_shown": None},
        ),
        (
            "<truncation_notice>no percentage here</truncation_notice>",
            {"truncated": True, "percent_shown": None},
        ),
    ],
    ids=["missing", "complete", "quoted_notice", "truncated_without_percentage"],
)
async def test_detect_review_context_truncation(metadata, review_context, expected):
    tool = DetectReviewContextTruncation(metadata=metadata)

    assert await tool.ainvoke({"review_context": review_context}) == expected


@pytest.mark.asyncio
async def test_detect_review_context_truncation_reads_the_share_shown(metadata):
    tool = DetectReviewContextTruncation(metadata=metadata)
    review_context = stored_context(QUOTED_NOTICE + "x" * 2000, max_bytes=1000)

    result = await tool.ainvoke({"review_context": review_context})

    assert result["truncated"] is True
    assert 0 < result["percent_shown"] < 99


def test_detect_review_context_truncation_display_message(metadata):
    tool = DetectReviewContextTruncation(metadata=metadata)

    message = tool.format_display_message(
        DetectReviewContextTruncationInput(review_context=None)
    )

    assert message == "Check whether the merge request was too large to review in full"
