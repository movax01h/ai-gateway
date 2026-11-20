"""Deterministic check of whether the review context was cut to fit the tool output limit.

A very large merge request is truncated before the reviewer sees it, and a clean review of a fraction of the diff reads
as a clean review of all of it. This step reads the context the reviewer was given and returns a small mapping, so the
publish step can say so without the large context passing through its logged tool call.
"""

import re
from typing import Any, Dict, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import (
    TRUNCATED_TOOL_OUTPUT_TAG,
    TRUNCATION_NOTICE_TAG,
    ToolTrustLevel,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

logger = structlog.stdlib.get_logger(__name__)

__all__ = ["DetectReviewContextTruncation", "DetectReviewContextTruncationInput"]

# The notice wraps the whole output, so it opens it, before any merge request text. Security scanning encodes the
# tag in untrusted output, the real notice included, so both forms count.
_NOTICE_OPENERS = (f"<{TRUNCATION_NOTICE_TAG}>", f"&lt;{TRUNCATION_NOTICE_TAG}&gt;")
# Written by `tool_output_manager._add_truncation_instruction`, ahead of the kept output.
_PERCENTAGE_SHOWN = re.compile(r"Percentage shown: ([\d.]+)%")


class DetectReviewContextTruncationInput(BaseModel):
    """Input schema for checking whether the review context was truncated."""

    review_context: Optional[Any] = Field(
        default=None,
        description="The `build_review_merge_request_context` output the reviewer was given.",
    )


class DetectReviewContextTruncation(DuoBaseTool):
    """Report whether the review context was truncated, and how much of it was shown.

    Nothing is posted and nothing is read: the tool touches neither the GitLab API nor the executor.
    """

    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    name: str = "detect_review_context_truncation"
    description: str = (
        "Check whether the merge request context a code review was given was "
        "truncated, and what percentage of it was shown. Returns structured data."
    )
    args_schema: Type[BaseModel] = DetectReviewContextTruncationInput

    async def _execute(
        self, review_context: Optional[Any] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        text = review_context.lstrip() if isinstance(review_context, str) else ""
        truncated = text.startswith(_NOTICE_OPENERS)
        header = text.partition(TRUNCATED_TOOL_OUTPUT_TAG)[0]
        match = _PERCENTAGE_SHOWN.search(header) if truncated else None
        percent_shown = float(match.group(1)) if match else None

        logger.info(
            "Checked review context truncation",
            truncated=truncated,
            percent_shown=percent_shown,
        )

        return {"truncated": truncated, "percent_shown": percent_shown}

    def format_display_message(
        self, args: DetectReviewContextTruncationInput, tool_response: Any = None
    ) -> str:
        return "Check whether the merge request was too large to review in full"
