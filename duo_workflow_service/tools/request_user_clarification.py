from typing import ClassVar, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RequestUserClarificationTool"]

TOOL_TITLE = "request_user_clarification_tool"


class RequestUserClarificationTool(BaseModel):
    """Tool for requesting user input with recommendations from the LLM judge Agent.

    This tool is used to structure communication with the user when clarification is needed,
    particularly in the disambiguation component.
    """

    message: str = Field(description="The main message to the user")
    recommendations: List[str] = Field(
        description="List of specific recommendations or clarifications needed from the user"
    )
    clarity_score: float = Field(
        description="Overall clarity score from the judge's assessment (0-5)",
        ge=0.0,
        le=5.0,
    )
    clarity_verdict: Literal["CLEAR", "NEEDS CLARIFICATION", "UNCLEAR"] = Field(
        description="Clarity Final Verdict from the judge's assessment [CLEAR/NEEDS CLARIFICATION/UNCLEAR]"
    )
    response: Optional[str] = Field(
        default=None, description="Optional message to a user to answer their questions"
    )

    tool_title: ClassVar[str] = TOOL_TITLE

    model_config = ConfigDict(title=TOOL_TITLE, frozen=True)
