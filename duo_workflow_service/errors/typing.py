from langchain_core.tools import ToolException

GENERIC_WORKFLOW_ERROR_MESSAGE = (
    "There was an error processing your request in the"
    " Duo Agent Platform, please contact support if"
    " the issue persists."
)


class NotifiableException(Exception):
    pass


class TierAccessDeniedException(ToolException):
    """User's GitLab tier is insufficient for the requested feature."""

    def __init__(
        self,
        required_plan: str | None,
        feature: str,
        message: str | None = None,
    ):
        self.required_plan = required_plan
        self.feature = feature
        plan_display = required_plan.capitalize() if required_plan else "higher"
        super().__init__(
            message or f"This feature requires a {plan_display} GitLab subscription."
        )


class InvalidWorkflowIdException(Exception):
    """Raised when a workflow ID is invalid or not found."""
