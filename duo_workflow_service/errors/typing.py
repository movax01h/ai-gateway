from langchain_core.tools import ToolException


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
