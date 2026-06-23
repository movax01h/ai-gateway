from langchain_core.tools import ToolException

GENERIC_WORKFLOW_ERROR_MESSAGE = (
    "There was an error processing your request in the"
    " Duo Agent Platform, please contact support if"
    " the issue persists."
)


class NotifiableException(Exception):
    pass


class NamespaceLevelWorkflowNotSupportedException(NotifiableException):
    """Raised when a workflow doesn't support namespace-level execution."""

    def __init__(self):
        super().__init__(
            "This feature is only available at the project level. Please try again from within a specific project."
        )


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


class EnvelopeVersionMismatchException(Exception):
    """Raised when an additional context envelope version does not satisfy the flow's constraint.

    This exception is raised during flow initialisation when the version field
    carried by an incoming envelope does not satisfy the semver constraint declared
    in the flow YAML configuration.  It is mapped to a ``FAILED_PRECONDITION``
    gRPC status so that callers receive a clear, actionable error rather than a
    generic ``INTERNAL`` failure.
    """
