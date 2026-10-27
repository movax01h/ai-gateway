from langchain_core.tools import ToolException

GENERIC_WORKFLOW_ERROR_MESSAGE = (
    "There was an error processing your request in the"
    " Duo Agent Platform, please contact support if"
    " the issue persists."
)


class NotifiableException(Exception):
    pass


class UnrecoverableWorkflowException(NotifiableException):
    """Raised when a session can never be executed again, whatever the client does.

    Unlike a generic failure this is a *rejection*: the session is already in a terminal-by-nature condition (e.g. its
    checkpoints have passed the retention window), so the run never starts. Callers must therefore track it as
    ``WORKFLOW_REJECT`` and must not drive a ``drop`` status event — there is no execution to fail.

    It is a ``NotifiableException`` because the message explains to the user what to do next and is safe to display
    verbatim.
    """


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


class WorkflowTokenRejectedError(Exception):
    """The token does not authorize the requested workflow, so the run is rejected before any work.

    Raised as the ``__cause__`` of a ``NotifiableException`` so the user sees a safe message while the server maps
    this class to ``PERMISSION_DENIED`` with ``details``. Unlike ``UnrecoverableWorkflowException`` the session is
    dropped: Rails still holds it as runnable, and a silent exit is what made the rejection look like a normal end.

    Attributes:
        details: The gRPC status detail, fixed text written by the server.
        drop_reason: The reason recorded on the Rails row when it is dropped.
        user_message: The safe text shown to the user, raised as the ``NotifiableException`` message. A subclass for
            a different rejection (e.g. a token carrying no workflow binding, which asks the user to update their
            client) overrides it while reusing the same reject-and-drop path.
    """

    drop_reason = "token_rejected"
    user_message = (
        "This session was blocked because its token does not belong to it."
        " Start a new session."
    )

    def __init__(self, details: str) -> None:
        super().__init__(details)
        self.details = details


class InvalidWorkflowIdException(Exception):
    """Raised when a workflow ID is invalid or not found."""


class CheckpointFetchError(Exception):
    """Raised when fetching a checkpoint from the Rails API fails.

    A distinct type lets callers tell a checkpoint-fetch failure apart from other runtime errors.
    """


class CheckpointSaveError(Exception):
    """Raised when the Rails API rejects a checkpoint POST.

    A distinct type lets callers tell a checkpoint-save failure apart from other runtime errors.
    """


class CheckpointAccessDeniedError(CheckpointSaveError):
    """Raised when GitLab refuses a checkpoint POST with 401 or 403.

    The session lost access, for example because GitLab Duo was turned off for its namespace, so no later checkpoint can
    save either.
    """


class EnvelopeVersionMismatchException(Exception):
    """Raised when an additional context envelope version does not satisfy the flow's constraint.

    This exception is raised during flow initialisation when the version field
    carried by an incoming envelope does not satisfy the semver constraint declared
    in the flow YAML configuration.  It is mapped to a ``FAILED_PRECONDITION``
    gRPC status so that callers receive a clear, actionable error rather than a
    generic ``INTERNAL`` failure.
    """


class WorkflowAlreadyFinishedException(Exception):
    """Raised when a run is requested for a session Rails already reports as ``finished``.

    Rails' state machine has no transition out of ``finished``, so the ``retry``
    event that any non-``created`` session would otherwise send is rejected with
    an HTTP 400 (``BadStatusEvent``) and surfaces to the caller as an
    ``INTERNAL`` gRPC error.  There is nothing left to execute, so this is not a
    failure: no status event is sent, the graph never starts, the Rails state is
    left untouched, and the RPC ends with ``OK``.
    """


class InvalidRequestException(Exception):
    """Raised when a workflow request contains invalid input and cannot be processed.

    Unlike most workflow errors, this exception must NOT transition the workflow
    to ``FAILED`` in Rails — the workflow state should remain unchanged so the
    user can correct their input and retry.  The gRPC call is terminated with
    ``INVALID_ARGUMENT`` so the caller receives a clear, actionable signal that
    the request was rejected due to bad input, without needing to inspect the
    session state further.

    The distinction from ``FAILED_PRECONDITION`` is intentional: the problem is
    the *input itself* (e.g. an empty goal on resume), not the state of the
    system.  ``INVALID_ARGUMENT`` is the correct gRPC status for arguments that
    are problematic regardless of system state.

    The ``message`` argument should describe the specific reason the request was
    rejected, as it is surfaced directly to the caller via the gRPC status detail.
    """
