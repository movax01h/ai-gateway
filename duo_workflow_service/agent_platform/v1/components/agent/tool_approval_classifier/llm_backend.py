"""The ``llm`` backend: a read-only reviewer agent on a generic LLM."""

from typing import Any, Optional, Protocol

import structlog.stdlib
from langchain_core.runnables import RunnableConfig

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.request import (
    render_tool_approval_request,
)
from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.types import (
    ToolApprovalBackendName,
    ToolApprovalRequest,
    ToolApprovalReview,
    ToolApprovalVerdict,
)
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.entities import WorkflowStatusEnum

__all__ = [
    "READ_ONLY_REVIEWER_TOOLS",
    "TOOL_APPROVAL_REVIEW_SCHEMA_ID",
    "LLMReviewerBackend",
    "review_from_answer",
]

log = structlog.stdlib.get_logger(__name__)

# The only tools a reviewer may hold: it inspects the workspace and never changes it or runs anything.
READ_ONLY_REVIEWER_TOOLS = frozenset(
    {"read_file", "read_files", "grep", "find_files", "list_dir"}
)

# The response schema a reviewer answers through: a ``verdict`` of allow, ask or deny, and a ``reason``.
TOOL_APPROVAL_REVIEW_SCHEMA_ID = "tool_approval_review"


class ReviewerGraph(Protocol):
    """The reviewer component's compiled graph, as ``AgentComponent.compile_as_tool_approval_reviewer`` returns it."""

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any: ...


class LLMReviewerBackend:
    """Reviews a pending call with a read-only reviewer agent from the flow config.

    The reviewer is an ``AgentComponent`` running the ``tool_approval_review`` prompt with the
    ``TOOL_APPROVAL_REVIEW_SCHEMA_ID`` response schema, compiled without a checkpointer. Each review runs it in an
    isolated state: the calling flow's context, so its inputs resolve, with ``goal`` replaced by a description of the
    pending call.

    Args:
        reviewer_name: The reviewer component's name, which scopes its answer in the isolated state.
        compiled_graph: The reviewer component's compiled graph.
    """

    def __init__(self, *, reviewer_name: str, compiled_graph: ReviewerGraph):
        self._reviewer_name = reviewer_name
        self._compiled_graph = compiled_graph

    @property
    def name(self) -> ToolApprovalBackendName:
        return ToolApprovalBackendName.LLM

    async def review(
        self, request: ToolApprovalRequest, config: RunnableConfig
    ) -> Optional[ToolApprovalReview]:
        """Run the reviewer on one pending call.

        Returns:
            The reviewer's verdict, or ``None`` when its answer isn't one.
        """
        result_state = await self._compiled_graph.ainvoke(
            self._initial_state(request), config=config
        )

        review = review_from_answer(self._final_answer(result_state))
        if review is None:
            log.warning(
                "Tool approval reviewer answered without a verdict",
                reviewer=self._reviewer_name,
                tool_name=request.tool_name,
            )
        return review

    def _initial_state(self, request: ToolApprovalRequest) -> FlowState:
        # The reviewer's own entry is dropped, so a stale answer is never read back as this run's.
        context = {
            key: value
            for key, value in request.flow_context.items()
            if key != self._reviewer_name
        }
        return {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {self._reviewer_name: []},
            "ui_chat_log": [],
            # The prompt's user template introduces it, so a change to that text bumps the prompt version.
            "context": {**context, "goal": render_tool_approval_request(request)},
            "agent_context_limits": {},
        }

    def _final_answer(self, result_state: Any) -> Any:
        # Plain lookups, so a malformed result state means no verdict rather than an error.
        if not isinstance(result_state, dict):
            return None
        context = result_state.get("context")
        if not isinstance(context, dict):
            return None
        own = context.get(self._reviewer_name)
        if not isinstance(own, dict):
            return None
        return own.get("final_answer")


def review_from_answer(answer: Any) -> Optional[ToolApprovalReview]:
    """Read a verdict from the reviewer's structured answer, its response schema's arguments.

    The schema already makes ``verdict`` one of ``allow``, ``ask`` or ``deny``. This checks it again, so an answer of
    any other shape, such as a text one from a reviewer configured without the schema, is no verdict, never a guess.

    Returns:
        The review, or ``None`` when the answer isn't a verdict.
    """
    if not isinstance(answer, dict):
        return None
    verdict = answer.get("verdict")
    reason = answer.get("reason")
    if not isinstance(verdict, str) or not isinstance(reason, (str, type(None))):
        return None
    try:
        return ToolApprovalReview(verdict=ToolApprovalVerdict(verdict), reason=reason)
    except ValueError:
        return None
