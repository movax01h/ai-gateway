"""Run a classifier backend on one pending tool call, failing closed."""

import asyncio
from typing import Optional, Protocol

import structlog.stdlib
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.errors import GraphBubbleUp

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.types import (
    ToolApprovalBackendName,
    ToolApprovalRequest,
    ToolApprovalReview,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ToolApprovalBackend",
    "ToolApprovalClassifier",
]

log = structlog.stdlib.get_logger(__name__)

# Wall-clock bound on one review, including every model and tool call the backend makes. A placeholder, set to the
# 30s ceiling from #3673, until #3677 measures p50/p99 latency per backend, so #3674 can set one per backend.
DEFAULT_TIMEOUT_SECONDS = 30.0


class ToolApprovalBackend(Protocol):
    """One way of reviewing a pending tool call, such as a decision model or an LLM reviewer agent.

    A backend must:

    - Pass the ``config`` it's given to every model and runnable call, so the review isn't streamed to the client.
    - Compile any graph it runs without a checkpointer, so it never writes into the flow's own thread.
    - Never call ``interrupt()``.
    - Let ``asyncio.CancelledError`` propagate.
    """

    @property
    def name(self) -> ToolApprovalBackendName: ...

    async def review(
        self, request: ToolApprovalRequest, config: RunnableConfig
    ) -> Optional[ToolApprovalReview]:
        """Review one pending call.

        Returns ``None`` when the backend reached no verdict, for example an unparsable answer or a refusal. It may
        raise; ``ToolApprovalClassifier`` turns any error into no verdict.
        """


class ToolApprovalClassifier:
    """Reviews pending tool calls with one backend, bounded by a timeout.

    It fails closed: a timeout, an error, an interrupt raised inside the backend's own run, or no verdict all return
    ``None``, which means the human decides, as without the classifier. It never turns a failure into ``allow``.

    Args:
        backend: The backend to review with.
        timeout_seconds: Wall-clock bound on one review.
    """

    def __init__(
        self,
        backend: ToolApprovalBackend,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self._backend = backend
        self._timeout_seconds = timeout_seconds

    @property
    def backend_name(self) -> ToolApprovalBackendName:
        return self._backend.name

    async def classify(
        self,
        request: ToolApprovalRequest,
        config: Optional[RunnableConfig] = None,
    ) -> Optional[ToolApprovalReview]:
        """Review one pending call.

        Args:
            request: What the backend sees of the call.
            config: The calling node's config, forwarded so the review nests under its trace. The review's own model
                output is never streamed to the client.

        Returns:
            The review, or ``None`` when the human should decide.
        """
        log_context = {
            "tool_name": request.tool_name,
            "calling_agent": request.calling_agent,
            "backend": self.backend_name.value,
        }

        try:
            review = await asyncio.wait_for(
                self._backend.review(request, _without_streaming(config)),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.warning(
                "Tool approval classifier timed out; falling back to human approval",
                timeout_seconds=self._timeout_seconds,
                **log_context,
            )
            return None
        except GraphBubbleUp:
            # The backend's own run paused or redirected, for example on an approval for one of its own tools. It must
            # not be resumed: on resume, LangGraph reruns a node from the top, so if ``classify`` shared a node with the
            # human approval ``interrupt()``, the backend's own ``interrupt()`` would receive the human's answer.
            log.warning(
                "Tool approval classifier paused or redirected inside its own run; falling back to human approval",
                **log_context,
            )
            return None
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception(
                "Tool approval classifier failed; falling back to human approval",
                **log_context,
            )
            return None

        if review is None:
            log.info(
                "Tool approval classifier reached no verdict; falling back to human approval",
                **log_context,
            )
            return None

        if not isinstance(review, ToolApprovalReview):
            log.warning(
                "Tool approval classifier returned an invalid review; falling back to human approval",
                review_type=type(review).__name__,
                **log_context,
            )
            return None

        log.info(
            "Tool approval classifier verdict",
            verdict=review.verdict.value,
            **log_context,
        )
        return review


def _without_streaming(config: Optional[RunnableConfig]) -> RunnableConfig:
    """Forward ``config`` with ``TAG_NOSTREAM``, so the review's tokens aren't streamed as the calling agent's."""
    merged: RunnableConfig = {**(config or {})}
    tags = list(merged.get("tags") or [])
    if TAG_NOSTREAM not in tags:
        tags.append(TAG_NOSTREAM)
    merged["tags"] = tags
    return merged
