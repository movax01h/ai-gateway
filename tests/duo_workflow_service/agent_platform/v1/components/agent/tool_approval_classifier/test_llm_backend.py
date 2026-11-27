from typing import Any, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier import (
    LLMReviewerBackend,
    ToolApprovalBackendName,
    ToolApprovalClassifier,
    ToolApprovalRequest,
    ToolApprovalReview,
    ToolApprovalVerdict,
    review_from_answer,
)
from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.request import (
    render_tool_approval_request,
)
from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.types import (
    MAX_REASON_CHARS,
)
from duo_workflow_service.entities import WorkflowStatusEnum

REVIEWER = "tool_approval_reviewer"

REQUEST = ToolApprovalRequest(
    tool_name="run_command",
    tool_args_json='{\n  "command": "make test"\n}',
    calling_agent="developer_agent",
    goal="Fix the failing test",
    display_message="Run `make test`",
    user_messages=("Fix the failing test", "Also run the linter"),
    recent_actions=('read_file {"file_path": "a.py"}',),
    flow_context={
        "goal": "Fix the failing test",
        "project_id": 7,
        "inputs": {"user_rule": "Use make."},
        REVIEWER: {"final_answer": {"verdict": "deny", "reason": "stale answer"}},
    },
)


class FakeReviewerGraph:
    def __init__(self, answer: Any = None, result: Any = None, error=None):
        self.answer = answer
        self.result = result
        self.error = error
        self.calls: list[tuple[Any, Optional[RunnableConfig]]] = []

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append((input, config))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return {"context": {REVIEWER: {"final_answer": self.answer}}}


def _backend(graph):
    return LLMReviewerBackend(reviewer_name=REVIEWER, compiled_graph=graph)


class TestReviewFromAnswer:
    @pytest.mark.parametrize("verdict", list(ToolApprovalVerdict))
    def test_reads_the_verdict_and_reason(self, verdict):
        assert review_from_answer(
            {"verdict": verdict.value, "reason": "because"}
        ) == ToolApprovalReview(verdict=verdict, reason="because")

    def test_bounds_the_reason(self):
        review = review_from_answer(
            {"verdict": "deny", "reason": "x" * (MAX_REASON_CHARS * 2)}
        )

        assert review is not None
        assert len(review.reason) == MAX_REASON_CHARS

    @pytest.mark.parametrize(
        "answer",
        [
            # A text answer, from a reviewer configured without the schema. 1.0.0's parser read some of these as allow.
            "allow: runs the tests.",
            "Allow me to explain why this is dangerous. deny",
            {"verdict": "Allow", "reason": "why"},
            {"verdict": "allow me to explain", "reason": "why"},
            {"verdict": "allow-listed? no: deny", "reason": "why"},
            {"verdict": ["allow"], "reason": "why"},
            {"verdict": "allow", "reason": ["why"]},
            {"reason": "why"},
            {},
            None,
        ],
    )
    def test_anything_but_a_schema_verdict_is_no_verdict(self, answer):
        assert review_from_answer(answer) is None


class TestLLMReviewerBackend:
    def test_is_the_llm_backend(self):
        assert _backend(FakeReviewerGraph()).name == ToolApprovalBackendName.LLM

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", list(ToolApprovalVerdict))
    async def test_returns_the_reviewers_verdict(self, verdict):
        backend = _backend(
            FakeReviewerGraph(answer={"verdict": verdict.value, "reason": "because"})
        )

        assert await backend.review(REQUEST, {}) == ToolApprovalReview(
            verdict=verdict, reason="because"
        )

    @pytest.mark.asyncio
    async def test_runs_the_reviewer_in_an_isolated_state(self):
        graph = FakeReviewerGraph(answer={"verdict": "allow", "reason": "safe"})
        config = RunnableConfig(tags=["flow"])

        await _backend(graph).review(REQUEST, config)

        [(state, forwarded)] = graph.calls
        assert forwarded is config
        assert state["status"] == WorkflowStatusEnum.EXECUTION
        assert state["conversation_history"] == {REVIEWER: []}
        assert state["ui_chat_log"] == []
        # The flow context resolves the reviewer's inputs, the goal describes the call, and an earlier answer of
        # the reviewer's own is never read back.
        assert state["context"] == {
            "goal": render_tool_approval_request(REQUEST),
            "project_id": 7,
            "inputs": {"user_rule": "Use make."},
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result",
        [
            {"context": {}},
            {"context": {REVIEWER: "not a dict"}},
            {"context": "not a dict"},
            "not a dict",
        ],
    )
    async def test_a_malformed_result_is_no_verdict(self, result):
        assert (
            await _backend(FakeReviewerGraph(result=result)).review(REQUEST, {}) is None
        )

    @pytest.mark.asyncio
    async def test_an_answer_without_a_verdict_is_no_verdict(self):
        backend = _backend(FakeReviewerGraph(answer="allow: looks fine to me."))

        assert await backend.review(REQUEST, {}) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [RuntimeError("boom"), GraphInterrupt()])
    async def test_the_classifier_turns_a_reviewer_failure_into_no_verdict(self, error):
        classifier = ToolApprovalClassifier(_backend(FakeReviewerGraph(error=error)))

        assert await classifier.classify(REQUEST) is None
