import asyncio
import dataclasses
from typing import Optional, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tracers.base import AsyncBaseTracer
from langgraph.constants import TAG_NOSTREAM
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.constants import TOOL_APPROVAL_REVIEW_TAG
from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier import (
    REVIEW_RUN_NAME,
    TRUNCATED_CALL_REASON,
    ToolApprovalBackendName,
    ToolApprovalClassifier,
    ToolApprovalRequest,
    ToolApprovalReview,
    ToolApprovalVerdict,
)

REQUEST = ToolApprovalRequest(
    tool_name="run_command",
    tool_args_json='{"command": "make test"}',
    calling_agent="developer",
)


class FakeBackend:
    def __init__(
        self,
        result=None,
        error: Optional[BaseException] = None,
        delay=0.0,
        name=ToolApprovalBackendName.OPENAI_DECISIONS,
    ):
        self.result = result
        self.error = error
        self.delay = delay
        self._name = name
        self.configs: list[RunnableConfig] = []

    @property
    def name(self) -> ToolApprovalBackendName:
        return self._name

    async def review(self, request, config):
        self.configs.append(config)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class ConfigDroppingBackend:
    """Breaks the backend contract: its model call drops the ``config`` it's given."""

    name = ToolApprovalBackendName.LLM

    def __init__(self):
        self.model = FakeListChatModel(responses=["the reviewer's tokens"])

    async def review(self, request, config):
        await self.model.ainvoke("review this call")
        return ToolApprovalReview(verdict=ToolApprovalVerdict.ALLOW)


class LLMModelBackend:
    """Runs its model call inside one runnable invoked with the ``config`` it's given, as the reviewer graph is."""

    name = ToolApprovalBackendName.LLM

    def __init__(self):
        self.model = FakeListChatModel(responses=["allow"])

    async def review(self, request, config):
        async def run(_input, config: RunnableConfig):
            await self.model.ainvoke("review this call", config)
            return ToolApprovalReview(verdict=ToolApprovalVerdict.ALLOW)

        return await RunnableLambda(run).ainvoke(None, config)


class RunCollector(AsyncBaseTracer):
    """Keeps every traced run, as LangSmith would receive it."""

    def __init__(self):
        super().__init__()
        self.runs = []

    async def _persist_run(self, run):
        pass

    async def _start_trace(self, run):
        await super()._start_trace(run)
        self.runs.append(run)


class _GraphState(TypedDict):
    done: bool


class TestToolApprovalClassifier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", list(ToolApprovalVerdict))
    async def test_returns_the_backend_review(self, verdict):
        review = ToolApprovalReview(verdict=verdict, reason="why")
        classifier = ToolApprovalClassifier(FakeBackend(result=review))

        with capture_logs() as logs:
            assert await classifier.classify(REQUEST) == review

        assert {
            "event": "Tool approval classifier verdict",
            "verdict": verdict.value,
            "tool_name": "run_command",
            "calling_agent": "developer",
            "backend": "openai_decisions",
            "truncated": False,
            "log_level": "info",
        } in logs

    @pytest.mark.asyncio
    async def test_asks_the_human_when_a_truncated_call_is_allowed(self):
        request = dataclasses.replace(REQUEST, truncated=True)
        backend = FakeBackend(
            result=ToolApprovalReview(verdict=ToolApprovalVerdict.ALLOW, reason="safe")
        )

        with capture_logs() as logs:
            review = await ToolApprovalClassifier(backend).classify(request)

        assert review == ToolApprovalReview(
            verdict=ToolApprovalVerdict.ASK, reason=TRUNCATED_CALL_REASON
        )
        assert [
            (log["event"], log.get("verdict"), log["truncated"]) for log in logs
        ] == [
            (
                "Tool approval classifier allowed a truncated call; asking the human instead",
                None,
                True,
            ),
            ("Tool approval classifier verdict", "ask", True),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "verdict", [ToolApprovalVerdict.ASK, ToolApprovalVerdict.DENY]
    )
    async def test_keeps_any_other_verdict_on_a_truncated_call(self, verdict):
        request = dataclasses.replace(REQUEST, truncated=True)
        review = ToolApprovalReview(verdict=verdict, reason="why")

        assert (
            await ToolApprovalClassifier(FakeBackend(result=review)).classify(request)
            == review
        )

    def test_exposes_the_backend_name(self):
        classifier = ToolApprovalClassifier(FakeBackend())

        assert classifier.backend_name == ToolApprovalBackendName.OPENAI_DECISIONS

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("backend_kwargs", "timeout_seconds", "expected_log"),
        [
            pytest.param(
                {},
                1.0,
                {
                    "event": "Tool approval classifier reached no verdict; falling back to human approval",
                    "log_level": "info",
                },
                id="no_verdict",
            ),
            pytest.param(
                {
                    "result": ToolApprovalReview(verdict=ToolApprovalVerdict.ALLOW),
                    "delay": 1,
                },
                0.01,
                {
                    "event": "Tool approval classifier timed out; falling back to human approval",
                    "log_level": "warning",
                    "timeout_seconds": 0.01,
                },
                id="timeout",
            ),
            pytest.param(
                {"error": RuntimeError("boom")},
                1.0,
                {
                    "event": "Tool approval classifier failed; falling back to human approval",
                    "log_level": "error",
                    "exc_info": True,
                },
                id="error",
            ),
            pytest.param(
                {"error": GraphInterrupt()},
                1.0,
                {
                    "event": "Tool approval classifier paused or redirected inside its own run; falling back to human "
                    "approval",
                    "log_level": "warning",
                },
                id="interrupt",
            ),
            pytest.param(
                {"result": {"verdict": "allow"}},
                1.0,
                {
                    "event": "Tool approval classifier returned an invalid review; falling back to human approval",
                    "log_level": "warning",
                    "review_type": "dict",
                },
                id="dict_result",
            ),
            pytest.param(
                {"result": ToolApprovalVerdict.ALLOW},
                1.0,
                {
                    "event": "Tool approval classifier returned an invalid review; falling back to human approval",
                    "log_level": "warning",
                    "review_type": "ToolApprovalVerdict",
                },
                id="verdict_result",
            ),
            pytest.param(
                {"result": "allow"},
                1.0,
                {
                    "event": "Tool approval classifier returned an invalid review; falling back to human approval",
                    "log_level": "warning",
                    "review_type": "str",
                },
                id="string_result",
            ),
        ],
    )
    async def test_fails_closed(self, backend_kwargs, timeout_seconds, expected_log):
        classifier = ToolApprovalClassifier(
            FakeBackend(**backend_kwargs), timeout_seconds=timeout_seconds
        )

        with capture_logs() as logs:
            assert await classifier.classify(REQUEST) is None

        assert expected_log.items() <= logs[-1].items()

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self):
        classifier = ToolApprovalClassifier(FakeBackend(error=asyncio.CancelledError()))

        with pytest.raises(asyncio.CancelledError):
            await classifier.classify(REQUEST)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("config", "expected_tags", "expected_metadata"),
        [
            pytest.param(
                RunnableConfig(tags=["flow"], metadata={"workflow_id": "1"}),
                ["flow", TAG_NOSTREAM, TOOL_APPROVAL_REVIEW_TAG],
                {"workflow_id": "1"},
                id="adds_the_tags",
            ),
            pytest.param(
                RunnableConfig(tags=[TOOL_APPROVAL_REVIEW_TAG, TAG_NOSTREAM]),
                [TOOL_APPROVAL_REVIEW_TAG, TAG_NOSTREAM],
                {},
                id="does_not_repeat_the_tags",
            ),
            pytest.param(
                None, [TAG_NOSTREAM, TOOL_APPROVAL_REVIEW_TAG], {}, id="no_config"
            ),
        ],
    )
    async def test_forwards_the_config_as_a_named_review_run_without_streaming(
        self, config, expected_tags, expected_metadata
    ):
        backend = FakeBackend()
        original = dict(config) if config else None

        await ToolApprovalClassifier(backend).classify(REQUEST, config)

        assert backend.configs == [
            {
                "tags": expected_tags,
                "run_name": REVIEW_RUN_NAME,
                "metadata": {
                    **expected_metadata,
                    "tool_approval_tool_name": "run_command",
                    "tool_approval_calling_agent": "developer",
                    "tool_approval_backend": "openai_decisions",
                    "tool_approval_truncated": False,
                },
            }
        ]
        if config:
            assert config == original

    @pytest.mark.asyncio
    async def test_the_review_nests_under_the_calling_node_in_the_trace(self):
        backend = LLMModelBackend()
        classifier = ToolApprovalClassifier(backend)
        tracer = RunCollector()

        async def agent_node(state, config: RunnableConfig):
            await classifier.classify(REQUEST, config)
            return {"done": True}

        graph = StateGraph(_GraphState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        await graph.compile().ainvoke({"done": False}, {"callbacks": [tracer]})

        runs = {run.name: run for run in tracer.runs}
        review = runs[REVIEW_RUN_NAME]
        assert review.parent_run_id == runs["agent"].id
        assert review.extra["metadata"]["tool_approval_tool_name"] == "run_command"
        assert TOOL_APPROVAL_REVIEW_TAG in review.tags
        # The backend's own model call is a child of the review run, and inherits its tag.
        assert runs["FakeListChatModel"].parent_run_id == review.id
        assert TOOL_APPROVAL_REVIEW_TAG in runs["FakeListChatModel"].tags

    @pytest.mark.asyncio
    async def test_a_backend_model_call_without_config_does_not_stream(self):
        classifier = ToolApprovalClassifier(ConfigDroppingBackend())
        agent_model = FakeListChatModel(responses=["the agent's tokens"])

        async def agent_node(state, config: RunnableConfig):
            await agent_model.ainvoke("work", config)
            await classifier.classify(REQUEST, config)
            return {"done": True}

        graph = StateGraph(_GraphState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)

        streamed = "".join(
            [
                message.content
                async for message, _ in graph.compile().astream(
                    {"done": False}, stream_mode="messages"
                )
            ]
        )

        assert streamed == "the agent's tokens"

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_a_non_positive_timeout(self, timeout):
        with pytest.raises(ValueError, match="timeout_seconds"):
            ToolApprovalClassifier(FakeBackend(), timeout_seconds=timeout)
