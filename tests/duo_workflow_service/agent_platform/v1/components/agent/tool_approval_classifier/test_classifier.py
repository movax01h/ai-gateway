import asyncio
from typing import Optional

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.errors import GraphInterrupt
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier import (
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
            "log_level": "info",
        } in logs

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
        ("config", "expected"),
        [
            pytest.param(
                RunnableConfig(tags=["flow"], metadata={"workflow_id": "1"}),
                {"tags": ["flow", TAG_NOSTREAM], "metadata": {"workflow_id": "1"}},
                id="adds_the_tag",
            ),
            pytest.param(
                RunnableConfig(tags=[TAG_NOSTREAM]),
                {"tags": [TAG_NOSTREAM]},
                id="does_not_repeat_the_tag",
            ),
            pytest.param(None, {"tags": [TAG_NOSTREAM]}, id="no_config"),
        ],
    )
    async def test_forwards_the_config_without_streaming(self, config, expected):
        backend = FakeBackend()
        original_tags = list(config["tags"]) if config else None

        await ToolApprovalClassifier(backend).classify(REQUEST, config)

        assert backend.configs == [expected]
        if config:
            assert config["tags"] == original_tags

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_a_non_positive_timeout(self, timeout):
        with pytest.raises(ValueError, match="timeout_seconds"):
            ToolApprovalClassifier(FakeBackend(), timeout_seconds=timeout)
