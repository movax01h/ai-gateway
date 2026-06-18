import time
from typing import Any, Optional
from uuid import UUID

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.event_types import (
    LlmInputSentEvent,
    LlmRequestFailedEvent,
    LlmResponseReceivedEvent,
    ToolExecutionFailedEvent,
    ToolInvokedEvent,
    ToolResponseReceivedEvent,
)

logger = structlog.stdlib.get_logger("audit_callback_handler")


def _extract_model_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
    invocation_params = kwargs.get("invocation_params", {})
    if model := invocation_params.get("model"):
        return model
    if model := invocation_params.get("model_name"):
        return model
    if model := serialized.get("kwargs", {}).get("model"):
        return model
    return "unknown"


class AuditEventCallbackHandler(AsyncCallbackHandler):  # pylint: disable=too-many-ancestors
    def __init__(self, collector: AuditEventCollector, workflow_id: str):
        self._collector = collector
        self._workflow_id = workflow_id
        self._llm_start_times: dict[str, float] = {}
        self._llm_model_names: dict[str, str] = {}
        self._tool_names: dict[str, str] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[str(run_id)] = time.monotonic()
        model_name = _extract_model_name(serialized, kwargs)
        self._llm_model_names[str(run_id)] = model_name
        self._collector.capture(
            LlmInputSentEvent(
                workflow_id=self._workflow_id,
                model_name=model_name,
                prompt_content=str(messages),
            )
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        latency_ms: Optional[float] = None
        start = self._llm_start_times.pop(str(run_id), None)
        if start:
            latency_ms = (time.monotonic() - start) * 1000

        model_name = self._llm_model_names.pop(str(run_id), None)
        llm_output = response.llm_output or {}
        if not model_name:
            model_name = llm_output.get("model_name", "unknown")

        response_content = ""
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            response_content = gen.text
            gen_info = gen.generation_info or {}
            finish_reason = gen_info.get("finish_reason")

            message = getattr(gen, "message", None)
            if message:
                usage_meta = getattr(message, "usage_metadata", None) or {}
                resp_meta = getattr(message, "response_metadata", None) or {}
                usage_from_resp = resp_meta.get("usage", {})

                prompt_tokens = (
                    usage_meta.get("input_tokens")
                    or usage_from_resp.get("input_tokens")
                    or usage_from_resp.get("prompt_tokens")
                )
                completion_tokens = (
                    usage_meta.get("output_tokens")
                    or usage_from_resp.get("output_tokens")
                    or usage_from_resp.get("completion_tokens")
                )
                if not finish_reason:
                    finish_reason = resp_meta.get(
                        "stop_reason", resp_meta.get("finish_reason")
                    )

        self._collector.capture(
            LlmResponseReceivedEvent(
                workflow_id=self._workflow_id,
                model_name=model_name,
                response_content=response_content,
                prompt_token_count=prompt_tokens,
                completion_token_count=completion_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self._tool_names[str(run_id)] = tool_name
        self._collector.capture(
            ToolInvokedEvent(
                workflow_id=self._workflow_id,
                tool_name=tool_name,
                tool_args=kwargs.get("inputs"),
            )
        )

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = self._tool_names.pop(str(run_id), "unknown")
        response_content = str(output)
        self._collector.capture(
            ToolResponseReceivedEvent(
                workflow_id=self._workflow_id,
                tool_name=tool_name,
                response_content=response_content,
                response_length=len(response_content),
            )
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = self._tool_names.pop(str(run_id), kwargs.get("name", "unknown"))
        self._collector.capture(
            ToolExecutionFailedEvent(
                workflow_id=self._workflow_id,
                tool_name=tool_name,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        latency_ms: Optional[float] = None
        start = self._llm_start_times.pop(str(run_id), None)
        if start:
            latency_ms = (time.monotonic() - start) * 1000

        model_name = self._llm_model_names.pop(str(run_id), "unknown")
        self._collector.capture(
            LlmRequestFailedEvent(
                workflow_id=self._workflow_id,
                model_name=model_name,
                error_type=type(error).__name__,
                error_message=str(error),
                latency_ms=latency_ms,
            )
        )
