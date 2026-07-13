import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from duo_workflow_service.agents.project_utils import resolve_project_name_for_tool
from duo_workflow_service.agents.prompt_adapter import BasePromptAdapter
from duo_workflow_service.agents.tool_call_validator import (
    retry_malformed_tool_calls,
    validate_tool_calls,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
    build_compaction_agent_message,
    build_compaction_tool_card,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.entities.state import (
    TIER_ACCESS_DENIED_SUB_TYPE,
    ApprovalStateRejection,
    ChatWorkflowState,
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.errors.typing import NotifiableException
from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfoService,
)
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.slash_commands.goal_parser import (
    is_slash_command,
)
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from duo_workflow_service.tools import Toolset
from duo_workflow_service.tracking.errors import log_exception
from lib.context import LLMFinishReason, extract_finish_reason

log = structlog.stdlib.get_logger("chat_agent")

_COMMAND_TOOL_NAMES = {"run_command", "run_git_command"}
_GIT_COMMAND_TOOL_NAME = "run_git_command"


def _suggest_patterns(tool_name: str, tool_args: dict[str, Any]) -> list[str]:
    """Suggest glob patterns for a tool call.

    For command tools, suggests up to two patterns: the most specific prefix
    (one level broader than the exact command) and the broadest useful prefix
    (program + subcommand). Users can also type a custom glob in the UI.

    Example: 'docker compose -f dev.yml up -d'
    suggests: ['docker compose -f dev.yml up *', 'docker compose *']

    Note: uses naive whitespace splitting, so commands with quoted arguments
    (e.g. echo "hello world") may produce imprecise patterns. This is
    acceptable — patterns are suggestions, not security boundaries.
    """
    if tool_name not in _COMMAND_TOOL_NAMES:
        return []

    if tool_name == _GIT_COMMAND_TOOL_NAME:
        # run_git_command uses command (subcommand) + args schema.
        # Prepend "git" so patterns like "git checkout *" work intuitively.
        subcommand = tool_args.get("command") or ""
        args = tool_args.get("args") or ""
        command = (
            f"git {subcommand} {args}".strip() if args else f"git {subcommand}".strip()
        )
    else:
        # run_command uses either "command" (current schema) or "program"+"args" (legacy)
        command = tool_args.get("command") or ""
        if not command and "program" in tool_args:
            program = tool_args.get("program", "")
            args = tool_args.get("args", "")
            command = f"{program} {args}".strip() if args else program

    if not command:
        return []

    parts = command.split()
    if len(parts) <= 2:
        return []

    most_specific = " ".join(parts[:-1]) + " *"
    broadest = " ".join(parts[:2]) + " *"

    if most_specific == broadest:
        return [most_specific]

    return [most_specific, broadest]


class ChatAgent:
    def __init__(
        self,
        name: str,
        prompt_adapter: BasePromptAdapter,
        tools_registry: ToolsRegistry,
        toolset: Toolset,
        system_template_override: str | None,
        optimizer_pipeline: HistoryOptimizerPipeline,
        manual_compactor: CompactionOptimizer | None = None,
    ):
        self.name = name
        self.prompt_adapter = prompt_adapter
        self.tools_registry = tools_registry
        self.system_template_override = system_template_override
        self._optimizer_pipeline = optimizer_pipeline
        self._manual_compactor = manual_compactor
        self.toolset = toolset

    async def _get_approvals(
        self, message: AIMessage, preapproved_tools: List[str], state: ChatWorkflowState
    ) -> tuple[bool, list[UiChatLog]]:
        approval_required = False
        approval_messages = []

        for call in message.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            auto_approved_by_agentic_mock_model = getattr(
                self.prompt_adapter.get_model(),
                "_is_auto_approved_by_agentic_mock_model",
                False,
            )
            needs_approval = (
                self.tools_registry
                and await self.tools_registry.approval_required(tool_name, tool_args)
                and tool_name not in preapproved_tools
                and not auto_approved_by_agentic_mock_model
            )

            if needs_approval:
                approval_required = True

                project_name = resolve_project_name_for_tool(state.get("project"), call)
                if project_name:
                    tool_info_args = {
                        **tool_args,
                        "project_name": project_name,
                    }
                else:
                    tool_info_args = tool_args
                tool_info = ToolInfo(name=tool_name, args=tool_info_args)
                suggested = _suggest_patterns(tool_name, tool_args)
                if suggested:
                    tool_info["suggested_patterns"] = suggested

                log.debug(
                    "Tool call requires approval",
                    extra={
                        "tool_name": tool_name,
                        "tool_info_keys": list(tool_info.keys()),
                        "suggested_patterns": suggested,
                    },
                )

                approval_messages.append(
                    UiChatLog(
                        message_type=MessageTypeEnum.REQUEST,
                        message_sub_type=None,
                        content=f"Tool {tool_name} requires approval. Please confirm if you want to proceed.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=tool_info,
                        additional_context=None,
                        message_id=f"request-{call['id']}",
                    )
                )

        return approval_required, approval_messages

    def _handle_wrong_messages_order_for_tool_execution(self, state: ChatWorkflowState):
        # A special fix for the following use case:
        #
        # - A user is asked to approve/deny a tool execution
        # - The user stops the chat instead and specifies a follow up message
        #
        # LLM returns an error because a tool call execution was followed by a human message instead of a tool result
        #
        # Expected to be refactored in:
        # - https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1461
        if (
            self.name in state["conversation_history"]
            and len(state["conversation_history"][self.name]) > 1
        ):
            tool_call_message = state["conversation_history"][self.name][-2]
            user_message = state["conversation_history"][self.name][-1]

            if (
                isinstance(tool_call_message, AIMessage)
                and len(tool_call_message.tool_calls) > 0
                and isinstance(user_message, HumanMessage)
            ):
                messages: list[BaseMessage] = [
                    ToolMessage(
                        content="Tool is cancelled and a user will provide a follow up message.",
                        tool_call_id=tool_call.get("id"),
                    )
                    for tool_call in getattr(tool_call_message, "tool_calls", [])
                ]

                state["conversation_history"][self.name][-2:] = [
                    tool_call_message,
                    *messages,
                    user_message,
                ]

    def _handle_approval_rejection(
        self, state: ChatWorkflowState, approval_state: ApprovalStateRejection
    ) -> list[BaseMessage]:
        last_message = state["conversation_history"][self.name][-1]

        # An empty text box for tool cancellation results in a 'null' message. Converting to None
        # todo: remove this line once we have fixed the frontend to return None instead of 'null'
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1259
        normalized_message = (
            None if approval_state.message == "null" else approval_state.message
        )

        tool_message = (
            f"Tool is cancelled temporarily as user has a comment. Comment: {normalized_message}"
            if normalized_message
            else "Tool is cancelled by user. Don't run the command and stop tool execution in progress."
        )

        messages: list[BaseMessage] = [
            ToolMessage(
                content=tool_message,
                tool_call_id=tool_call.get("id"),
            )
            for tool_call in getattr(last_message, "tool_calls", [])
        ]

        # update history
        state["conversation_history"][self.name].extend(messages)
        return messages

    async def _get_agent_response(self, state: ChatWorkflowState) -> BaseMessage:
        return await self.prompt_adapter.get_response(
            state, system_template_override=self.system_template_override
        )

    async def _build_response(
        self, agent_response: BaseMessage, state: ChatWorkflowState
    ) -> Dict[str, Any]:
        result = {
            "conversation_history": {self.name: [agent_response]},
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
        }

        self._build_text_response(agent_response, state, result)
        if isinstance(agent_response, AIMessage) and agent_response.tool_calls:
            await self._build_tool_response(agent_response, state, result)

        return result

    def _build_text_response(
        self,
        agent_response: BaseMessage,
        state: ChatWorkflowState,
        result: Dict[str, Any],
    ):
        content = agent_response.text()
        ui_chat_log = []

        if content:
            tier_payload = self._extract_tier_access_denied(state)
            chat_log = UiChatLog(
                message_type=MessageTypeEnum.AGENT,
                message_sub_type=(
                    TIER_ACCESS_DENIED_SUB_TYPE if tier_payload else None
                ),
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                correlation_id=None,
                tool_info=None,
                additional_context=None,
                message_id=agent_response.id,
            )
            if tier_payload and tier_payload.get("required_plan") is not None:
                chat_log["required_plan"] = tier_payload["required_plan"]
            ui_chat_log.append(chat_log)

        result["ui_chat_log"] = ui_chat_log

    def _extract_tier_access_denied(
        self, state: ChatWorkflowState
    ) -> Optional[Dict[str, Any]]:
        """Return tier_access_denied payload from the current turn's ToolMessages, else None.

        Relies on the new agent_response NOT yet being in state["conversation_history"]
        """
        history = state.get("conversation_history", {}).get(self.name, [])
        for msg in reversed(history):
            if isinstance(msg, AIMessage):
                return None
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content
            if (
                not isinstance(content, str)
                or TIER_ACCESS_DENIED_SUB_TYPE
                not in content  # fast path: skip json.loads when payload can't match
            ):
                continue
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("error") == TIER_ACCESS_DENIED_SUB_TYPE
            ):
                return payload
        return None

    async def _build_tool_response(
        self,
        agent_response: AIMessage,
        state: ChatWorkflowState,
        result: Dict[str, Any],
    ):
        result["status"] = WorkflowStatusEnum.EXECUTION

        preapproved_tools = state.get("preapproved_tools") or []
        tools_need_approval, approval_messages = await self._get_approvals(
            agent_response, preapproved_tools, state
        )

        if len(agent_response.tool_calls) > 0 and tools_need_approval:
            result["status"] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
            result["ui_chat_log"].extend(approval_messages)

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        log.info("We are hitting an error while making an llm call.")

        if isinstance(error, APIStatusError) and 500 <= error.status_code < 600:
            ui_content = (
                "There was an error connecting to the chosen LLM provider, please contact support "
                "if the issue persists."
            )
        else:
            ui_content = (
                "There was an error processing your request in the Duo Agent Platform, please contact support "
                "if the issue persists."
            )

        raise NotifiableException(ui_content) from error

    @staticmethod
    def _append_optimizer_ui_logs(
        base: List[UiChatLog],
        optimizer_logs: List[UiChatLog],
    ) -> List[UiChatLog]:
        """Append optimizer-produced UI entries after the base entries.

        The optimizer runs before the assistant LLM call, so chronologically the optimizer events happen first. Front
        end sorts UI entries by timestamp, so the displayed order remains optimizer → assistant regardless of array
        position. Appending (rather than prepending) is what the front end currently requires to render the compaction
        tool card; prepending causes it to be silently dropped.
        """
        if not optimizer_logs:
            return base
        return [*base, *optimizer_logs]

    def _detect_compact_command(
        self, state: ChatWorkflowState
    ) -> tuple[bool, str | None]:
        """Detect a trailing ``/compact`` slash command in conversation history.

        Returns:
            A ``(detected, user_instruction)`` tuple. ``detected`` is True when
            the last ``HumanMessage`` is ``/compact`` (with any trailing text).
            ``user_instruction`` is the text after ``/compact`` (or ``None`` if
            absent or not detected), forwarded to the manual compaction prompt.
        """
        history = state.get("conversation_history", {}).get(self.name, [])
        if not history or not isinstance(history[-1], HumanMessage):
            return False, None
        last_content = history[-1].content
        if not isinstance(last_content, str) or not is_slash_command(last_content):
            return False, None
        command_name, remaining_text = slash_command_parse(last_content)
        if command_name != "compact":
            return False, None
        return True, remaining_text

    async def _handle_manual_compaction(
        self,
        state: ChatWorkflowState,
        user_instruction: str | None,
    ) -> Dict[str, Any]:
        """Handle a user-initiated ``/compact`` slash command.

        Runs compaction in manual mode, replaces conversation_history in place, and returns a single tool-card UI entry.
        On failure or no-op, returns a status entry and leaves history unchanged.
        """
        if self._manual_compactor is None:
            return {
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": [
                    build_compaction_tool_card(
                        trigger="manual",
                        result=None,
                        content="Compaction failed",
                        status=ToolStatus.FAILURE,
                    ),
                    build_compaction_agent_message("Compaction is not available."),
                ],
            }

        history = state["conversation_history"].get(self.name, [])
        history_to_compact = history[:-1]

        if not history_to_compact:
            return {
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": [
                    build_compaction_tool_card(
                        trigger="manual",
                        result=None,
                        content="Nothing to compact",
                        status=ToolStatus.SUCCESS,
                    ),
                    build_compaction_agent_message(
                        "There is no conversation history to compact yet."
                    ),
                ],
            }

        result = await self._manual_compactor.optimize_manual(
            history_to_compact,
            user_instruction=user_instruction,
        )

        if not result.succeeded:
            log.warning(
                "Manual compaction did not produce a summary",
                agent_name=self.name,
                error_type=(
                    type(result.error).__name__ if result.error is not None else None
                ),
                was_compacted=result.was_compacted,
            )
            return {
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": list(result.ui_chat_logs),
            }

        state["conversation_history"][self.name] = result.messages

        return {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "ui_chat_log": list(result.ui_chat_logs),
        }

    async def run(self, state: ChatWorkflowState) -> Dict[str, Any]:
        approval_state = state.get("approval", None)

        # When the conversation ends with an AIMessage (no new user input), we have two scenarios:
        # 1. AIMessage has pending tool_calls: This occurs when the workflow was interrupted
        #    mid-execution (e.g., during RETRY after connection loss). We insert synthetic
        #    ToolMessages to satisfy LLM's expectation of tool results, allowing it to
        #    respond gracefully about the interruption.
        # 2. AIMessage has no tool_calls: The AI already responded, and there's no new user
        #    input to process. Return INPUT_REQUIRED to wait for actual user input.
        if not approval_state:
            history = state.get("conversation_history", {}).get(self.name, [])
            if history and isinstance(history[-1], AIMessage):
                last_ai = history[-1]

                if getattr(last_ai, "tool_calls", None):
                    log.warning(
                        "Agent called with pending tool_calls - inserting cancellation messages",
                        tool_call_count=len(last_ai.tool_calls),
                    )
                    synthetic_tool_messages = [
                        ToolMessage(
                            content=(
                                "Tool execution was interrupted. Retry this tool call,"
                                " or stop and report to the user if you detect you"
                                " are in a retry loop."
                            ),
                            tool_call_id=tc.get("id"),
                        )
                        for tc in last_ai.tool_calls
                    ]
                    state["conversation_history"][self.name].extend(
                        synthetic_tool_messages
                    )
                else:
                    log.info(
                        "No new user input detected, skipping LLM call",
                        last_message_type=type(last_ai).__name__,
                    )
                    return {
                        "status": WorkflowStatusEnum.INPUT_REQUIRED,
                        "ui_chat_log": [],
                    }

        is_compact, compact_user_instruction = self._detect_compact_command(state)
        if is_compact:
            return await self._handle_manual_compaction(state, compact_user_instruction)

        self._handle_wrong_messages_order_for_tool_execution(state)

        # Handle approval rejection
        if isinstance(approval_state, ApprovalStateRejection):
            self._handle_approval_rejection(state, approval_state)

        history = state["conversation_history"].get(self.name, [])
        (
            optimized_history,
            optimization_results,
        ) = await self._optimizer_pipeline.optimize(history)
        state["conversation_history"][self.name] = optimized_history
        optimizer_ui_logs: List[UiChatLog] = [
            entry for result in optimization_results for entry in result.ui_chat_logs
        ]

        try:
            with GitLabServiceContext(
                GitLabInstanceInfoService(),
                project=state.get("project"),
                namespace=state.get("namespace"),
            ):
                agent_response = await self._get_agent_response(state)

                # Check for abnormal finish reasons
                finish_reason = extract_finish_reason(agent_response.response_metadata)
                if finish_reason in LLMFinishReason.abnormal_values():
                    log.warning(f"LLM stopped abnormally with reason: {finish_reason}")

                # Validate tool calls before building the response. If malformed, retry the call.
                if isinstance(agent_response, AIMessage) and agent_response.tool_calls:
                    validation_errors = validate_tool_calls(
                        self.toolset, agent_response
                    )
                    if validation_errors:
                        agent_response = await retry_malformed_tool_calls(
                            toolset=self.toolset,
                            agent_response=agent_response,
                            validation_errors=validation_errors,
                            state=state,
                            agent_name=self.name,
                            get_agent_response=self._get_agent_response,
                        )

            response = await self._build_response(agent_response, state)
            response["ui_chat_log"] = self._append_optimizer_ui_logs(
                response["ui_chat_log"], optimizer_ui_logs
            )
            return response

        except SlashCommandValidationError as error:
            log_exception(
                error, extra={"context": "User provided an invalid slash command"}
            )
            # Handle invalid slash commands with a user-friendly message
            error_message = AIMessage(content=str(error))
            ui_chat_log = UiChatLog(
                message_type=MessageTypeEnum.AGENT,
                message_sub_type=None,
                content=str(error),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.FAILURE,
                correlation_id=None,
                tool_info=None,
                additional_context=None,
                message_id=f"error-{uuid4()!s}",
            )
            ui_chat_logs = self._append_optimizer_ui_logs(
                [ui_chat_log], optimizer_ui_logs
            )
            return {
                "conversation_history": {self.name: [error_message]},
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": ui_chat_logs,
            }
        except Exception as error:
            log_exception(error, extra={"context": "Error processing chat agent"})
            return self._create_error_response(error)
