from datetime import datetime, timezone
from typing import Any, List, Union

from anthropic import APIStatusError
from dependency_injector.wiring import Provide, inject
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.runnables import Runnable

from ai_gateway.container import ContainerApplication
from duo_workflow_service.entities.event import WorkflowEvent, WorkflowEventType
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorHandler
from duo_workflow_service.gitlab.events import get_event
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.structured_logging import _workflow_id
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from duo_workflow_service.tools import Toolset
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum


class Agent:  # pylint: disable=too-many-instance-attributes
    name: str

    _model: Runnable
    _goal: str
    _system_prompt: str
    _workflow_id: str
    _http_client: GitlabHttpClient
    _toolset: Toolset
    _check_events: bool
    _internal_event_client: InternalEventsClient

    @inject
    def __init__(
        self,
        *,
        goal: str,
        model: BaseChatModel,
        name: str,
        system_prompt: str,
        toolset: Toolset,
        workflow_id: str,
        http_client: GitlabHttpClient,
        workflow_type: CategoryEnum,
        check_events: bool = True,
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
    ):
        self._model = model.bind_tools(toolset.bindable)
        self._goal = goal
        self._system_prompt = system_prompt
        self.name = name
        self._error_handler = ModelErrorHandler()
        self._workflow_id = workflow_id
        self._http_client = http_client
        self._workflow_type = workflow_type
        self._toolset = toolset
        self._check_events = check_events
        self._internal_event_client = internal_event_client

    async def run(self, state: DuoWorkflowStateType) -> dict:
        with duo_workflow_metrics.time_compute(
            operation_type=f"{self.name}_processing"
        ):
            updates: dict[str, Any] = {
                "handover": [],
            }
            model_completion: list[MessageLikeRepresentation]

            if self._check_events:
                event: Union[WorkflowEvent, None] = await get_event(
                    self._http_client, self._workflow_id, False
                )

                if event and event["event_type"] == WorkflowEventType.STOP:
                    return {"status": WorkflowStatusEnum.CANCELLED}

            if self.name in state["conversation_history"]:
                model_completion = await self._model_completion(
                    state["conversation_history"][self.name]
                )
                updates["conversation_history"] = {self.name: model_completion}
            else:
                messages = self._conversation_preamble(state)
                model_completion = await self._model_completion(messages)
                updates["conversation_history"] = {
                    self.name: [*messages, *model_completion]
                }

            return {
                **updates,
                **self._respond_to_human(state, model_completion),
            }

    def _respond_to_human(self, state, model_completion) -> dict[str, Any]:
        if not isinstance(model_completion[0], AIMessage):
            return {}

        last_human_input = state.get("last_human_input")
        if (
            isinstance(last_human_input, dict)
            and last_human_input.get("event_type") == WorkflowEventType.MESSAGE
        ):
            content = self._parse_model_content(model_completion[0].content)
            return {
                "ui_chat_log": ([self._create_ui_chat_log(content)] if content else []),
                "last_human_input": None,
            }

        return {}

    async def _model_completion(
        self, messages: list[BaseMessage]
    ) -> list[MessageLikeRepresentation]:
        while True:
            try:
                approximate_token_count = ApproximateTokenCounter(
                    self.name
                ).count_tokens(messages)

                model_name = getattr(self._model, "model_name", "unknown")
                request_type = f"{self.name}_completion"
                with duo_workflow_metrics.time_llm_request(
                    model=model_name, request_type=request_type
                ):
                    response = await self._model.ainvoke(messages)

                self._track_tokens_data(response, approximate_token_count)
                duo_workflow_metrics.count_llm_response(
                    model=model_name,
                    request_type=request_type,
                    stop_reason=(
                        response.response_metadata.get("stop_reason")
                        if response.response_metadata
                        else None
                    ),
                )
                return [response]
            except APIStatusError as e:
                error_message = str(e)
                status_code = e.response.status_code
                model_error = ModelError(
                    error_type=self._error_handler.get_error_type(status_code),
                    status_code=status_code,
                    message=error_message,
                )
                model_class = getattr(self._model, "bound", None)
                duo_workflow_metrics.count_model_completion_errors(
                    model=getattr(self._model, "model_name", "unknown"),
                    provider=type(model_class).__name__ if model_class else "unknown",
                    http_status=str(status_code),
                    error_type=self._error_handler.get_error_type(status_code),
                )

                await self._error_handler.handle_error(model_error)

    def _track_tokens_data(self, message, estimated):
        usage_metadata = message.usage_metadata if message.usage_metadata else {}

        additional_properties = InternalEventAdditionalProperties(
            label=self.name,
            property=EventPropertyEnum.WORKFLOW_ID.value,
            value=_workflow_id.get(),
            input_tokens=usage_metadata.get("input_tokens"),
            output_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
            estimated_input_tokens=estimated,
        )
        self._internal_event_client.track_event(
            event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
            additional_properties=additional_properties,
            category=self._workflow_type.value,
        )

    def _parse_model_content(self, content: str | List) -> str | None:
        if isinstance(content, str):
            return content

        if isinstance(content, List) and all(isinstance(item, str) for item in content):
            return "\n".join(content)

        return next(
            (
                item.get("text")
                for item in content
                if isinstance(item, dict) and item.get("text", False)
            ),
            None,
        )

    def _create_ui_chat_log(self, content: str) -> UiChatLog:
        return UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=None,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
        )

    def _conversation_preamble(self, state: DuoWorkflowStateType) -> list[BaseMessage]:
        conversation_preamble: list[BaseMessage] = [
            SystemMessage(content=self._system_prompt)
        ]

        if state.get("handover"):  # type: ignore
            conversation_preamble.extend(
                [
                    HumanMessage(
                        content="The steps towards goal accomplished so far are as follow:"
                    ),
                    *state.get("handover"),  # type: ignore
                ]
            )

        conversation_preamble.append(
            HumanMessage(content=f"Your goal is: {self._goal}")
        )

        return conversation_preamble
