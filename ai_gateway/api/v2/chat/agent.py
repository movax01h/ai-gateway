from datetime import datetime
from typing import Annotated, AsyncIterator, Tuple

from dependency_injector import providers
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import (
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)
from starlette.responses import StreamingResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import X_GITLAB_VERSION_HEADER
from ai_gateway.api.v2.chat.typing import AgentRequest
from ai_gateway.async_dependency_resolver import (
    get_container_application,
    get_internal_event_client,
    get_prompt_registry,
)
from ai_gateway.chat.agents import (
    AgentStep,
    AgentToolAction,
    Message,
    ReActAgentInputs,
    TypeAgentEvent,
)
from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.prompts import BasePromptRegistry, Prompt

__all__ = [
    "router",
    "create_event_stream",
]

from ai_gateway.models import Role
from ai_gateway.structured_logging import get_request_logger

request_log = get_request_logger("chat")

router = APIRouter()


async def get_gl_agent_remote_executor_factory():
    yield get_container_application().chat.gl_agent_remote_executor_factory


def authorize_additional_context(
    current_user: StarletteUser,
    agent_request: AgentRequest,
    internal_event_client: InternalEventsClient,
):
    if agent_request.messages:
        for message in agent_request.messages:
            if message.additional_context:
                for additional_context in message.additional_context:
                    unit_primitive = GitLabUnitPrimitive[
                        f"include_{additional_context.category}_context".upper()
                    ]
                    if current_user.can(unit_primitive):
                        internal_event_client.track_event(
                            f"request_{unit_primitive}",
                            category=__name__,
                        )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Unauthorized to access {unit_primitive}",
                        )


def get_agent(
    current_user: StarletteUser,
    agent_request: AgentRequest,
    prompt_registry: BasePromptRegistry,
) -> Prompt:
    try:
        prompt = prompt_registry.get_on_behalf(
            current_user,
            "chat/react",
            None,
            agent_request.model_metadata,
            __name__,
        )
    except WrongUnitPrimitives:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access duo chat",
        )

    return prompt


def _build_scratchpad_from_request(
    agent_request: AgentRequest, last_message: Message
) -> list[AgentStep]:
    if agent_request.options:
        return [
            AgentStep(
                action=AgentToolAction(
                    thought=step.thought.replace("\\_", "_"),
                    tool=step.tool.replace("\\_", "_"),
                    tool_input=step.tool_input,
                ),
                observation=step.observation,
            )
            for step in agent_request.options.agent_scratchpad.steps
        ]

    # Empty content in the assistant's last message indicates an active in-progress query,
    # it is used to extract agent_scratchpad and pass it to the model.
    if last_message.role == Role.ASSISTANT and last_message.content is None:
        return [
            AgentStep(
                action=AgentToolAction(
                    type=step.action.type,
                    thought=step.action.thought.replace("\\_", "_"),
                    tool=step.action.tool.replace("\\_", "_"),
                    tool_input=step.action.tool_input,
                ),
                observation=step.observation,
            )
            for step in (last_message.agent_scratchpad or [])
            if step.action
        ]

    return []


async def create_event_stream(
    current_user: StarletteUser,
    agent_request: AgentRequest,
    agent: Prompt,
    gl_agent_remote_executor_factory: providers.Factory[
        GLAgentRemoteExecutor[ReActAgentInputs, TypeAgentEvent]
    ],
    gl_version: str,
    agent_scratchpad: list[AgentStep],
) -> Tuple[ReActAgentInputs, AsyncIterator[TypeAgentEvent]]:

    gl_agent_remote_executor = gl_agent_remote_executor_factory(agent=agent)
    gl_agent_remote_executor.on_behalf(
        current_user, gl_version, agent_request.model_metadata
    )
    inputs = ReActAgentInputs(
        messages=agent_request.messages,
        agent_scratchpad=agent_scratchpad,
        unavailable_resources=agent_request.unavailable_resources,
        current_date=datetime.now().strftime("%A, %B, %d, %Y"),
    )

    stream_events = gl_agent_remote_executor.stream(inputs=inputs)
    return inputs, stream_events


@router.post("/agent")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def chat(
    request: Request,
    agent_request: AgentRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    gl_agent_remote_executor_factory: Annotated[
        providers.Factory[GLAgentRemoteExecutor[ReActAgentInputs, TypeAgentEvent]],
        Depends(get_gl_agent_remote_executor_factory),
    ],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
):
    agent = get_agent(current_user, agent_request, prompt_registry)

    authorize_additional_context(current_user, agent_request, internal_event_client)

    async def _stream_handler(stream_events: AsyncIterator[TypeAgentEvent]):
        async for event in stream_events:
            yield f"{event.dump_as_response()}\n"

    last_message = agent_request.messages[-1]

    scratchpad = _build_scratchpad_from_request(agent_request, last_message)

    gl_version = request.headers.get(X_GITLAB_VERSION_HEADER, "")

    stream_result: Tuple[ReActAgentInputs, AsyncIterator[TypeAgentEvent]] = (
        await create_event_stream(
            current_user=current_user,
            agent_request=agent_request,
            agent=agent,
            gl_agent_remote_executor_factory=gl_agent_remote_executor_factory,
            gl_version=gl_version,
            agent_scratchpad=scratchpad,
        )
    )

    inputs, stream_events = stream_result

    request_log.info("Request to V2 Chat Agent", source=__name__, inputs=inputs)

    # When StreamingResponse is returned, clients get 200 even if there was an error during the process.
    # This is because the status code is returned before the actual process starts,
    # and there is no way to tell clients that the status code was changed after the streaming started.
    # Ref: https://github.com/encode/starlette/discussions/1739#discussioncomment-3094935.
    # If an exception is raised during the process, you will see `exception_message` field in the access log.
    return StreamingResponse(
        _stream_handler(stream_events), media_type="application/x-ndjson; charset=utf-8"
    )
