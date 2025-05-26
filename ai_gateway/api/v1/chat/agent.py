from time import time
from typing import Annotated, AsyncIterator, Tuple

from dependency_injector import providers
from dependency_injector.providers import Factory, FactoryAggregate
from fastapi import APIRouter, Depends, Request, status
from gitlab_cloud_connector import GitLabUnitPrimitive
from starlette.responses import StreamingResponse

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import track_metadata
from ai_gateway.api.middleware import X_GITLAB_VERSION_HEADER
from ai_gateway.api.v1.chat.auth import ChatInvokable, authorize_with_unit_primitive
from ai_gateway.api.v1.chat.typing import (
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
)
from ai_gateway.api.v2.chat.agent import (
    create_event_stream,
    get_agent,
    get_gl_agent_remote_executor_factory,
)
from ai_gateway.api.v2.chat.typing import AgentRequest
from ai_gateway.async_dependency_resolver import (
    get_chat_anthropic_claude_factory_provider,
    get_chat_litellm_factory_provider,
    get_internal_event_client,
    get_prompt_registry,
)
from ai_gateway.chat.agents import ReActAgentInputs, TypeAgentEvent
from ai_gateway.chat.agents.typing import Message
from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.models.base_chat import Role
from ai_gateway.prompts import BasePromptRegistry

__all__ = [
    "router",
]

router = APIRouter()

CHAT_INVOKABLES = [
    ChatInvokable(name="explain_code", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(name="write_tests", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(name="refactor_code", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
    ChatInvokable(
        name="explain_vulnerability",
        unit_primitive=GitLabUnitPrimitive.EXPLAIN_VULNERABILITY,
    ),
    ChatInvokable(
        name="summarize_comments",
        unit_primitive=GitLabUnitPrimitive.SUMMARIZE_COMMENTS,
    ),
    ChatInvokable(
        name="troubleshoot_job",
        unit_primitive=GitLabUnitPrimitive.TROUBLESHOOT_JOB,
    ),
    # Deprecated. Added for backward compatibility.
    # Please, refer to `v2/chat/agent` for additional details.
    ChatInvokable(name="agent", unit_primitive=GitLabUnitPrimitive.DUO_CHAT),
]

path_unit_primitive_map = {ci.name: ci.unit_primitive for ci in CHAT_INVOKABLES}


def convert_v1_to_v2_inputs(chat_request: ChatRequest) -> AgentRequest:
    """
    Adapts a v1 ChatRequest into a v2 AgentRequest.
    If the payload content is a string, wrap it in a Message.
    """
    prompt_component = chat_request.prompt_components[0]
    payload = prompt_component.payload

    if isinstance(payload.content, str):
        messages = [Message(role=Role.USER, content=payload.content)]
    else:
        system_message_buffer = []
        messages = []

        for message_data in payload.content:
            role = message_data.role
            content = message_data.content

            if role == "system":
                system_message_buffer.append(content)
            elif role == "user" and system_message_buffer:
                full_user_content = ""
                full_user_content += "\\n".join(system_message_buffer) + "\\n\\n"
                system_message_buffer = []
                full_user_content += content

                message = Message(role=Role.USER, content=full_user_content)
                messages.append(message)
            else:
                message = Message(**message_data.model_dump())
                messages.append(message)

    return AgentRequest(
        messages=messages,
        options=None,
    )


@router.post(
    "/{chat_invokable}",
    response_model=ChatResponse,
    deprecated=True,
    summary="Deprecated endpoint",
    description="This endpoint is deprecated and will be removed https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/825",
    status_code=status.HTTP_200_OK,
)
@authorize_with_unit_primitive("chat_invokable", chat_invokables=CHAT_INVOKABLES)
@track_metadata("chat_invokable", mapping=path_unit_primitive_map)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_invokable: str,  # pylint: disable=unused-argument
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    anthropic_claude_factory: Annotated[  # pylint: disable=unused-argument
        FactoryAggregate, Depends(get_chat_anthropic_claude_factory_provider)
    ],
    litellm_factory: Annotated[  # pylint: disable=unused-argument
        Factory, Depends(get_chat_litellm_factory_provider)
    ],
    internal_event_client: Annotated[  # pylint: disable=unused-argument
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    gl_agent_remote_executor_factory: Annotated[
        providers.Factory[GLAgentRemoteExecutor[ReActAgentInputs, TypeAgentEvent]],
        Depends(get_gl_agent_remote_executor_factory),
    ],
):

    agent_request = convert_v1_to_v2_inputs(chat_request)
    payload = chat_request.prompt_components[0].payload

    agent = get_agent(current_user, prompt_registry)

    gl_version = request.headers.get(X_GITLAB_VERSION_HEADER, "")

    stream_result: Tuple[ReActAgentInputs, AsyncIterator[TypeAgentEvent]] = (
        await create_event_stream(
            current_user=current_user,
            agent_request=agent_request,
            agent=agent,
            gl_agent_remote_executor_factory=gl_agent_remote_executor_factory,
            gl_version=gl_version,
            agent_scratchpad=[],
        )
    )

    _, stream_events = stream_result

    if chat_request.stream:

        async def stream_handler():
            async for event in stream_events:
                # Transform each event
                yield (
                    event.text if hasattr(event, "text") else event.dump_as_response()
                )

        return StreamingResponse(stream_handler(), media_type="text/event-stream")

    final_text = ""
    async for event in stream_events:
        if hasattr(event, "text"):
            final_text += event.text
    return ChatResponse(
        response=final_text,
        metadata=ChatResponseMetadata(
            provider=payload.provider,
            model=None,
            timestamp=int(time()),
        ),
    )
