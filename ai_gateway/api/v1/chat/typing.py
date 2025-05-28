from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, StringConstraints

from ai_gateway.models import (
    KindAnthropicModel,
    KindLiteLlmModel,
    KindModelProvider,
    Message,
)

__all__ = [
    "ChatRequest",
    "ChatResponseMetadata",
    "ChatResponse",
]


class PromptMetadata(BaseModel):
    source: Annotated[str, StringConstraints(max_length=100)]
    version: Annotated[str, StringConstraints(max_length=100)]


class PromptPayload(BaseModel):
    content: Union[
        Annotated[str, StringConstraints(max_length=400000)],
        Annotated[list[Message], Field(min_length=1, max_length=100)],
    ]
    provider: Optional[
        Literal[KindModelProvider.ANTHROPIC, KindModelProvider.LITELLM]
    ] = None
    model: Optional[KindAnthropicModel | KindLiteLlmModel] = (
        KindAnthropicModel.CLAUDE_3_5_SONNET_V2
    )


class PromptComponent(BaseModel):
    type: Annotated[str, StringConstraints(max_length=100)]
    metadata: PromptMetadata
    payload: PromptPayload


# We expect only a single prompt component in the first iteration.
# Details: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/135837#note_1642865693
class ChatRequest(BaseModel):
    prompt_components: Annotated[
        List[PromptComponent], Field(min_length=1, max_length=1)
    ]
    stream: Optional[bool] = False


class ChatResponseMetadata(BaseModel):
    provider: Optional[str]
    model: Optional[str]
    timestamp: int


class ChatResponse(BaseModel):
    response: str
    metadata: ChatResponseMetadata
