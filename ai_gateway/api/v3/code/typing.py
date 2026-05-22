from typing import Annotated, Any, List, Literal, Optional, Union

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ai_gateway.code_suggestions import ModelProvider
from ai_gateway.code_suggestions.handler import (
    CodeEditorComponents,
    CompletionResponse,
    ModelMetadata,
    ResponseMetadataBase,
    StreamHandler,
    StreamModelEngine,
    StreamSuggestionsResponse,
)
from ai_gateway.models import Message
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.litellm import KindGitLabModel

__all__ = [
    "CodeEditorComponents",
    "CompletionRequest",
    "CompletionResponse",
    "EditorContentCompletionPayload",
    "EditorContentGenerationPayload",
    "ModelMetadata",
    "ResponseMetadataBase",
    "StreamHandler",
    "StreamModelEngine",
    "StreamSuggestionsResponse",
]


class MetadataBase(BaseModel):
    source: Annotated[str, StringConstraints(max_length=255)]
    version: Annotated[str, StringConstraints(max_length=255)]


class EditorContentPayload(BaseModel):
    # Opt out protected namespace "model_" (https://github.com/pydantic/pydantic/issues/6322).
    model_config = ConfigDict(protected_namespaces=())

    file_name: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=255)
    ] = Field(examples=["example.py"])
    content_above_cursor: Annotated[str, StringConstraints(max_length=100000)] = Field(
        examples=["def hello_world():\n    print("]
    )
    content_below_cursor: Annotated[str, StringConstraints(max_length=100000)] = Field(
        examples=[""]
    )
    language_identifier: Optional[Annotated[str, StringConstraints(max_length=255)]] = (
        Field(None, examples=["python"])
    )
    model_provider: Optional[
        Literal[
            ModelProvider.VERTEX_AI, ModelProvider.ANTHROPIC, KindModelProvider.AMAZON_Q
        ]
    ] = None
    model_name: Optional[str] = Field(
        None, examples=[KindGitLabModel.CODESTRAL_2508_VERTEX]
    )
    stream: Optional[bool] = False
    role_arn: Optional[str] = None


class EditorContentCompletionPayload(EditorContentPayload):
    choices_count: Optional[int] = 0
    prompt: Optional[str | list[Message]] = Field(
        None, examples=["Complete the function"]
    )


class EditorContentGenerationPayload(EditorContentPayload):
    prompt: Optional[Annotated[str, StringConstraints(max_length=400000)]] = None
    prompt_id: Optional[str] = None
    prompt_enhancer: Optional[dict[str, Any]] = None
    prompt_version: Optional[str] = None


class CodeEditorCompletion(BaseModel):
    type: Literal[CodeEditorComponents.COMPLETION]
    payload: EditorContentCompletionPayload
    metadata: Optional[MetadataBase] = None


class CodeEditorGeneration(BaseModel):
    type: Literal[CodeEditorComponents.GENERATION]
    payload: EditorContentGenerationPayload
    metadata: Optional[MetadataBase] = None


class CodeContextPayload(BaseModel):
    type: Annotated[str, StringConstraints(max_length=1024)]
    name: Annotated[str, StringConstraints(max_length=1024)]
    content: Annotated[str, StringConstraints(max_length=100000)]


class CodeContext(BaseModel):
    type: Literal[CodeEditorComponents.CONTEXT]
    payload: CodeContextPayload
    metadata: Optional[MetadataBase] = None


PromptComponent = Annotated[
    Union[CodeEditorCompletion, CodeEditorGeneration, CodeContext],
    Body(discriminator="type"),
]


class CompletionRequest(BaseModel):
    prompt_components: Annotated[
        List[PromptComponent], Field(min_length=1, max_length=100)
    ]
