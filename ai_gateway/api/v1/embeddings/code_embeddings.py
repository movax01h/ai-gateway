from json import JSONDecodeError
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.embeddings.typing import (
    EMBEDDING_MODEL_NAME,
    EmbeddingsRequest,
    EmbeddingsResponse,
)
from ai_gateway.async_dependency_resolver import get_prompt_registry
from ai_gateway.model_metadata import TypeModelMetadata, create_model_metadata
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.v2.embedding_litellm import (
    EmbeddingAuthenticationError,
    EmbeddingBadRequestError,
    EmbeddingRateLimitError,
)
from ai_gateway.prompts.base import BasePromptRegistry
from ai_gateway.structured_logging import get_request_logger
from lib.context.auth import StarletteUser, get_current_user

__all__ = [
    "router",
]


request_log = get_request_logger("embeddings")

router = APIRouter()


CODE_EMBEDDINGS_PROMPT_ID = "embeddings_code"


@router.post("/code_embeddings")
@router.post("/code_embeddings/index")
@feature_category(GitLabFeatureCategory.GLOBAL_SEARCH)
async def code_embeddings_index(
    request: Request,
    payload: EmbeddingsRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    return await _generate_code_embeddings(
        _request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
    )


@router.post("/code_embeddings/search")
@feature_category(GitLabFeatureCategory.GLOBAL_SEARCH)
async def code_embeddings_search(
    request: Request,
    payload: EmbeddingsRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    return await _generate_code_embeddings(
        _request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
    )


async def _generate_code_embeddings(
    _request: Request,
    payload: EmbeddingsRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    request_log.debug("embeddings input:", payload=payload)

    model_metadata = _validate_and_get_model_metadata(payload.model_metadata)

    try:
        prompt = prompt_registry.get_on_behalf(
            user=current_user,
            prompt_id=CODE_EMBEDDINGS_PROMPT_ID,
            model_metadata=model_metadata,
            internal_event_category=__name__,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    try:
        input: dict[str, Any] = {"contents": payload.contents}
        if payload.dimensions:
            input["dimensions"] = payload.dimensions
        message = await prompt.ainvoke(input=input)
    except EmbeddingBadRequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except EmbeddingRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except EmbeddingAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    # Explicitly type `predictions` as a list of dicts since we know
    #   it contains a list of dicts from the embedding response
    # The `type: ignore[assignment]` comment suppresses mypy's assignment error
    #   because message.content is typed as `str | list[str | dict]`
    predictions: list[dict[str, Any]] = message.content  # type: ignore[assignment]

    return EmbeddingsResponse(
        model=EmbeddingsResponse.Model(
            engine=prompt.model_provider,
            name=prompt.model_name,
            identifier=payload.model_metadata.identifier,
        ),
        predictions=[
            EmbeddingsResponse.Prediction(
                embedding=p["embedding"],
                index=p["index"],
            )
            for p in predictions
        ],
    )


def _validate_and_get_model_metadata(
    payload_model_metadata: EmbeddingsRequest.ModelMetadata,
) -> TypeModelMetadata:
    _validate_allowed_providers(payload_model_metadata)
    _validate_litellm_provider_payload(payload_model_metadata)

    # We need to explicitly build the model_metadata to be passed to the Prompt Registry
    #   in order to return a 422 error response if the model_metadata params is invalid
    # Without this, the Prompt Registry will get the model_metadata from
    #   `current_model_metadata_context`, which is simply not set if params is invalid.
    # If model_metadata is not set when calling `get_on_behalf` AND not set in the context,
    #   Prompt Registry then falls back to the default model, which is something that should
    #   not be allowed for embeddings.
    try:
        model_metadata = create_model_metadata(data=payload_model_metadata.model_dump())

        if model_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No model metadata created",
            )

        return model_metadata
    except (ValueError, JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error creating the model_metadata: {str(e)}",
        )


def _validate_allowed_providers(model_metadata: EmbeddingsRequest.ModelMetadata):
    allowed_providers = [KindModelProvider.GITLAB, KindModelProvider.LITELLM]
    if model_metadata.provider not in allowed_providers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Allowed providers are: {'|'.join([p.value for p in allowed_providers])}.",
        )


def _validate_litellm_provider_payload(model_metadata: EmbeddingsRequest.ModelMetadata):
    if model_metadata.provider != KindModelProvider.LITELLM:
        return

    if not model_metadata.name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model `name` must be set when using `litellm` provider.",
        )

    if model_metadata.name != EMBEDDING_MODEL_NAME:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model `name` must be '{EMBEDDING_MODEL_NAME}' when using `litellm` provider.",
        )

    if not model_metadata.endpoint:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model `endpoint` must be set when using `litellm` provider.",
        )
