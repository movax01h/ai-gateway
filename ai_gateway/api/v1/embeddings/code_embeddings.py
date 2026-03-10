from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.embeddings.typing import EmbeddingsRequest, EmbeddingsResponse
from ai_gateway.async_dependency_resolver import get_prompt_registry
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.v2.embedding_litellm import EmbeddingBadRequestError
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
@feature_category(GitLabFeatureCategory.GLOBAL_SEARCH)
async def code_embeddings(
    _request: Request,
    payload: EmbeddingsRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    request_log.debug("embeddings input:", payload=payload)

    _validate_model_metadata_payload(payload.model_metadata)

    prompt = prompt_registry.get_on_behalf(
        user=current_user,
        prompt_id=CODE_EMBEDDINGS_PROMPT_ID,
        internal_event_category=__name__,
    )

    try:
        message = await prompt.ainvoke(input={"contents": payload.contents})
    except EmbeddingBadRequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
        ),
        predictions=[
            EmbeddingsResponse.Prediction(
                embedding=p["embedding"],
                index=p["index"],
            )
            for p in predictions
        ],
    )


def _validate_model_metadata_payload(model_metadata: EmbeddingsRequest.ModelMetadata):
    if model_metadata.provider != KindModelProvider.GITLAB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Gitlab-operated models are supported in this endpoint.",
        )
