import time
from typing import Annotated

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.search.typing import (
    SearchRequest,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
)
from ai_gateway.async_dependency_resolver import (
    get_internal_event_client,
    get_search_factory_provider,
)
from ai_gateway.searches import Searcher
from lib.context import StarletteUser, get_current_user
from lib.internal_events import InternalEventsClient

__all__ = [
    "router",
]

router = APIRouter()


@router.post(
    "/gitlab-docs", response_model=SearchResponse, status_code=status.HTTP_200_OK
)
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def docs(
    request: Request,  # pylint: disable=unused-argument
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    search_request: SearchRequest,
    search_factory: Annotated[Factory[Searcher], Depends(get_search_factory_provider)],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
):
    if not current_user.can(GitLabUnitPrimitive.DOCUMENTATION_SEARCH):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to search documentations",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.DOCUMENTATION_SEARCH}",
        category=__name__,
    )

    payload = search_request.payload

    search_params = {
        "query": payload.query,
        "page_size": payload.page_size,
        "gl_version": search_request.metadata.version,
    }

    searcher = search_factory()

    results = await searcher.search_with_retry(**search_params)

    return SearchResponse(
        response=SearchResponseDetails(
            results=results,
        ),
        metadata=SearchResponseMetadata(
            provider=searcher.provider(),
            timestamp=int(time.time()),
        ),
    )
