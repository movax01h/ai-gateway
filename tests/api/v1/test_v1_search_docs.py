from time import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.search.typing import (
    DEFAULT_PAGE_SIZE,
    SearchResponse,
    SearchResponseDetails,
    SearchResponseMetadata,
    SearchResult,
)


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(scopes=["documentation_search"]),
    )


@pytest.fixture(name="request_body")
def request_body_fixture():
    return {
        "type": "search-docs",
        "metadata": {"source": "GitLab EE", "version": "17.0.0"},
        "payload": {"query": "What is gitlab mission?"},
    }


@pytest.fixture(name="search_results")
def search_results_fixture():
    return [
        {
            "id": "doc_id_1",
            "content": "GitLab's mission is to make software development easier and more efficient.",
            "metadata": {
                "source": "GitLab Docs",
                "version": "17.0.0",
                "source_url": "https://docs.gitlab.com/ee/foo",
            },
        }
    ]


@pytest.mark.asyncio
async def test_success(
    mock_client: TestClient,
    mock_track_internal_event,
    request_body: dict,
    search_results: dict,
):
    from ai_gateway.searches.typing import SearchResult

    time_now = time()
    search_result_objects = [
        SearchResult(
            id=result["id"],
            content=result["content"],
            metadata=result["metadata"],
        )
        for result in search_results
    ]
    with patch(
        "ai_gateway.searches.search.VertexAISearch.search_with_retry",
        return_value=search_result_objects,
    ) as mock_search_with_retry:
        with patch("time.time", return_value=time_now):
            response = mock_client.post(
                "/search/gitlab-docs",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json=request_body,
            )

    assert response.status_code == 200

    expected_response = SearchResponse(
        response=SearchResponseDetails(
            results=[
                SearchResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                )
                for result in search_results
            ]
        ),
        metadata=SearchResponseMetadata(
            provider="vertex-ai",
            timestamp=int(time_now),
        ),
    )

    assert response.json() == expected_response.model_dump()

    mock_search_with_retry.assert_called_once_with(
        query=request_body["payload"]["query"],
        gl_version=request_body["metadata"]["version"],
        page_size=DEFAULT_PAGE_SIZE,
    )

    mock_track_internal_event.assert_called_once_with(
        "request_documentation_search",
        category="ai_gateway.api.v1.search.docs",
    )


@pytest.mark.asyncio
async def test_missing_param(
    mock_client: TestClient,
):
    request_body = {
        "type": "search-docs",
        "metadata": {"source": "GitLab EE", "version": "17.0.0"},
        "payload": {},
    }

    response = mock_client.post(
        "/search/gitlab-docs",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
        },
        json=request_body,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_authentication(
    mock_client: TestClient,
    request_body: dict,
):
    response = mock_client.post(
        "/search/gitlab-docs",
        json=request_body,
    )

    assert response.status_code == 401
