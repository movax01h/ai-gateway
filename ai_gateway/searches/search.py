import re
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, override

from fastapi import HTTPException, status
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import discoveryengine
from google.protobuf.json_format import MessageToDict

from ai_gateway.models import ModelAPIError
from ai_gateway.searches.typing import SearchResult
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import log_exception

SEARCH_APP_NAME = "gitlab-docs"

request_log = get_request_logger("search")


def _convert_version(version: str) -> str:
    # Regex to match the major and minor version numbers
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match:
        # Extract major and minor parts and join them with a hyphen
        major, minor = match.groups()
        return f"{major}-{minor}"

    raise ValueError(f"Invalid version: {version}")


def _get_data_store_id(gl_version: str) -> str:
    data_store_version = _convert_version(gl_version)

    return f"{SEARCH_APP_NAME}-{data_store_version}"


# TODO: Both Vertex Model API and Search API use the same error hierarchy under the hood via
# google-api-core (https://googleapis.dev/python/google-api-core/latest/). We would need to
# extract ModelAPIError to a common module that can be shared between /searches and /models
# module.
class VertexAPISearchError(ModelAPIError):
    @classmethod
    def from_exception(cls, ex: GoogleAPIError):
        message = f"Vertex Search API error: {type(ex).__name__}"

        if hasattr(ex, "message"):
            message = f"{message} {ex.message}"

        return cls(message, errors=(ex,))


class DataStoreNotFound(Exception):
    def __init__(self, message="", input=""):
        super().__init__(message)
        self.input = input


class Searcher:
    async def search_with_retry(self, *args, **kwargs) -> List[SearchResult]:
        return await self.search(*args, **kwargs)

    @abstractmethod
    async def search(
        self,
        query: str,
        gl_version: str,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def provider(self):
        pass

    def log_search_results(
        self,
        query: str,
        page_size: int,
        gl_version: str,
        results: list[SearchResult],
        token_count: int | None = None,
    ):
        """Log details of the search results."""
        log_data = {
            "query": query,
            "page_size": page_size,
            "gl_version": gl_version,
            "results_metadata": [res.metadata for res in results],
            "filtered_results": len(results),
            "class": self.__class__.__name__,
            "total_tokens": token_count,
        }

        request_log.info("Search completed", **log_data)

    def dump_results(self, search_results: List[SearchResult]) -> list[dict]:
        return [result.model_dump() for result in search_results]


class VertexAISearch(Searcher):
    def __init__(
        self,
        client: discoveryengine.SearchServiceAsyncClient,
        project: str,
        fallback_datastore_version: str,
        *_args: Any,
        **_kwargs: Any,
    ):
        self.client = client
        self.project = project
        self.fallback_datastore_version = fallback_datastore_version

    @override
    async def search_with_retry(self, *args, **kwargs):
        try:
            try:
                return await self.search(*args, **kwargs)
            except DataStoreNotFound as ex:
                log_exception(ex, extra={"input": ex.input})

            # Retry with the fallback datastore version
            kwargs["gl_version"] = self.fallback_datastore_version

            try:
                return await self.search(*args, **kwargs)
            except DataStoreNotFound as ex:
                log_exception(ex, extra={"input": ex.input})
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                )
        except VertexAPISearchError as ex:
            log_exception(ex)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vertex API Search Error.",
            )

    @override
    async def search(
        self,
        query: str,
        gl_version: str,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[SearchResult]:
        try:
            data_store_id = _get_data_store_id(gl_version)
        except ValueError as ex:
            raise DataStoreNotFound(str(ex), input=gl_version)

        # The full resource name of the searches engine serving config
        # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
        serving_config = self.client.serving_config_path(
            project=self.project,
            location="global",
            data_store=data_store_id,
            serving_config="default_config",
        )

        # Refer to the `SearchRequest` reference for all supported fields:
        # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=page_size,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
            **kwargs,
        )

        try:
            response = await self.client.search(request)
        except NotFound:
            raise DataStoreNotFound("Data store not found", input=data_store_id)
        except GoogleAPIError as ex:
            raise VertexAPISearchError.from_exception(ex)

        parsed_response: List[Dict[Any, Any]] = self._parse_response(
            MessageToDict(response._pb)
        )

        results = [
            SearchResult(
                id=res["id"],
                content=res["content"],
                metadata=res["metadata"],
            )
            for res in parsed_response
        ]

        self.log_search_results(query, page_size, gl_version, results)

        return results

    @override
    def provider(self):
        return "vertex-ai"

    def _parse_response(self, response) -> list[dict]:
        results = []

        if "results" in response:
            for r in response["results"]:
                search_result = {
                    "id": r["document"]["id"],
                    "content": r["document"]["structData"]["content"],
                    "metadata": r["document"]["structData"]["metadata"],
                }
                results.append(search_result)

        return results

    @override
    def dump_results(self, search_results: List[SearchResult]) -> list[dict]:
        snippets_grouped = defaultdict(list)
        pages = {}

        # Restructure the output data to make the LLM focus on useful data only
        for result in search_results:
            md5 = result.metadata["md5sum"]

            if md5 not in pages:
                pages[md5] = {
                    "source_url": result.metadata["source_url"],
                    "source_title": result.metadata["title"],
                }

            snippets_grouped[md5].append(result.content)

        return [
            {
                "relevant_snippets": snippets,
                **pages[md5],
            }
            for md5, snippets in snippets_grouped.items()
        ]
