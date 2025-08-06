# pylint: disable=direct-environment-variable-reference

import json
import os
from typing import Any, List, Type

from google.cloud import discoveryengine
from pydantic import BaseModel, Field

from ai_gateway.searches import VertexAISearch
from duo_workflow_service.interceptors.gitlab_version_interceptor import gitlab_version
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

DEFAULT_PAGE_SIZE = 4
DEFAULT_GL_VERSION = "18.0.0"


class SearchInput(BaseModel):
    search: str = Field(description="The search term")


def _get_env_var(var_name: str) -> str:
    value = os.environ.get(var_name)
    if value is None:
        error_message = f"{var_name} environment variable is not set"
        raise RuntimeError(error_message)
    return value


class DocumentationSearch(DuoBaseTool):
    name: str = "gitlab_documentation_search"
    description: str = """Find GitLab documentations,
    useful for answering questions concerning GitLab and its features, e.g.:
    projects, groups, issues, merge requests, epics, milestones, labels, CI/CD pipelines, git repositories, and more.

    Parameters:
    - search: The search term (required)

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_documentation_search',
        'type': 'tool_use'
        'input': {
            'search': 'How do I set up a new project?',
        },
    }
    """
    args_schema: Type[BaseModel] = SearchInput

    async def _arun(self, search: str) -> str:
        try:
            results = await self._fetch_documentation(search)
            return json.dumps({"search_results": results})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _fetch_documentation(self, query: str) -> List[dict]:
        client = discoveryengine.SearchServiceAsyncClient()

        gl_version = gitlab_version.get() or DEFAULT_GL_VERSION

        # TODO: obtain project from Pydantic Setting
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1188
        project = _get_env_var("AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT")
        fallback_datastore_version = _get_env_var(
            "AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION"
        )

        search = VertexAISearch(
            client=client,
            project=project,
            fallback_datastore_version=fallback_datastore_version,
        )
        search_results = await search.search_with_retry(
            query=query, gl_version=gl_version, page_size=DEFAULT_PAGE_SIZE
        )
        return search_results

    def format_display_message(
        self, args: SearchInput, _tool_response: Any = None
    ) -> str:
        return f"Searching GitLab documentation for: '{args.search}'"
