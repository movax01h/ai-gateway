# pylint: disable=direct-environment-variable-reference

import json
import os
from collections import defaultdict
from typing import Any, List, Type

from google.cloud import discoveryengine
from pydantic import BaseModel, Field

from ai_gateway.searches import VertexAISearch
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from lib.context import gitlab_version

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
    description: str = """Find GitLab documentation snippets relevant to the user's question.
    This tool searches GitLab's official documentation and returns relevant snippets.

    ## When to Use This Tool:

    Use this tool when the user's question involves:**
    - GitLab features, configurations, or workflows
    - GitLab architecture, infrastructure, or technical implementation details
    - GitLab.com platform capabilities or limitations
    - Any technical aspect that might be documented in GitLab's official docs
    - Questions about "how GitLab works" even if phrased as hypotheticals

    Use this tool even when:
    - The question seems to ask for advice or methodology (the docs may contain relevant technical context)
    - The question mentions specific GitLab services (GitLab.com, GitLab CI, etc.)
    - The question is exploratory ("how would X affect Y in GitLab?")

    Parameters:
    - search: A concise search query optimized for documentation retrieval (required)

    ## Guidelines for Creating Effective Search Queries:

    1. **Extract key concepts**: Focus on the core technical terms and feature names
    - Good: "WebSocket connections limits"
    - Poor: "How do I use WebSockets?"

    2. **Use GitLab-specific terminology**: Prefer official GitLab terms
    - Good: "merge request approval rules"
    - Poor: "pull request reviews"

    3. **For impact/estimation questions**: Search for related limits, performance, or architecture docs
    - User asks about "impact of feature X" → Search: "feature X limits performance"
    - User asks about "scaling concern Y" → Search: "Y scalability architecture"

    4. **Be specific but concise**: Include relevant qualifiers without unnecessary words
    - Good: "protected branches permissions"
    - Poor: "How can I protect my branches and set up permissions?"

    5. **For how-to questions**: Convert to feature names or action phrases
    - User asks: "How do I set up CI/CD?" → Search: "CI/CD setup getting started"
    """

    args_schema: Type[BaseModel] = SearchInput
    trust_level: ToolTrustLevel = ToolTrustLevel.UNTRUSTED_EXTERNAL

    async def _execute(self, search: str) -> str:
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

        snippets_grouped = defaultdict(list)
        pages = {}

        # Restructure the output data to make the LLM focus on useful data only
        for result in search_results:
            md5 = result["metadata"]["md5sum"]

            if md5 not in pages:
                pages[md5] = {
                    "source_url": result["metadata"]["source_url"],
                    "source_title": result["metadata"]["title"],
                }

            snippets_grouped[md5].append(result["content"])

        return [
            {
                "relevant_snippets": snippets,
                **pages[md5],
            }
            for md5, snippets in snippets_grouped.items()
        ]

    def format_display_message(
        self, args: SearchInput, _tool_response: Any = None
    ) -> str:
        return f"Searching GitLab documentation for: '{args.search}'"
