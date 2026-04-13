import json
from typing import Any, List, Type

from dependency_injector.wiring import Provide, inject
from pydantic import BaseModel, Field

from ai_gateway.container import ContainerApplication
from ai_gateway.searches import Searcher
from ai_gateway.searches.typing import SearchResult
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from lib.context import gitlab_version

DEFAULT_PAGE_SIZE = 4
DEFAULT_GL_VERSION = "18.0.0"


class SearchInput(BaseModel):
    search: str = Field(description="The search term")


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
        results = await self._fetch_documentation(search)
        return json.dumps({"search_results": results})

    @inject
    async def _fetch_documentation(
        self,
        query: str,
        searcher: Searcher = Provide[ContainerApplication.searches.search_provider],
    ) -> List[dict]:
        gl_version = gitlab_version.get() or DEFAULT_GL_VERSION

        search_results: List[SearchResult] = await searcher.search_with_retry(
            query=query, gl_version=gl_version, page_size=DEFAULT_PAGE_SIZE
        )

        return searcher.dump_results(search_results)

    def format_display_message(
        self, args: SearchInput, _tool_response: Any = None
    ) -> str:
        return f"Searching GitLab documentation for: '{args.search}'"
