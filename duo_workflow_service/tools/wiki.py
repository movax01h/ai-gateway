import json
import logging
from typing import Any, Optional, Type
from urllib.parse import quote

from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

logger = logging.getLogger(__name__)

# editorconfig-checker-disable
WIKI_IDENTIFICATION_DESCRIPTION = """If you encounter a GitLab URL, extract the parameters as follows:
- https://gitlab.com/namespace/project/-/wikis/home
  > project_id="namespace/project", slug="home"
- https://gitlab.com/groups/namespace/group/-/wikis/home
  > group_id="namespace/group", slug="home"
- https://gitlab.com/namespace/project/-/wikis/dir/page_name
  > project_id="namespace/project", slug="dir/page_name"
"""

# editorconfig-checker-enable


class WikiBaseTool(DuoBaseTool):
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.DUO_AGENT_PLATFORM


class WikiResourceInput(BaseModel):
    """Input model for wiki page resources that can belong to either a project or group."""

    project_id: Optional[str] = Field(
        default=None,
        description=(
            "The namespace path of the project (e.g., 'namespace/project'). "
            "Required if group_id is not provided."
        ),
    )
    group_id: Optional[str] = Field(
        default=None,
        description=(
            "The namespace path of the group (e.g., 'namespace/group'). "
            "Required if project_id is not provided."
        ),
    )
    slug: str = Field(
        description=(
            "The slug of the wiki page (e.g., 'home', 'dir/page_name'). Required."
        ),
    )


class GetWikiPage(WikiBaseTool):
    name: str = "get_wiki_page"
    description: str = f"""Get a single wiki page from a GitLab project or group, including all its comments.

{WIKI_IDENTIFICATION_DESCRIPTION}
"""
    args_schema: Type[BaseModel] = WikiResourceInput  # type: ignore

    def _get_resource_info(
        self, project_id: Optional[str], group_id: Optional[str]
    ) -> tuple[str, str]:
        """Validate and determine the resource type and ID.

        Args:
            project_id: Optional project ID
            group_id: Optional group ID

        Returns:
            Tuple of (resource_type, resource_id)

        Raises:
            ToolException: If validation fails
        """
        if project_id and group_id:
            raise ToolException(
                "Only one of 'project_id' or 'group_id' should be provided, not both"
            )

        if not project_id and not group_id:
            raise ToolException("Either 'project_id' or 'group_id' must be provided")

        resource_type = "project" if project_id else "group"
        resource_id = project_id if project_id else group_id

        # Type assertion for mypy - validation above ensures resource_id is not None
        assert resource_id is not None

        return resource_type, resource_id

    async def _fetch_wiki_page(
        self, resource_type: str, resource_id: str, slug: str
    ) -> tuple[Any, Optional[int]]:
        """Fetch a wiki page from the GitLab API.

        Args:
            resource_type: Either "project" or "group"
            resource_id: The ID of the project or group
            slug: The wiki page slug

        Returns:
            Tuple of (wiki_page, wiki_page_meta_id)

        Raises:
            ToolException: If the API call fails or response cannot be parsed
        """
        # URL-encode the slug and resource_id for API calls
        encoded_slug = quote(slug, safe="")
        encoded_resource_id = quote(resource_id, safe="")

        # Determine API path for wiki page based on resource type
        if resource_type == "project":
            wiki_path = f"/api/v4/projects/{encoded_resource_id}/wikis/{encoded_slug}"
        else:  # group
            wiki_path = f"/api/v4/groups/{encoded_resource_id}/wikis/{encoded_slug}"

        # Get the wiki page
        wiki_response = await self.gitlab_client.aget(
            path=wiki_path,
            parse_json=False,
            use_http_response=True,  # type: ignore[call-arg]
        )

        if not wiki_response.is_success():
            logger.error(
                "Wiki page API error - Status: %s, Body: %s",
                wiki_response.status_code,
                wiki_response.body,
            )
            raise ToolException(
                f"Failed to fetch wiki page: HTTP {wiki_response.status_code}. "
                f"Verify that the {resource_type}_id '{resource_id}' and slug '{slug}' are correct."
            )

        wiki_page = wiki_response.body

        # Parse wiki_page if it's a string (when parse_json=False)
        if isinstance(wiki_page, str):
            try:
                wiki_page_dict = json.loads(wiki_page)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse wiki page JSON: %s", str(e))
                raise ToolException(
                    f"Failed to parse wiki page response as JSON: {str(e)}"
                )
        else:
            wiki_page_dict = wiki_page

        # Extract wiki_page_meta_id from the response for fetching notes
        wiki_page_meta_id = None
        if isinstance(wiki_page_dict, dict):
            wiki_page_meta_id = wiki_page_dict.get("wiki_page_meta_id")

        if not wiki_page_meta_id:
            logger.warning(
                "Could not extract wiki_page_meta_id from response, cannot fetch notes. Response type: %s",
                type(wiki_page_dict).__name__,
            )

        return wiki_page, wiki_page_meta_id

    async def _fetch_wiki_page_notes(
        self, resource_type: str, resource_id: str, wiki_page_meta_id: int
    ) -> tuple[Optional[Any], Optional[str]]:
        """Fetch notes/comments for a wiki page from the GitLab API.

        Args:
            resource_type: Either "project" or "group"
            resource_id: The ID of the project or group
            wiki_page_meta_id: The meta ID of the wiki page

        Returns:
            Tuple of (notes, error_message). If successful, returns (notes, None).
            If failed, returns (None, error_message).
        """
        # URL-encode the resource_id for notes API call
        encoded_resource_id = quote(resource_id, safe="")

        # Determine notes API path based on resource type and wiki_page_meta_id
        if resource_type == "project":
            notes_path = f"/api/v4/projects/{encoded_resource_id}/wiki_pages/{wiki_page_meta_id}/notes"
        else:  # group
            notes_path = f"/api/v4/groups/{encoded_resource_id}/wiki_pages/{wiki_page_meta_id}/notes"

        # Fetch the notes/comments for the wiki page
        try:
            notes_response = await self.gitlab_client.aget(
                path=notes_path,
                parse_json=False,
                use_http_response=True,  # type: ignore[call-arg]
            )

            if not notes_response.is_success():
                logger.warning(
                    "Wiki notes API error - Status: %s, Body: %s",
                    notes_response.status_code,
                    notes_response.body,
                )
                return None, f"Failed to fetch notes: HTTP {notes_response.status_code}"

            return notes_response.body, None

        except Exception as notes_error:
            logger.warning("Exception while fetching wiki notes: %s", str(notes_error))
            return None, f"Failed to fetch notes: {str(notes_error)}"

    async def _execute(
        self,
        slug: str,
        project_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> str:

        # Get and validate resource info
        resource_type, resource_id = self._get_resource_info(project_id, group_id)

        # Fetch the wiki page
        wiki_page, wiki_page_meta_id = await self._fetch_wiki_page(
            resource_type, resource_id, slug
        )

        # If we couldn't get the wiki_page_meta_id, return wiki page without notes
        if not wiki_page_meta_id:
            return json.dumps(
                {
                    "wiki_page": wiki_page,
                    "notes_error": "Could not extract wiki_page_meta_id from response",
                }
            )

        # Fetch the notes for the wiki page
        notes, notes_error = await self._fetch_wiki_page_notes(
            resource_type, resource_id, wiki_page_meta_id
        )

        # Return the result with notes if successful, or with error if failed
        if notes_error:
            return json.dumps({"wiki_page": wiki_page, "notes_error": notes_error})

        return json.dumps({"wiki_page": wiki_page, "notes": notes})

    def format_display_message(
        self, args: WikiResourceInput, _tool_response: Any = None
    ) -> str:
        resource_type, resource_id = self._get_resource_info(
            args.project_id, args.group_id
        )
        return f"Read wiki page '{args.slug}' in {resource_type} {resource_id}"
