import base64
import fnmatch
import json
from typing import Any, Dict, List, Optional, Type
from urllib.parse import quote

import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput


class PostDuoCodeReviewInput(BaseModel):
    """Input schema for posting Duo Code Review."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    review_output: str = Field(
        description="The complete review output containing review comments in XML format"
    )


class PostDuoCodeReview(DuoBaseTool):
    """Tool for posting Duo Code Review to a merge request."""

    name: str = "post_duo_code_review"
    description: str = (
        "Post a Duo Code Review to a merge request.\n"
        "Example: post_duo_code_review(project_id=123, merge_request_iid=45, "
        'review_output="<review>...</review>")'
    )
    args_schema: Type[BaseModel] = PostDuoCodeReviewInput
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_MERGE_REQUEST

    async def _arun(
        self, project_id: int, merge_request_iid: int, review_output: str, **kwargs: Any
    ) -> str:
        """Execute the tool to post the code review."""
        try:
            response = await self._post_review(
                project_id, merge_request_iid, review_output
            )
            return self._format_response(response, merge_request_iid)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _post_review(
        self, project_id: int, merge_request_iid: int, review_output: str
    ) -> dict:
        """Post review to GitLab API."""
        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "review_output": review_output,
        }
        return await self.gitlab_client.apost(
            path="/api/v4/ai/duo_workflows/code_review/add_comments",
            body=json.dumps(request_body),
        )

    def _format_response(self, response: dict, merge_request_iid: int) -> str:
        """Format API response as JSON string."""
        if (
            isinstance(response, dict)
            and response.get("message") == "Comments added successfully"
        ):
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Review posted to MR !{merge_request_iid}",
                }
            )
        return json.dumps({"error": f"Failed to post review: {response}"})

    def format_display_message(
        self, args: PostDuoCodeReviewInput, _tool_response: Any = None
    ) -> str:
        """Format a user-friendly display message."""
        return (
            f"Post Duo Code Review to merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )


class BuildReviewMergeRequestContextInput(ProjectResourceInput):
    """Input schema for building merge request review context."""

    merge_request_iid: Optional[int] = Field(
        default=None,
        description="The internal ID of the project merge request. Required if URL is not provided.",
    )
    only_diffs: bool = Field(
        default=False,
        description="If True, only include diffs without fetching original file contents. Useful for initial scanning.",
    )


class BuildReviewMergeRequestContext(DuoBaseTool):
    """Build comprehensive merge request context for code review."""

    name: str = "build_review_merge_request_context"
    description: str = (
        "Build comprehensive merge request context for code review.\n"
        "Fetches MR details, AI-reviewable diffs, and original files content.\n"
        "Set only_diffs=True to skip fetching original file contents for faster scanning.\n"
        "Identify merge request with either:\n"
        "- project_id and merge_request_iid\n"
        "- GitLab URL (https://gitlab.com/namespace/project/-/merge_requests/42)\n"
        "Examples:\n"
        "- build_review_merge_request_context(project_id=13, merge_request_iid=9)\n"
        "- build_review_merge_request_context(project_id=13, merge_request_iid=9, only_diffs=True)\n"
        "- build_review_merge_request_context(url='https://gitlab.com/...')"
    )
    args_schema: Type[BaseModel] = BuildReviewMergeRequestContextInput
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_MERGE_REQUEST

    async def _arun(self, **kwargs: Any) -> str:
        """Execute the tool to build merge request context."""
        validation_result = self._validate_merge_request_url(
            kwargs.get("url"), kwargs.get("project_id"), kwargs.get("merge_request_iid")
        )

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            only_diffs = kwargs.get("only_diffs", False)
            context = await self._build_context(validation_result, only_diffs)
            return self._format_output(context)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _build_context(
        self, validation_result, only_diffs: bool = False
    ) -> Dict[str, Any]:
        """Build complete merge request context by fetching all necessary data."""
        # Fetch MR metadata
        mr_data = await self._fetch_mr_data(validation_result)

        # Fetch and process diffs
        diffs_data = await self._fetch_mr_diffs(validation_result)
        diffs_and_paths, modified_files = self._process_filtered_diffs(diffs_data)

        # If only_diffs is True, skip fetching original files and custom instructions
        if only_diffs:
            return {
                "mr_data": mr_data,
                "diffs_and_paths": diffs_and_paths,
            }

        # Get all diff file paths for instruction matching
        diff_file_paths = list(diffs_and_paths.keys())

        # Fetch original file content
        target_branch = mr_data.get("target_branch")
        if not target_branch:
            raise ValueError("Target branch not found in merge request data")

        files_content = await self._fetch_original_files(
            validation_result.project_id, target_branch, modified_files
        )

        # Get custom instructions filtered by matching files
        custom_instructions = await self._get_custom_instructions(
            validation_result.project_id, target_branch, files_content, diff_file_paths
        )

        return {
            "mr_data": mr_data,
            "diffs_and_paths": diffs_and_paths,
            "files_content": files_content,
            "custom_instructions": custom_instructions,
        }

    async def _fetch_mr_data(self, validation_result) -> Dict[str, Any]:
        """Fetch merge request metadata."""
        path = (
            f"/api/v4/projects/{validation_result.project_id}/"
            f"merge_requests/{validation_result.merge_request_iid}"
        )
        response = await self.gitlab_client.aget(path, parse_json=False)
        return json.loads(response)

    async def _fetch_mr_diffs(self, validation_result) -> List[Dict[str, Any]]:
        """Fetch merge request diffs."""
        path = (
            f"/api/v4/projects/{validation_result.project_id}/"
            f"merge_requests/{validation_result.merge_request_iid}/diffs"
        )
        response = await self.gitlab_client.aget(path, parse_json=False)
        return json.loads(response)

    async def _fetch_original_files(
        self, project_id: int, branch: str, file_paths: List[str]
    ) -> Dict[str, str]:
        """Fetch original file content for modified files."""
        if not file_paths:
            return {}

        diff_policy = DiffExclusionPolicy(self.project)
        files_content = {}

        for file_path in file_paths:
            if not diff_policy.is_allowed(file_path):
                continue

            try:
                content = await self._fetch_file_content(project_id, branch, file_path)

                # Check line count and skip if too large
                line_count = content.count("\n") + 1
                if line_count > 10000:
                    continue

                files_content[file_path] = content
            except Exception:
                # Skip files that can't be fetched
                continue

        return files_content

    async def _fetch_file_content(
        self, project_id: int, branch: str, file_path: str
    ) -> str:
        """Fetch a single file's content from the repository."""
        encoded_path = quote(file_path, safe="")
        path = f"/api/v4/projects/{project_id}/repository/files/{encoded_path}"

        response = await self.gitlab_client.aget(
            path, params={"ref": branch}, parse_json=False
        )

        file_data = json.loads(response)
        return base64.b64decode(file_data["content"]).decode("utf-8")

    def _process_filtered_diffs(
        self, diffs_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, str], List[str]]:
        """Apply filters and extract diff paths with modified files."""
        diff_policy = DiffExclusionPolicy(self.project)
        filtered_diffs, _ = diff_policy.filter_allowed_diffs(diffs_data)

        ai_reviewable = self._get_reviewable_diffs(filtered_diffs)

        return self._extract_diffs_and_modified_files(ai_reviewable)

    def _get_reviewable_diffs(
        self, diffs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter diffs to only AI-reviewable ones."""
        return [
            diff
            for diff in diffs
            if not diff.get("generated_file", False) and diff.get("diff", "").strip()
        ]

    def _extract_diffs_and_modified_files(
        self, diffs: List[Dict[str, Any]]
    ) -> tuple[Dict[str, str], List[str]]:
        """Extract diff content and identify modified files."""
        diffs_and_paths: Dict[str, str] = {}
        modified_files: List[str] = []

        for diff in diffs:
            path = diff.get("new_path") or diff.get("old_path")
            if not path:
                continue

            diffs_and_paths[path] = diff.get("diff", "")

            # Track modified files (not new files)
            old_path = diff.get("old_path")
            if old_path and not diff.get("new_file", False):
                modified_files.append(old_path)

        return diffs_and_paths, modified_files

    async def _get_custom_instructions(
        self,
        project_id: int,
        branch: str,
        files_content: Dict[str, str],
        diff_file_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Get custom instructions filtered by matching file paths."""
        instructions_path = ".gitlab/duo/mr-review-instructions.yaml"
        instructions_content: Optional[str]
        # Check if instructions file is in the diff
        if instructions_path in files_content:
            instructions_content = files_content[instructions_path]
        else:
            instructions_content = await self._fetch_custom_instructions_file(
                project_id, branch
            )

        all_instructions = self._parse_custom_instructions(instructions_content)
        return self._filter_matching_instructions(all_instructions, diff_file_paths)

    async def _fetch_custom_instructions_file(
        self, project_id: int, branch: str
    ) -> Optional[str]:
        """Fetch custom instructions file from repository."""
        try:
            return await self._fetch_file_content(
                project_id, branch, ".gitlab/duo/mr-review-instructions.yaml"
            )
        except Exception:
            return None

    def _parse_custom_instructions(
        self, content: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Parse YAML custom instructions content."""
        if not content:
            return []

        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict) or "instructions" not in data:
                return []

            return [
                self._parse_instruction_item(item)
                for item in data["instructions"]
                if isinstance(item, dict) and self._is_valid_instruction(item)
            ]
        except Exception:
            return []

    def _is_valid_instruction(self, item: Dict[str, Any]) -> bool:
        """Check if instruction item has all required fields."""
        return bool(
            item.get("name") and item.get("instructions") and item.get("fileFilters")
        )

    def _parse_instruction_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single instruction item into standardized format."""
        file_filters = item.get("fileFilters", [])

        return {
            "name": item.get("name"),
            "instructions": item.get("instructions"),
            "include_patterns": [f for f in file_filters if not f.startswith("!")],
            "exclude_patterns": [f[1:] for f in file_filters if f.startswith("!")],
        }

    def _filter_matching_instructions(
        self, all_instructions: List[Dict], diff_file_paths: List[str]
    ) -> List[Dict]:
        """Filter instructions to only include those matching at least one diff file."""
        if not all_instructions:
            return []

        return [
            instruction
            for instruction in all_instructions
            if any(self._matches_pattern(path, instruction) for path in diff_file_paths)
        ]

    def _matches_pattern(self, path: str, instruction: Dict) -> bool:
        """Check if a file path matches the instruction's include/exclude patterns."""
        includes = instruction.get("include_patterns", [])
        excludes = instruction.get("exclude_patterns", [])

        # With include patterns: match only files matching includes (minus exclusions)
        # Without include patterns: match all files (minus exclusions)
        matches_include = not includes or any(
            fnmatch.fnmatch(path, pattern) for pattern in includes
        )
        matches_exclude = any(fnmatch.fnmatch(path, pattern) for pattern in excludes)

        return matches_include and not matches_exclude

    def _format_output(self, context: dict) -> str:
        """Format output in the simple template structure."""
        custom_instructions_section = self._format_custom_instructions(
            context.get("custom_instructions", [])
        )
        diff_section = self._format_diffs(context["diffs_and_paths"])
        files_section = self._format_original_files(context.get("files_content", {}))

        return f"""Here are the merge request details for you to review:

<input>
<mr_title>
{context['mr_data'].get('title', '')}
</mr_title>

<mr_description>
{context['mr_data'].get('description', '')}
</mr_description>

{custom_instructions_section}

<git_diffs>
{diff_section}</git_diffs>

{files_section}
</input>"""

    def _format_custom_instructions(
        self, custom_instructions: List[Dict[str, Any]]
    ) -> str:
        """Format custom instructions section."""
        if not custom_instructions:
            return ""

        instruction_items = []
        for instruction in custom_instructions:
            include_patterns = ", ".join(instruction["include_patterns"]) or "all files"
            exclude_patterns = ", ".join(instruction["exclude_patterns"]) or "none"

            instruction_items.append(
                f'For files matching "{include_patterns}" '
                f'(excluding: {exclude_patterns}) - {instruction["name"]}:\n'
                f'{instruction["instructions"].strip()}\n'
            )

        instructions_text = "\n".join(instruction_items)

        return f"""<custom_instructions>
Apply these additional review instructions to matching files:

{instructions_text}
IMPORTANT: Only apply each custom instruction to files that match its specified pattern. If a file doesn't match any custom instruction pattern, only apply the standard review criteria.

When commenting based on custom instructions, format as:
"According to custom instructions in '[instruction_name]': [your comment here]"

Example: "According to custom instructions in 'Security Best Practices': This API endpoint should validate input parameters to prevent SQL injection."

This formatting is only required for custom instruction comments. Regular review comments based on standard review criteria should NOT include this prefix.
</custom_instructions>"""

    def _format_diffs(self, diffs_and_paths: Dict[str, str]) -> str:
        """Format diffs section."""
        return "\n".join(
            f"<diff filename='{file_path}'>\n{diff_content}\n</diff>\n"
            for file_path, diff_content in diffs_and_paths.items()
        )

    def _format_original_files(self, files_content: Dict[str, str]) -> str:
        """Format original files section."""
        if not files_content:
            return ""

        lines = [
            "<original_files>",
            "Use this context to better understand the changes and identify genuine "
            "issues in the code. Original file content (before changes):",
        ]

        for file_path, content in files_content.items():
            lines.append(
                f"<full_file filename='{file_path}'>\n{content}\n</full_file>\n"
            )

        lines.append("</original_files>")
        return "\n".join(lines)

    def format_display_message(
        self, args: BuildReviewMergeRequestContextInput, tool_response: Any = None
    ) -> str:
        """Format a user-friendly display message."""
        if args.url:
            base_msg = f"Build review context for merge request {args.url}"
        else:
            base_msg = (
                f"Build review context for merge request !{args.merge_request_iid} "
                f"in project {args.project_id}"
            )

        if args.only_diffs:
            base_msg += " (diffs only)"

        if tool_response:
            base_msg += self._format_exclusion_message(tool_response)

        return base_msg

    def _format_exclusion_message(self, tool_response: Any) -> str:
        """Format exclusion message from tool response."""
        try:
            excluded_files = json.loads(tool_response.content).get("excluded_files", [])
            if excluded_files:
                return DiffExclusionPolicy.format_user_exclusion_message(excluded_files)
        except (json.JSONDecodeError, AttributeError):
            pass

        return ""
