import base64
import fnmatch
import html
import json
import re
from typing import Annotated, Any, Dict, Iterable, Iterator, List, Optional, Type
from urllib.parse import quote, unquote

import structlog
import yaml
from langchain_core.tools import InjectedToolArg, ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput
from duo_workflow_service.tools.tool_output_manager import TruncationConfig
from duo_workflow_service.tools.version_compatibility import (
    supports_group_level_custom_instructions,
)

logger = structlog.stdlib.get_logger(__name__)

# States for the <review_scope> block. Only DELTA narrows the review's focus, so an
# empty set of marked lines is never read as "nothing to review".
SCOPE_DELTA = "delta"
SCOPE_NO_NEW_LINES = "no_new_lines"
SCOPE_TRUNCATED = "truncated"
SCOPE_FAILED = "failed"

# The guidance travels with the diff, in the user message, because the system prompt
# that explains these states is a long way from the lines they apply to.
SCOPE_DELTA_GUIDANCE = (
    'Lines added since the previous review carry since_last_review="true". '
    "Review those first and in depth."
)
# Every other state means the markers are unusable. The state attribute already names
# the reason, and none of them change what the model should do about it.
FULL_REVIEW_GUIDANCE = "No lines are marked. Review the whole diff at full priority."

# Several flows expose this tool in an agent toolset, so baseline_sha can arrive from a
# model rather than from Rails. Anything not shaped like a commit SHA is dropped before
# it can cost a request and render a <review_scope> block into a prompt that never
# describes one. SHA-256 repositories push the upper bound to 64.
SHA_PATTERN = re.compile(r"\A[0-9a-fA-F]{7,64}\Z")

CUSTOM_INSTRUCTION_FORMAT_HINT = """

When commenting based on custom instructions, format as:
"According to custom instructions in '[instruction_name]' ([brief paraphrase of relevant instruction]): [your specific comment about the code]"

Example: "According to custom instructions in 'Security Best Practices' (validate all API input): This endpoint should validate input parameters to prevent SQL injection."

This formatting is only required for custom instruction comments. Regular review comments based on standard review criteria should NOT include this prefix."""


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
    lightweight: bool = Field(
        default=False,
        description=(
            "If True, return only changed file paths and custom instructions "
            "(no diff content). Useful for context analysis."
        ),
    )
    include_diff_links: bool = Field(
        default=False,
        description=(
            "If True, include clickable diff link URLs for each changed file. "
            "Useful for producing review summaries with file references."
        ),
    )
    # Injected, so the field stays out of the schema the model is shown while the
    # flow config can still set it. security_review and fix_pipeline expose this tool
    # in an agent toolset, and a baseline means nothing to either: it would only give
    # the model a SHA to invent and a <review_scope> block to earn for a prompt that
    # never describes one. Those flows keep the schema they had before this field
    # existed. See _resolve_review_scope for the guard on the value itself.
    baseline_sha: Annotated[Optional[str], InjectedToolArg] = Field(
        default=None,
        description=(
            "Head commit SHA at the previous review. Lines added since that commit "
            'are marked since_last_review="true". Ignored in lightweight mode.'
        ),
    )
    include_instruction_format_hint: Annotated[bool, InjectedToolArg] = Field(
        default=True,
        description=(
            "If True, the custom-instructions section tells the model to prefix "
            "comments with an 'According to custom instructions...' attribution. "
            "Set False for flows that render the attribution themselves from "
            "structured findings, so the model is not instructed to write it too."
        ),
    )
    include_changed_files_list: Annotated[bool, InjectedToolArg] = Field(
        default=False,
        description=(
            "If True, add a <changed_files> list of every changed file path "
            "immediately before <git_diffs>, as an explicit coverage checklist "
            "the reviewer works through as it reads the diffs. No effect in "
            "lightweight mode, whose output is the list."
        ),
    )


class BuildReviewMergeRequestContext(DuoBaseTool):
    """Build comprehensive merge request context for code review."""

    name: str = "build_review_merge_request_context"
    description: str = (
        "Build comprehensive merge request context for code review.\n"
        "Fetches MR details, AI-reviewable diffs, and original files content.\n"
        "Set only_diffs=True to skip fetching original file contents for faster scanning.\n"
        "Set lightweight=True to return only changed file paths and custom instructions.\n"
        "Set baseline_sha to mark the lines added since a previous review.\n"
        "Identify merge request with either:\n"
        "- project_id and merge_request_iid\n"
        "- GitLab URL (https://gitlab.com/namespace/project/-/merge_requests/42)\n"
        "Examples:\n"
        "- build_review_merge_request_context(project_id=13, merge_request_iid=9)\n"
        "- build_review_merge_request_context(project_id=13, merge_request_iid=9, only_diffs=True)\n"
        "- build_review_merge_request_context(project_id=13, merge_request_iid=9, lightweight=True)\n"
        "- build_review_merge_request_context(url='https://gitlab.com/...')"
    )
    args_schema: Type[BaseModel] = BuildReviewMergeRequestContextInput
    truncation_config: TruncationConfig = Field(
        default_factory=lambda: TruncationConfig(
            max_bytes=1 * 1024 * 1024,  # 1 MiB (~262K tokens)
            truncated_size=800 * 1024,  # 800 KiB (~200K tokens)
        )
    )

    async def _execute(self, **kwargs: Any) -> str:
        """Execute the tool to build merge request context."""
        validation_result = self._validate_merge_request_url(
            kwargs.get("url"), kwargs.get("project_id"), kwargs.get("merge_request_iid")
        )

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        only_diffs = kwargs.get("only_diffs", False)
        lightweight = kwargs.get("lightweight", False)
        include_diff_links = kwargs.get("include_diff_links", False)
        baseline_sha = kwargs.get("baseline_sha")
        include_instruction_format_hint = kwargs.get(
            "include_instruction_format_hint", True
        )
        include_changed_files_list = kwargs.get("include_changed_files_list", False)
        context = await self._build_context(
            validation_result, only_diffs, lightweight, baseline_sha
        )

        if lightweight:
            return self._format_lightweight_output(
                context, include_instruction_format_hint=include_instruction_format_hint
            )
        return self._format_output(
            context,
            include_diff_links=include_diff_links,
            include_instruction_format_hint=include_instruction_format_hint,
            include_changed_files_list=include_changed_files_list,
        )

    async def _build_context(
        self,
        validation_result,
        only_diffs: bool = False,
        lightweight: bool = False,
        baseline_sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build complete merge request context by fetching all necessary data."""
        # Fetch MR metadata
        mr_data = await self._fetch_mr_data(validation_result)

        # Fetch and process diffs
        diffs_data = await self._fetch_mr_diffs(validation_result)
        diffs_and_paths, modified_files = self._process_filtered_diffs(diffs_data)

        # Get all diff file paths for instruction matching
        diff_file_paths = list(diffs_and_paths.keys())

        # Get target branch (used for fetching original file contents below)
        target_branch = mr_data.get("target_branch")
        if not target_branch:
            raise ValueError("Target branch not found in merge request data")

        # Get custom instructions filtered by matching files
        custom_instructions = await self._get_custom_instructions(
            validation_result.project_id,
            validation_result.merge_request_iid,
            diff_file_paths,
        )

        # Lightweight mode: return only file paths and custom instructions
        if lightweight:
            return {
                "file_paths": diff_file_paths,
                "custom_instructions": custom_instructions,
            }

        renamed_files = {}
        for diff in diffs_data:
            if diff.get("renamed_file", False):
                renamed_files[diff.get("new_path")] = diff.get("old_path")

        review_scope, changed_lines = await self._resolve_review_scope(
            validation_result, mr_data, baseline_sha, diffs_and_paths
        )

        mr_context_data = {
            "mr_data": mr_data,
            "diffs_and_paths": diffs_and_paths,
            "custom_instructions": custom_instructions,
            "renamed_files": renamed_files,
            "review_scope": review_scope,
            "changed_lines": changed_lines,
        }

        # If only_diffs is True, skip fetching original files
        if only_diffs:
            return mr_context_data

        files_content = await self._fetch_original_files(
            validation_result.project_id, target_branch, modified_files
        )

        mr_context_data["files_content"] = files_content

        return mr_context_data

    async def _fetch_mr_data(self, validation_result) -> Dict[str, Any]:
        """Fetch merge request metadata."""
        path = (
            f"/api/v4/projects/{validation_result.project_id}/"
            f"merge_requests/{validation_result.merge_request_iid}"
        )
        response = await self.gitlab_client.aget(path, parse_json=False)

        body = self._process_http_response(
            "fetch merge request metadata", response, logger
        )
        return json.loads(body)

    async def _fetch_mr_diffs(self, validation_result) -> List[Dict[str, Any]]:
        """Fetch merge request diffs."""
        path = (
            f"/api/v4/projects/{validation_result.project_id}/"
            f"merge_requests/{validation_result.merge_request_iid}/diffs"
        )
        return await self._paginate_get(path)

    async def _resolve_review_scope(
        self,
        validation_result,
        mr_data: Dict[str, Any],
        baseline_sha: Optional[str],
        diffs_and_paths: Dict[str, str],
    ) -> tuple[Optional[str], set]:
        """Work out which lines arrived since the previous review.

        Returns the scope state and the ``(new_path, new_line)`` pairs added since
        ``baseline_sha``. Only ``delta`` carries pairs; every other state returns an
        empty set and means the whole diff still needs a full-priority review.

        Both diffs are taken against the current head, so ``new_line`` means the same
        thing in each. ``old_line`` does not, because the bases differ.
        """
        if not baseline_sha:
            return None, set()

        if not SHA_PATTERN.match(baseline_sha):
            logger.warning(
                "Ignoring a review baseline that is not a commit SHA",
                merge_request_iid=validation_result.merge_request_iid,
            )
            return None, set()

        diff_refs = mr_data.get("diff_refs") or {}
        head_sha = diff_refs.get("head_sha") or mr_data.get("sha")

        if not head_sha:
            logger.warning(
                "No head SHA on merge request, cannot compute the incremental diff",
                merge_request_iid=validation_result.merge_request_iid,
            )
            return SCOPE_FAILED, set()

        # Nothing was pushed since the last review, the common re-review-on-comment
        # case. There is no point asking for a comparison.
        if baseline_sha == head_sha:
            return SCOPE_NO_NEW_LINES, set()

        # For a fork, both commits live in the source project, so compare there.
        project_id = mr_data.get("source_project_id") or validation_result.project_id

        return await self._changed_lines_since(
            project_id, baseline_sha, head_sha, diffs_and_paths
        )

    async def _changed_lines_since(
        self,
        project_id: int,
        baseline_sha: str,
        head_sha: str,
        diffs_and_paths: Dict[str, str],
    ) -> tuple[str, set]:
        """Compare two commits and collect the lines they added to this merge request."""
        compare = await self._fetch_compare(project_id, baseline_sha, head_sha)

        if compare is None:
            return SCOPE_FAILED, set()

        compare_diffs = compare.get("diffs") or []

        # The compare under-reports what changed. A partial set would leave new lines
        # unmarked while the state still claimed a complete delta.
        #
        # compare_timeout is a misnomer: Entities::Compare exposes it as
        # diffs.overflow?, which GitLab sets when it stops emitting files partway
        # through, so the missing ones never reach us. _has_collapsed_diff catches the
        # other shape, a file that arrives with its patch dropped for size.
        if compare.get("compare_timeout") or self._has_collapsed_diff(compare_diffs):
            logger.warning(
                "Compare against the review baseline was incomplete, falling back to a full review",
                baseline_sha=baseline_sha,
                head_sha=head_sha,
            )
            return SCOPE_TRUNCATED, set()

        # Intersect with this merge request's own diff. The compare also carries
        # commits a merge or rebase pulled in from the target branch, and files this
        # review filters out. Neither is reviewable here, so a compare made only of
        # them must not be reported as a delta.
        changed_lines = self._added_line_keys(
            (diff.get("new_path"), diff.get("diff")) for diff in compare_diffs
        ) & self._added_line_keys(diffs_and_paths.items())

        if not changed_lines:
            return SCOPE_NO_NEW_LINES, set()

        return SCOPE_DELTA, changed_lines

    @staticmethod
    def _has_collapsed_diff(diffs: List[Dict[str, Any]]) -> bool:
        """Report whether any file came back without its patch.

        GitLab returns an empty ``diff`` when a patch exceeds the size limit. A pure
        rename and a deletion are patchless by design and have no added line to lose.
        A new file is all added lines, so a patchless one is a dropped patch.

        A rename carrying large content changes still slips through: the compare API
        gives no flag that tells it apart from a pure rename.
        """
        return any(
            diff.get("new_path")
            and not diff.get("diff")
            and not diff.get("renamed_file")
            and not diff.get("deleted_file")
            for diff in diffs
        )

    async def _fetch_compare(
        self, project_id: int, from_sha: str, to_sha: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch the diff between two commits, or None when it cannot be computed.

        Failure is ordinary here: a force push orphans the baseline and 404s, a
        private fork 403s. Neither deserves an error-level log, so this avoids
        ``_process_http_response``, which logs every non-2xx as an error.

        ``straight=true`` compares the two commits directly. The merge base default
        would, after a rebase, fall back to the original fork point and return the
        whole merge request, marking every line as new.
        """
        path = (
            f"/api/v4/projects/{project_id}/repository/compare"
            f"?from={quote(from_sha, safe='')}&to={quote(to_sha, safe='')}"
            "&straight=true"
        )

        try:
            response = await self.gitlab_client.aget(path, parse_json=False)

            if isinstance(response, GitLabHttpResponse):
                if not response.is_success():
                    logger.warning(
                        "Compare against the review baseline was rejected",
                        status_code=response.status_code,
                        project_id=project_id,
                    )
                    return None
                response = response.body

            return json.loads(response)
        except Exception:
            # Keep the traceback. Without it a genuinely broken feature reads
            # exactly like an ordinary force push.
            logger.warning("Compare against the review baseline failed", exc_info=True)
            return None

    def _added_line_keys(self, pairs: Iterable[tuple]) -> set:
        """Collect ``(path, new_line)`` for every added line in ``(path, raw_diff)`` pairs."""
        return {
            (path, line_new)
            for path, raw_diff in pairs
            if path and raw_diff
            for kind, _, line_new, _ in self._walk_diff_lines(raw_diff)
            if kind == "added"
        }

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

        body = self._process_http_response(
            f"fetch file content for {file_path}", response, logger
        )
        file_data = json.loads(body)
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
        merge_request_iid: int,
        diff_file_paths: List[str],
    ) -> List[Dict[str, Any]]:
        if supports_group_level_custom_instructions():
            try:
                return await self._fetch_all_custom_instructions(
                    project_id, merge_request_iid
                )
            except Exception:
                logger.warning(
                    "Failed to fetch custom instructions, continuing review without them",
                    project_id=project_id,
                    merge_request_iid=merge_request_iid,
                    exc_info=True,
                )
                return []

        default_branch = self.project.get("default_branch")
        if not default_branch:
            return []

        return await self._fetch_project_only_custom_instructions(
            project_id, default_branch, diff_file_paths
        )

    async def _fetch_project_only_custom_instructions(
        self,
        project_id: int,
        branch: str,
        diff_file_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Get custom instructions filtered by matching file paths."""
        instructions_content = await self._fetch_custom_instructions_file(
            project_id, branch
        )

        all_instructions = self._parse_custom_instructions(instructions_content)
        return self._filter_matching_instructions(all_instructions, diff_file_paths)

    async def _fetch_all_custom_instructions(
        self, project_id: int, merge_request_iid: int
    ) -> List[Dict[str, Any]]:
        """Fetch custom instructions from the GitLab API, including group-level instructions."""
        response = await self.gitlab_client.aget(
            "/api/v4/ai/duo_workflows/code_review/custom_instructions",
            params={
                # project_id may be an already-URL-encoded path ("group%2Fproject")
                # when the tool was invoked via URL; decode it so the client's own
                # param encoding doesn't double-encode it into a 404. Safe on
                # never-encoded values: "%" is not a legal character in GitLab
                # namespace/project paths (Gitlab::PathRegex) and numeric ids have
                # none, so unquote() is a no-op for them.
                "project_id": unquote(str(project_id)),
                "merge_request_iid": merge_request_iid,
            },
            parse_json=False,
        )

        body = self._process_http_response(
            "fetch custom instructions", response, logger
        )

        data = json.loads(body)
        return data.get("instructions", [])

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

    def _format_lightweight_output(
        self, context: dict, include_instruction_format_hint: bool = True
    ) -> str:
        """Format lightweight output with only file paths and custom instructions."""
        file_paths = "\n".join(f"- {path}" for path in context["file_paths"])
        custom_instructions_section = self._format_custom_instructions(
            context.get("custom_instructions", []),
            include_format_hint=include_instruction_format_hint,
        )

        output = f"<changed_files>\n{file_paths}\n</changed_files>"

        if custom_instructions_section:
            output += f"\n\n{custom_instructions_section}"

        return output

    def _format_output(
        self,
        context: dict,
        include_diff_links: bool = False,
        include_instruction_format_hint: bool = True,
        include_changed_files_list: bool = False,
    ) -> str:
        """Format output with escaped user content."""

        # Escape user-controlled fields to prevent HTML/XML injection
        title = html.escape(context["mr_data"].get("title") or "")
        description = html.escape(context["mr_data"].get("description") or "")

        custom_instructions_section = self._format_custom_instructions(
            context.get("custom_instructions", []),
            include_format_hint=include_instruction_format_hint,
        )

        file_diffs_section = self._format_diffs(
            context["diffs_and_paths"], context.get("changed_lines")
        )
        renamed_files_section = self._format_renamed_files(context["renamed_files"])
        diff_section = "\n\n".join(
            list(filter(None, [file_diffs_section, renamed_files_section]))
        )

        files_section = self._format_original_files(context.get("files_content", {}))

        diff_links_section = (
            self._format_diff_links(context["mr_data"], context["diffs_and_paths"])
            if include_diff_links
            else ""
        )

        review_scope_section = self._format_review_scope(context.get("review_scope"))

        # Pure renames have no diff, so diffs_and_paths misses them; dict.fromkeys keeps
        # the order and dedupes a rename that also changed content.
        changed_paths = dict.fromkeys(
            [*context["diffs_and_paths"], *context["renamed_files"]]
        )
        changed_files_section = (
            "<changed_files>\n"
            + "\n".join(f"- {path}" for path in changed_paths)
            + "\n</changed_files>"
            if include_changed_files_list
            else ""
        )

        return f"""Here are the merge request details for you to review:

<input>
<mr_title>
{title}
</mr_title>

<mr_description>
{description}
</mr_description>

{review_scope_section}

{diff_links_section}

{custom_instructions_section}

{changed_files_section}

<git_diffs>
{diff_section}
</git_diffs>

{files_section}
</input>"""

    @staticmethod
    def _format_review_scope(review_scope: Optional[str]) -> str:
        """Render the block, or nothing without a baseline.

        Five other flows call this tool and none of their prompts describe it.
        """
        if not review_scope:
            return ""

        guidance = (
            SCOPE_DELTA_GUIDANCE
            if review_scope == SCOPE_DELTA
            else FULL_REVIEW_GUIDANCE
        )

        return f'<review_scope state="{review_scope}">\n{guidance}\n</review_scope>'

    def _format_custom_instructions(
        self,
        custom_instructions: List[Dict[str, Any]],
        include_format_hint: bool = True,
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
                f"(excluding: {exclude_patterns}) - {instruction['name']}:\n"
                f"{instruction['instructions'].strip()}\n"
            )

        instructions_text = "\n".join(instruction_items)
        format_hint = CUSTOM_INSTRUCTION_FORMAT_HINT if include_format_hint else ""

        return f"""<custom_instructions>
Apply these additional review instructions to matching files:

{instructions_text}
IMPORTANT: Only apply each custom instruction to files that match its specified pattern. If a file doesn't match any custom instruction pattern, only apply the standard review criteria.{format_hint}
</custom_instructions>"""

    def _format_diffs(
        self, diffs_and_paths: Dict[str, str], changed_lines: Optional[set] = None
    ) -> str:
        """Format diffs section with structured line format."""

        formatted_diffs = []
        for file_path, diff_content in diffs_and_paths.items():
            formatted_lines = self._parse_and_format_diff(
                diff_content, file_path, changed_lines
            )
            formatted_diffs.append(
                f'<file_diff filename="{file_path}">\n{formatted_lines}\n</file_diff>'
            )

        return "\n\n".join(formatted_diffs)

    def _format_renamed_files(self, renamed_files: Dict[str, str]) -> str:
        """Format diffs section with structured line format."""
        if not renamed_files:
            return ""

        formatted_renamed_files = ["<renamed_files>"]
        for new_path, old_path in renamed_files.items():
            formatted_renamed_files.append(
                f'<file old_path="{old_path}" new_path="{new_path}"></file>'
            )
        formatted_renamed_files += ["</renamed_files>"]

        return "\n".join(formatted_renamed_files)

    @staticmethod
    def _walk_diff_lines(raw_diff: str) -> Iterator[tuple[str, int, int, str]]:
        """Walk a unified diff, yielding ``(kind, old_line, new_line, text)`` per line.

        ``kind`` is one of ``chunk_header``, ``nonewline``, ``added``, ``deleted`` or
        ``context``. The line numbers belong to the yielded line, not the running
        counters. File metadata lines are skipped.

        Metadata is only skipped before the first hunk header. Inside a hunk a ``+++``
        is an added line whose own text starts with ``++``, and a ``---`` is a deleted
        line starting ``--``. Skipping those would drop the line and leave every later
        counter in the hunk one short, which misplaces the rendered line number and
        stops the incremental-diff collector matching the same line across two diffs.
        """
        line_old = 1
        line_new = 1
        in_hunk = False

        for line in raw_diff.split("\n"):
            if not line:
                continue

            if line.startswith("@@"):
                # Parse chunk header
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    line_old = int(match.group(1))
                    line_new = int(match.group(2))
                    in_hunk = True
                    yield "chunk_header", line_old, line_new, line
                continue

            # A new file section closes the hunk that came before it, so its own
            # ``---``/``+++`` headers are metadata again.
            if line.startswith("diff --git"):
                in_hunk = False
                continue

            if not in_hunk and (line.startswith("+++") or line.startswith("---")):
                continue

            # Handle "No newline at end of file"
            if line.startswith("\\"):
                yield "nonewline", line_old, line_new, line
                continue

            # Determine line type and extract text without prefix
            if line.startswith("+"):
                yield "added", line_old, line_new, line[1:]
                line_new += 1
            elif line.startswith("-"):
                yield "deleted", line_old, line_new, line[1:]
                line_old += 1
            elif line.startswith(" "):
                yield "context", line_old, line_new, line[1:]
                line_old += 1
                line_new += 1
            else:
                # Unexpected line format, treat as context
                yield "context", line_old, line_new, line
                line_old += 1
                line_new += 1

    def _parse_and_format_diff(
        self,
        raw_diff: str,
        file_path: Optional[str] = None,
        changed_lines: Optional[set] = None,
    ) -> str:
        """Parse raw diff and format each line with type and line numbers.

        Added lines listed in ``changed_lines`` are marked with
        ``since_last_review="true"``.
        """
        if not raw_diff.strip() or "Binary files" in raw_diff:
            return ""

        lines = []

        for kind, line_old, line_new, text in self._walk_diff_lines(raw_diff):
            if kind == "chunk_header":
                lines.append(f"<chunk_header>{text}</chunk_header>")
                continue

            # An added line has no old number and a deleted line has no new one.
            # A context or nonewline line carries both.
            old = "" if kind == "added" else line_old
            new = "" if kind == "deleted" else line_new
            marked = (
                kind == "added"
                and changed_lines
                and (file_path, line_new) in changed_lines
            )
            marker = ' since_last_review="true"' if marked else ""

            lines.append(
                f'<line type="{kind}" old_line="{old}" new_line="{new}"{marker}>{text}</line>'
            )

        return "\n".join(lines)

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

    @staticmethod
    def _format_diff_links(
        mr_data: Dict[str, Any], diffs_and_paths: Dict[str, str]
    ) -> str:
        """Build a diff_links block mapping each changed file to a blob permalink.

        Links are pinned to the immutable commit that was reviewed
        (``diff_refs.head_sha``), so they keep pointing at the reviewed version
        even after new commits are pushed to the merge request:

            {project_web_url}/-/blob/{head_sha}/{file_path}

        Downstream prompts append ``#L{line}`` to anchor a specific line. Using a
        blob permalink built from data already returned by the API avoids
        replicating the monolith's internal diff-anchor hashing, which would
        couple this tool to monolith implementation details.
        """
        mr_web_url = mr_data.get("web_url", "")
        diff_refs = mr_data.get("diff_refs") or {}
        head_sha = diff_refs.get("head_sha") or mr_data.get("sha")
        if not mr_web_url or not head_sha or not diffs_and_paths:
            return ""

        project_web_url = mr_web_url.split("/-/merge_requests/")[0]

        entries = []
        for file_path in diffs_and_paths:
            encoded_path = quote(file_path, safe="/")
            url = f"{project_web_url}/-/blob/{head_sha}/{encoded_path}"
            entries.append(f'  <file path="{file_path}" url="{url}" />')

        return "<diff_links>\n" + "\n".join(entries) + "\n</diff_links>"

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

        if args.lightweight:
            base_msg += " (lightweight)"
        elif args.only_diffs:
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
