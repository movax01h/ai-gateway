"""Assembly of the local review payload: resolve a target, decide scope, fetch, render.

Nothing changed at all
----------------------
An empty diff raises rather than returning a payload. A deterministic step writes
`execution_result` and nothing else a router can match on, so a payload saying "nothing changed"
would still reach the reviewer, which is the one outcome worth avoiding. `only_instructions` is
exempt: its question is still answerable. A change whose every file was excluded is not empty --
see `render.render_body`.

Custom instructions come only from `.gitlab/duo/mr-review-instructions.yaml`. Ambient review also
reads group-level instructions from
`/api/v4/ai/duo_workflows/code_review/custom_instructions`, which requires a merge request IID.
Both paths render the block through `format_instructions` and forward the flow's
`include_instruction_format_hint`, so a flow that renders attribution itself turns the hint off on
both surfaces. One casualty of scoping to the executor's directory: that file lives
at the repository root, so below it the review runs without instructions. Nothing deduplicates
findings across repeated local runs on a branch, which ambient review does via existing
discussions.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import ToolException

from duo_workflow_service.tools.code_review.custom_instructions import (
    INSTRUCTIONS_FILE_PATH,
    filter_matching_instructions,
    format_instructions,
    parse_instructions,
)
from duo_workflow_service.tools.code_review.local_review import exclusions, render
from duo_workflow_service.tools.code_review.local_review.diffs import (
    fetch_diffs,
    fetch_untracked_diffs,
)
from duo_workflow_service.tools.code_review.local_review.git_output import (
    FileChange,
    WorkingTreeStatus,
    complete_listing,
    parse_numstat,
    parse_status,
)
from duo_workflow_service.tools.code_review.local_review.target import (
    ReviewTarget,
    resolve_target,
)
from duo_workflow_service.tools.code_review.local_review.transport import (
    DIFF_FLAGS,
    STATUS_FLAGS,
    GitRunner,
)


class LocalReviewContextBuilder:
    """Build review context from a local checkout.

    Not a tool: `build_review_merge_request_context` dispatches here for the checkout surface, so
    truncation, secret redaction and prompt-security scanning are applied by that tool's `_arun`
    to whatever this returns. Its truncation budget and security overrides are the same ones this
    payload needs, so nothing is lost by not being a tool in its own right.
    """

    def __init__(self, metadata: Dict[str, Any], project: Any) -> None:
        self.metadata = metadata
        self.project = project
        self._git = GitRunner(metadata)

    async def build(
        self,
        *,
        mode: str = "working_tree",
        base_ref: Optional[str] = None,
        intent: Optional[str] = None,
        only_instructions: bool = False,
        lightweight: bool = False,
        include_changed_files_list: bool = True,
        include_instruction_format_hint: bool = False,
    ) -> str:
        """Assemble the review payload from the executor's checkout.

        Args:
            mode: `working_tree` (the default) compares the base against the working tree; `branch` compares it
                against `HEAD`, so committed changes only.
            base_ref: Ref to compare against; defaults to `origin/<default_branch>`, then `<default_branch>`.
            intent: The author's statement of what the change is for, rendered as `<review_intent>`.
            only_instructions: Return only the reviewable paths and the custom instructions matching them, fetching
                no content. `lightweight` is the merge-request path's name for the same thing and is folded in.
            lightweight: Alias of `only_instructions`.
            include_changed_files_list: Render `<changed_files>`; ignored in instructions-only mode, whose whole
                output is that list.
            include_instruction_format_hint: Forwarded to `format_instructions`; see `_render_instructions`.

        Returns:
            The `<input>` payload, or in instructions-only mode the path list and matching instructions.

        Raises:
            ToolException: The base ref is invalid or absent, a git command failed, a listing was truncated, or
                the diff is empty and `only_instructions` is false.
        """
        only_instructions = only_instructions or lightweight

        # Reads refs and objects only, so no filter runs and a bad ref is rejected first.
        target = await resolve_target(self._git, self.project, mode, base_ref)

        # Everything below reads the working tree, starting with `status`.
        drivers = await self._git.filter_drivers()

        # `status` has no `--relative`, so its root-relative paths are rebased here.
        prefix = await self._git.prefix()
        status = parse_status(
            complete_listing(
                await self._git.run("status", *STATUS_FLAGS, filter_drivers=drivers),
                "status",
            ),
            prefix,
        )
        numstat = parse_numstat(
            complete_listing(
                await self._git.run(
                    "diff",
                    *DIFF_FLAGS,
                    "--numstat",
                    "-z",
                    *target.range,
                    filter_drivers=drivers,
                ),
                "diff --numstat",
            )
        )

        untracked = status.untracked if mode == "working_tree" else []

        # `execution_result` is the only thing a router can branch on.
        if not only_instructions and not numstat and not untracked:
            raise ToolException(
                f"Error: nothing to review. {target.describe()} is empty."
            )

        candidates, untracked_excluded = exclusions.select_untracked(
            self.project, untracked
        )

        generated, filtered = await exclusions.attribute_paths(
            self._git,
            sorted({*numstat, *candidates}),
            target.attribute_source,
            drivers,
        )

        # One selection path for both modes; content exclusions need the full review.
        tracked, excluded = exclusions.select_files(
            self.project, numstat, generated, filtered
        )
        untracked_allowed, attributed_excluded = exclusions.exclude_attributed(
            candidates, generated, filtered
        )
        excluded.extend(sorted([*untracked_excluded, *attributed_excluded]))

        if only_instructions:
            return await self._format_instructions_output(
                sorted({*tracked, *untracked_allowed}),
                excluded,
                include_instruction_format_hint,
            )

        return await self._format_review_output(
            target,
            status,
            numstat,
            tracked,
            untracked_allowed,
            excluded,
            intent,
            drivers,
            prefix,
            include_changed_files_list,
            include_instruction_format_hint,
        )

    async def _format_review_output(
        self,
        target: ReviewTarget,
        status: WorkingTreeStatus,
        numstat: Dict[str, FileChange],
        tracked: List[str],
        untracked: List[str],
        excluded: List[Tuple[str, str]],
        intent: Optional[str],
        drivers: Tuple[str, ...] = (),
        prefix: str = "",
        include_changed_files_list: bool = True,
        include_instruction_format_hint: bool = False,
    ) -> str:
        diffs: Dict[str, str] = {}
        renamed: Dict[str, str] = {}
        if tracked:
            diffs, renamed, unreadable = await fetch_diffs(
                self._git, target, tracked, numstat, drivers
            )
            excluded.extend(unreadable)

        if untracked:
            untracked_diffs, unreadable = await fetch_untracked_diffs(
                self._git, untracked
            )
            diffs.update(untracked_diffs)
            excluded.extend(unreadable)

        # `<changed_files>` is sorted, so the diffs must be too, untracked ones included.
        diffs = dict(sorted(diffs.items()))

        # Ordered as the merge-request payload is, so the two can be compared tag for tag.
        sections = [
            target.render(dirty=status.is_dirty, prefix=prefix),
            render.render_intent(intent),
            await self._render_instructions(
                sorted(diffs), include_instruction_format_hint
            ),
            render.render_list("changed_files", sorted(diffs))
            if include_changed_files_list
            else "",
            render.render_uncommitted(target.mode, status),
            render.render_pairs("excluded_files", excluded),
            render.render_body(diffs, renamed, excluded),
        ]

        body = "\n\n".join(section for section in sections if section)

        return (
            f"Here is the local change for you to review:\n\n<input>\n{body}\n</input>"
        )

    async def _format_instructions_output(
        self,
        changed_files: List[str],
        excluded: List[Tuple[str, str]],
        include_instruction_format_hint: bool = False,
    ) -> str:
        sections = [
            render.render_list("changed_files", changed_files)
            or render.render_body({}, {}, excluded),
            await self._render_instructions(
                changed_files, include_instruction_format_hint
            ),
        ]

        return "\n\n".join(section for section in sections if section)

    async def _render_instructions(
        self, changed_files: List[str], include_format_hint: bool = False
    ) -> str:
        """Instructions matching the files actually under review.

        The hint tells the model to prefix its own comments with the instruction name. Its value is
        forwarded from the flow, as on the merge-request path: a flow whose terminal step renders
        that from each finding's `custom_instruction_ref` passes `False`, because sending the hint
        as well would attribute every such comment twice.
        """
        if not changed_files:
            return ""

        instructions = filter_matching_instructions(
            parse_instructions(await self._read_instructions_file()), changed_files
        )

        return format_instructions(
            instructions, include_format_hint=include_format_hint
        )

    async def _read_instructions_file(self) -> Optional[str]:
        return await self._git.read_file(INSTRUCTIONS_FILE_PATH)
