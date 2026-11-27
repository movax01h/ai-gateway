"""What is being compared, and how the base ref is resolved.

`working_tree` (the default) diffs the merge base of the default branch against the working tree,
so it covers everything the branch has changed, committed or not. `branch` diffs that merge base
against `HEAD` instead: committed changes only, the same product as a merge request diff.

`working_tree` is the default because asking for a review before committing is the ordinary
interactive case, and because it is a superset -- on a clean tree the two modes produce identical
output, so the default costs nothing there. Choose `branch` when the target must be reproducible
and immutable: evaluations need stable ground truth, and a mutable target makes a stale finding
anchor the common case rather than an edge case.

The base ref resolves in order: an explicit `base_ref`, `origin/<default_branch>`,
`<default_branch>`, matched against the refs the checkout actually has. This never fetches, which
would be network access and would write to `.git`; if nothing resolves it fails naming what it
tried. Only exit 1, a missing ref, moves on to the next candidate: a corrupt repository or a
partial-clone refusal is reported as itself, not as a ref to fetch.
"""

import re
from typing import Any, List, Optional, Tuple

from langchain_core.tools import ToolException
from pydantic import BaseModel

from ai.features.review.code_review.components.local_review.render import escape_text
from ai.features.review.code_review.components.local_review.transport import (
    GitCommandError,
    GitRunner,
)

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class ReviewTarget(BaseModel):
    mode: str
    base_ref: str
    base_sha: str
    head_sha: str

    @property
    def attribute_source(self) -> Optional[str]:
        # Attributes come from the revision under review; check-attr reads the tree.
        return self.head_sha if self.mode == "branch" else None

    @property
    def range(self) -> Tuple[str, ...]:
        # Two-dot: with HEAD the committed change, without it the working tree.
        return (self.base_sha, "HEAD") if self.mode == "branch" else (self.base_sha,)

    def describe(self) -> str:
        """Name the comparison in the user's own terms, for an error they have to act on."""
        side = (
            "the committed diff of HEAD"
            if self.mode == "branch"
            else "the working tree"
        )

        return f"{side} against {self.base_ref}"

    def render(self, dirty: bool, prefix: str = "") -> str:
        # Below the root, this separates "nothing else changed" from "out of scope".
        scope = (
            f"<scope>Only changes under {escape_text(prefix)} are in scope; the rest of the "
            "repository is "
            "outside the working directory and cannot be read.</scope>\n"
            if prefix
            else ""
        )

        # `dirty` rather than `mode`: rendering needs to know if an anchor can move.
        return (
            "<review_target>\n"
            f"<mode>{self.mode}</mode>\n"
            f"{scope}"
            f"<uncommitted_changes_present>{str(dirty).lower()}</uncommitted_changes_present>\n"
            f"<base_ref>{self.base_ref}</base_ref>\n"
            f"<base_sha>{self.base_sha}</base_sha>\n"
            f"<head_sha>{self.head_sha}</head_sha>\n"
            "</review_target>"
        )


async def resolve_target(
    git: GitRunner, project: Any, mode: str, base_ref: Optional[str]
) -> ReviewTarget:
    """Pick the base to compare against and pin both ends of the comparison to SHAs.

    Candidates are tried in order with one `rev-parse --verify` each, never a ref listing: a repository with a few
    thousand refs overruns the executor's output cap, and the truncated middle is where `origin/<default>` lives.
    """
    if base_ref is not None and not _is_valid_ref(base_ref):
        raise ToolException(f"Error: {base_ref!r} is not a valid git ref name")

    candidates = _base_ref_candidates(base_ref, project)
    if not candidates:
        raise ToolException(
            "Error: no base ref to compare against. The project has no default branch and no base_ref was given."
        )

    # Listing every ref overruns the executor's output cap, losing `origin/<default>`.
    resolved = None
    for candidate in candidates:
        try:
            await git.run("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        except GitCommandError as error:
            # `--quiet` makes a missing ref exit 1; anything else is not an answer about this ref.
            if error.code != "1":
                raise
            continue

        resolved = candidate
        break

    if resolved is None:
        raise ToolException(
            f"Error: none of {', '.join(candidates)} exist in this checkout. "
            "Fetch the default branch and try again."
        )

    head_sha = (await git.run("rev-parse", "HEAD")).strip()
    base_sha = (await git.run("merge-base", resolved, "HEAD")).strip()

    return ReviewTarget(
        mode=mode, base_ref=resolved, base_sha=base_sha, head_sha=head_sha
    )


def _is_valid_ref(base_ref: str) -> bool:
    return bool(_REF_RE.match(base_ref)) and ".." not in base_ref


def _base_ref_candidates(base_ref: Optional[str], project: Any) -> List[str]:
    if base_ref:
        return [base_ref]

    default_branch = (project or {}).get("default_branch")
    if not default_branch:
        return []

    return [f"origin/{default_branch}", default_branch]
