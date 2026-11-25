"""Deterministic acquisition of a local review target: the diff, the changed-file list, and custom instructions.

`build_review_merge_request_context` dispatches here when its `surface` is `checkout`, in place of
the merge-request API calls it makes for `merge_request`. The graph, the flow config and the
reviewer prompt are the same on both surfaces; only where the context comes from differs.

Read in this order:

- `transport` -- the executor boundary: what makes this package read-only, and what stops
    configured git filters from executing.
- `target` -- what is being compared: the two modes, and how the base ref resolves.
- `git_output` -- parsing `status`, `--numstat` and `check-attr`, and the records they produce.
- `exclusions` -- which paths are reviewable, and the reason recorded for every one that is not.
- `diffs` -- fetching each file's diff and formatting it.
- `render` -- the payload's tags.
- `builder` -- the orchestration that calls all of the above in order.
"""

from duo_workflow_service.tools.code_review.local_review.builder import (
    LocalReviewContextBuilder,
)

__all__ = ["LocalReviewContextBuilder"]
