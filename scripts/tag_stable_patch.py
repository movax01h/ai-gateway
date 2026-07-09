#!/usr/bin/env python
"""Cut the next patch tag for a stable branch.

A stable branch ``stable-<MAJOR>-<MINOR>-ee`` holds tags of the form
``self-hosted-v<MAJOR>.<MINOR>.<PATCH>-ee``. On the branch-creation pipeline the
line has no tag yet, so the baseline ``.0`` is cut; on each subsequent commit
the highest existing patch is bumped. Tags are pinned to the current commit and
trigger a separate tag pipeline that builds and releases the self-hosted images.

Designed for GitLab CI: reads ``CI_COMMIT_BRANCH`` / ``CI_COMMIT_SHA`` and
authenticates to the tags API with ``AIGW_TAGGING_ACCESS_TOKEN``.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

PROJECT_ID = "39903947"
API_BASE = f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/repository"

BRANCH_RE = re.compile(r"^stable-(\d+)-(\d+)-ee$")


def version_line(branch):
    """Return the ``<MAJOR>.<MINOR>`` line for a stable branch, or None."""
    match = BRANCH_RE.match(branch or "")
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def patch_numbers(tag_names, prefix):
    """Yield the patch integers of tags shaped ``<prefix><patch>-ee``."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)-ee$")
    for name in tag_names:
        match = pattern.match(name)
        if match:
            yield int(match.group(1))


def next_tag(tag_names, prefix):
    """Return the next patch tag name for the given line prefix.

    When the line has no tag yet, returns the baseline ``<prefix>0-ee`` (cut on the branch-creation pipeline); otherwise
    bumps the highest existing patch.
    """
    highest = max(patch_numbers(tag_names, prefix), default=-1)
    return f"{prefix}{highest + 1}-ee"


def _request(method, path, token, params=None):
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method=method)
    req.add_header("PRIVATE-TOKEN", token)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or "null")
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace").strip()
        print(
            f"Error: {method} {path} failed with HTTP {err.code}: {body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except urllib.error.URLError as err:
        print(f"Error: {method} {path} failed: {err.reason}", file=sys.stderr)
        sys.exit(1)


def head_already_tagged(tags, sha, prefix):
    """True if a tag on this version line already points at ``sha``."""
    pattern = re.compile(rf"^{re.escape(prefix)}\d+-ee$")
    return any(
        pattern.match(tag.get("name", "")) and tag.get("commit", {}).get("id") == sha
        for tag in tags
    )


def line_tags(prefix, token):
    """Fetch tag objects whose name starts with this version line prefix."""
    tags = _request(
        "GET",
        "/tags",
        token,
        {"search": f"^{prefix}", "per_page": 100},
    )
    return tags or []


def create_tag(tag_name, ref, token):
    _request(
        "POST",
        "/tags",
        token,
        {"tag_name": tag_name, "ref": ref},
    )


def main():
    # pylint: disable=direct-environment-variable-reference
    branch = os.environ.get("CI_COMMIT_BRANCH", "")
    sha = os.environ.get("CI_COMMIT_SHA", "")
    token = os.environ.get("AIGW_TAGGING_ACCESS_TOKEN", "")
    # pylint: enable=direct-environment-variable-reference

    line = version_line(branch)
    if not line:
        print(f"Not a stable branch ({branch or 'unset'}); skipping")
        return 0
    if not token:
        print("Error: AIGW_TAGGING_ACCESS_TOKEN is not set", file=sys.stderr)
        return 1

    prefix = f"self-hosted-v{line}."
    print(f"Stable branch: {branch} (version line {line})")

    tags = line_tags(prefix, token)
    if head_already_tagged(tags, sha, prefix):
        print(f"HEAD ({sha}) is already tagged on this line; nothing to do")
        return 0

    tag = next_tag([tag["name"] for tag in tags], prefix)
    print(f"Creating tag {tag} at {sha}")
    create_tag(tag, sha, token)
    print(f"Created {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
