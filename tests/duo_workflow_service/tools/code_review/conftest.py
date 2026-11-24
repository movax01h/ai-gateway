from typing import Dict, List, Optional, Set
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.code_review.local_review import (
    LocalReviewContextBuilder,
)
from duo_workflow_service.tools.code_review.local_review.transport import (
    DIFF_FLAGS,
    GIT_GLOBAL_FLAGS,
    GitCommandError,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        default_branch="main",
        languages=[],
        exclusion_rules=["**/*.log", "/secrets/**", "**/node_modules/**"],
    )


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project_mock):
    return {
        "workflow_id": "test-workflow-123",
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project_mock,
    }


# --- shared fake git executor ---------------------------------------------------
# Used by the local-review tests and by the payload parity test, so the parity test keeps
# exercising the same replay behaviour as the local one evolves.

HEAD_SHA = "1111111111111111111111111111111111111111"
BASE_SHA = "2222222222222222222222222222222222222222"

REFS = "main\norigin/main\norigin/HEAD\nfeature\n"

DIFF_ARGV = ["diff", *DIFF_FLAGS]
DIFF = " ".join(DIFF_ARGV)

APP_DIFF = """diff --git a/app/a.rb b/app/a.rb
index 111..222 100644
--- a/app/a.rb
+++ b/app/a.rb
@@ -1,1 +1,1 @@
-a = 1
+a = 2
"""


def numstat(*entries: str) -> str:
    return "\0".join(entries) + "\0"


class FakeGit:
    """Replays canned git output and records every action the tool emitted."""

    def __init__(
        self,
        responses: Dict[str, str],
        files: Optional[Dict[str, str]] = None,
        known_refs: Optional[Set[str]] = None,
    ):
        self.responses = responses
        self.files = files or {}
        self.known_refs = (
            known_refs if known_refs is not None else set(_parse_lines(REFS))
        )
        self.actions: List = []

    def __call__(self, _metadata, action):
        self.actions.append(action)

        if action.WhichOneof("action") == "runReadFile":
            path = action.runReadFile.filepath
            if path not in self.files:
                raise ToolException(f"Action error: no such file {path}")
            return self.files[path]

        argv = _strip_global_flags(list(action.runCommand.arguments))

        # Ref existence is a question, not a canned string: the tool asks about one candidate
        # at a time and reads the exit code, so the fake has to answer per ref.
        if argv[:2] == ["rev-parse", "--verify"]:
            ref = argv[-1].removesuffix("^{commit}")
            if ref not in self.known_refs:
                raise GitCommandError(
                    "rev-parse", "1", f"fatal: Needed a single revision: {ref}"
                )
            return ""

        # Longest match wins: the working_tree range is a prefix of the branch range, so
        # relying on dict order would silently serve the wrong response.
        matches = [
            (len(key.split()), value)
            for key, value in self.responses.items()
            if argv[: len(key.split())] == key.split()
        ]
        if matches:
            return max(matches, key=lambda match: match[0])[1]

        raise AssertionError(f"unexpected git invocation: {argv}")

    @property
    def git_argvs(self) -> List[List[str]]:
        return [_strip_global_flags(argv) for argv in self.raw_argvs]

    @property
    def raw_argvs(self) -> List[List[str]]:
        """Arguments exactly as emitted, including the filter overrides."""
        return [
            list(a.runCommand.arguments)
            for a in self.actions
            if a.WhichOneof("action") == "runCommand"
        ]


def _parse_lines(raw: str) -> List[str]:
    return [line.strip() for line in raw.split("\n") if line.strip()]


def _strip_global_flags(argv: List[str]) -> List[str]:
    assert argv[: len(GIT_GLOBAL_FLAGS)] == list(GIT_GLOBAL_FLAGS), argv
    rest = argv[len(GIT_GLOBAL_FLAGS) :]

    # The filter overrides are `-c` pairs sitting between the global flags and the
    # subcommand; strip them so tests can match on the subcommand as before.
    while len(rest) > 1 and rest[0] == "-c" and rest[1].startswith("filter."):
        rest = rest[2:]

    return rest


def both_ranges(numstat_out: str, diff_out: Optional[str] = None) -> Dict[str, str]:
    """Serve the same content for the branch and working_tree ranges.

    Most tests are mode-agnostic; naming a single range would couple them to whichever mode is the default.
    """
    served = {
        f"{DIFF} --numstat -z {BASE_SHA} HEAD": numstat_out,
        f"{DIFF} --numstat -z {BASE_SHA}": numstat_out,
    }
    if diff_out is not None:
        served[f"{DIFF} {BASE_SHA} HEAD"] = diff_out
        served[f"{DIFF} {BASE_SHA}"] = diff_out

    return served


def responses(**overrides: str) -> Dict[str, str]:
    base = {
        # No filter.* keys: the overrides come out empty and every command is unaffected.
        "config": "",
        "rev-parse HEAD": f"{HEAD_SHA}\n",
        # Empty prefix: the executor's directory is the repository root.
        "rev-parse --show-prefix": "",
        "merge-base": f"{BASE_SHA}\n",
        "status": "",
        "check-attr": "",
        **both_ranges(numstat("1\t1\tapp/a.rb"), APP_DIFF),
    }
    base.update(_mirror_ranges(overrides))
    return base


def _mirror_ranges(overrides: Dict[str, str]) -> Dict[str, str]:
    """Let a test name one mode's range and have it serve both.

    A test that deliberately differs per mode names both explicitly, and those win.
    """
    mirrored = dict(overrides)
    for key, value in overrides.items():
        if not key.startswith(DIFF):
            continue

        twin = key[: -len(" HEAD")] if key.endswith(" HEAD") else f"{key} HEAD"
        mirrored.setdefault(twin, value)

    return mirrored


async def run_tool(fake: FakeGit, metadata, **kwargs) -> str:
    builder = LocalReviewContextBuilder(metadata, metadata.get("project"))
    with patch(
        "duo_workflow_service.tools.code_review.local_review.transport._execute_action",
        side_effect=fake,
    ):
        return await builder.build(**kwargs)


def _file_diff(name: str, lines: int, text: str) -> str:
    body = "\n".join(f"+{text}" for _ in range(lines))

    return (
        f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n"
        f"@@ -0,0 +1,{lines} @@\n{body}\n"
    )
