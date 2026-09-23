"""Tests for scripts/prompt_diff_comment.py."""

from pathlib import Path
from unittest.mock import MagicMock

import gitlab.exceptions
import pytest

from scripts.prompt_diff_comment import (
    PROMPTS_ROOT,
    SENTINEL,
    PromptDiff,
    build_comment_body,
    build_prompt_diff,
    code_fence,
    compute_diff,
    detect_new_prompt_files,
    highest_version_below,
    is_versioned_prompt_file,
    list_mr_diffs,
    list_target_branch_stems,
    parse_version,
    prompt_key,
    run,
    sync_comment,
)

SYSTEM_DIR = f"{PROMPTS_ROOT}/chat/explain_code/system"


@pytest.mark.parametrize(
    "path,expected",
    [
        (f"{SYSTEM_DIR}/1.0.0.jinja", True),
        (f"{PROMPTS_ROOT}/chat/explain_code/base/1.2.3.yml", True),
        (f"{PROMPTS_ROOT}/code_suggestions/f/user/2.0.0-dev.jinja", True),
        (f"{PROMPTS_ROOT}/common/developer/system/1.1.0-rc.yml", True),
        # Wrong root
        ("scripts/prompt_diff_comment.py", False),
        ("ai_gateway/other/1.0.0.jinja", False),
        # Not a version
        (f"{SYSTEM_DIR}/README.md", False),
        (f"{SYSTEM_DIR}/config.jinja", False),
        (f"{SYSTEM_DIR}/2.0.0-orbit.yml", False),
        # Wrong extension
        (f"{SYSTEM_DIR}/1.0.0.txt", False),
    ],
)
def test_is_versioned_prompt_file(path, expected):
    assert is_versioned_prompt_file(path) is expected


@pytest.mark.parametrize(
    "lower,higher",
    [
        ("1.0.0", "1.0.1"),
        ("1.0.1", "1.1.0"),
        ("1.9.0", "1.10.0"),
        ("1.1.0-dev", "1.1.0"),
        ("1.1.0-alpha", "1.1.0-rc"),
        ("1.0.1", "1.1.0-dev"),
    ],
)
def test_parse_version_ordering(lower, higher):
    assert parse_version(lower) < parse_version(higher)


def test_parse_version_returns_none_for_invalid():
    assert parse_version("2.0.0-orbit") is None


@pytest.mark.parametrize(
    "stems,new_stem,expected",
    [
        (["1.0.0", "1.0.1", "1.1.0"], "1.1.0", "1.0.1"),
        (["1.0.0", "1.0.1", "1.0.2", "1.1.0"], "2.0.0", "1.1.0"),
        (["1.0.0", "1.1.0"], "1.1.0", "1.0.0"),
        (["1.0.0", "1.1.0-dev"], "1.1.0", "1.1.0-dev"),
        (["1.0.0", "1.0.1"], "1.0.1-dev", "1.0.0"),
        (["1.0.0", "2.0.0-orbit"], "1.1.0", "1.0.0"),
        (["1.0.0"], "1.0.0", None),
        ([], "1.0.0", None),
        (["1.0.0"], "not-a-version", None),
    ],
)
def test_highest_version_below(stems, new_stem, expected):
    assert highest_version_below(stems, new_stem) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        (f"{SYSTEM_DIR}/1.1.0.jinja", "chat/explain_code/system"),
        (f"{PROMPTS_ROOT}/common/developer/base/1.0.0.yml", "common/developer/base"),
    ],
)
def test_prompt_key(path, expected):
    assert prompt_key(path) == expected


def test_compute_diff_produces_unified_diff_with_labels():
    result = compute_diff(
        "line one\nline two\n",
        "old/1.0.0.jinja",
        "line one\nline TWO\n",
        "new/1.1.0.jinja",
    )
    assert "--- old/1.0.0.jinja" in result
    assert "+++ new/1.1.0.jinja" in result
    assert "-line two\n" in result
    assert "+line TWO\n" in result


def test_compute_diff_identical_content():
    text = "same content\n"
    assert compute_diff(text, "a", text, "b") == "(no textual differences)"


def test_build_comment_body():
    diffs = [
        PromptDiff("chat/explain_code/system", "1.0.0", "1.1.0", "-old\n+new\n"),
        PromptDiff("chat/explain_code/user", "1.0.0", "1.0.1", "-foo\n+bar\n"),
    ]
    body = build_comment_body(diffs)

    assert body.startswith(SENTINEL)
    assert "### Prompt version diff" in body
    assert body.count("<details>") == body.count("</details>") == 2
    assert "<summary>`chat/explain_code/system: 1.0.0 → 1.1.0`</summary>" in body
    assert "<summary>`chat/explain_code/user: 1.0.0 → 1.0.1`</summary>" in body
    assert "```diff\n-old\n+new\n```" in body
    assert "```diff\n-foo\n+bar\n```" in body


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-old\n+new\n", "```"),
        ("+`inline`\n", "```"),
        ("+```markdown\n+text\n+```\n", "````"),
        ("+`````\n", "``````"),
    ],
)
def test_code_fence_outgrows_longest_backtick_run(text, expected):
    assert code_fence(text) == expected


def test_build_comment_body_survives_fences_inside_prompt():
    diff = "+Wrap it in:\n+```markdown\n+text\n+```\n"
    body = build_comment_body([PromptDiff("x/system", "1.0.0", "1.1.0", diff)])

    assert body.count("````") == 2
    assert f"````diff\n{diff.rstrip()}\n````" in body


def test_list_mr_diffs_uses_paginated_diffs_endpoint():
    mr = MagicMock()
    mr.manager.path = "/projects/1/merge_requests"
    mr.encoded_id = 7
    mr.manager.gitlab.http_list.return_value = iter([{"new_path": "a"}])

    assert list_mr_diffs(mr) == [{"new_path": "a"}]
    mr.manager.gitlab.http_list.assert_called_once_with(
        "/projects/1/merge_requests/7/diffs", get_all=True
    )


def test_detect_new_prompt_files_only_returns_added_versioned_files():
    changes = [
        {"new_path": f"{SYSTEM_DIR}/1.1.0.jinja", "new_file": True},
        {"new_path": f"{SYSTEM_DIR}/1.0.0.jinja", "new_file": False},
        {
            "new_path": f"{PROMPTS_ROOT}/chat/explain_code/base/1.1.0.yml",
            "new_file": True,
        },
        {"new_path": "ai_gateway/prompts/registry.py", "new_file": True},
        {"new_path": f"{SYSTEM_DIR}/README.md", "new_file": True},
    ]

    assert detect_new_prompt_files(changes) == [
        f"{PROMPTS_ROOT}/chat/explain_code/base/1.1.0.yml",
        f"{SYSTEM_DIR}/1.1.0.jinja",
    ]


@pytest.fixture(name="project")
def project_fixture():
    project = MagicMock()
    project.repository_tree.return_value = [
        {"type": "blob", "name": "1.0.0.jinja"},
        {"type": "blob", "name": "1.0.1.jinja"},
        {"type": "blob", "name": "1.0.1.yml"},
        {"type": "tree", "name": "1.9.9.jinja"},
    ]
    project.files.raw.return_value = b"You are a helpful assistant.\n"
    return project


@pytest.fixture(name="repo_root")
def repo_root_fixture(tmp_path):
    prompt_dir = tmp_path / SYSTEM_DIR
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "1.1.0.jinja").write_text("You are a very helpful assistant.\n")
    return tmp_path


def test_list_target_branch_stems_filters_by_extension_and_type(project):
    stems = list_target_branch_stems(project, SYSTEM_DIR, ".jinja", "main")

    assert stems == ["1.0.0", "1.0.1"]
    project.repository_tree.assert_called_once_with(
        path=SYSTEM_DIR, ref="main", get_all=True
    )


def test_list_target_branch_stems_missing_directory(project):
    project.repository_tree.side_effect = gitlab.exceptions.GitlabGetError

    assert list_target_branch_stems(project, SYSTEM_DIR, ".jinja", "main") == []


def test_build_prompt_diff(project, repo_root):
    result = build_prompt_diff(project, f"{SYSTEM_DIR}/1.1.0.jinja", "main", repo_root)

    assert result == PromptDiff(
        key="chat/explain_code/system",
        old_version="1.0.1",
        new_version="1.1.0",
        diff=(
            "--- chat/explain_code/system/1.0.1.jinja\n"
            "+++ chat/explain_code/system/1.1.0.jinja\n"
            "@@ -1 +1 @@\n"
            "-You are a helpful assistant.\n"
            "+You are a very helpful assistant.\n"
        ),
    )
    project.files.raw.assert_called_once_with(
        file_path=f"{SYSTEM_DIR}/1.0.1.jinja", ref="main"
    )


def test_build_prompt_diff_prefers_predecessor_added_in_same_mr(project, repo_root):
    (repo_root / SYSTEM_DIR / "1.0.2.jinja").write_text("You are a kind assistant.\n")

    result = build_prompt_diff(
        project, f"{SYSTEM_DIR}/1.1.0.jinja", "main", repo_root, added_stems=["1.0.2"]
    )

    assert result is not None
    assert (result.old_version, result.new_version) == ("1.0.2", "1.1.0")
    assert "-You are a kind assistant." in result.diff
    project.files.raw.assert_not_called()


def test_build_prompt_diff_returns_none_for_brand_new_prompt(project, repo_root):
    project.repository_tree.return_value = []

    assert (
        build_prompt_diff(project, f"{SYSTEM_DIR}/1.1.0.jinja", "main", repo_root)
        is None
    )
    project.files.raw.assert_not_called()


def test_build_prompt_diff_returns_none_when_new_file_unreadable(project, tmp_path):
    result = build_prompt_diff(project, f"{SYSTEM_DIR}/1.1.0.jinja", "main", tmp_path)

    assert result is None
    project.repository_tree.assert_not_called()


def test_build_prompt_diff_returns_none_when_previous_fetch_fails(project, repo_root):
    project.files.raw.side_effect = gitlab.exceptions.GitlabGetError

    assert (
        build_prompt_diff(project, f"{SYSTEM_DIR}/1.1.0.jinja", "main", repo_root)
        is None
    )


@pytest.fixture(name="existing_note")
def existing_note_fixture():
    note = MagicMock()
    note.body = f"{SENTINEL}\n\n### Prompt version diff\n"
    return note


@pytest.fixture(name="unrelated_note")
def unrelated_note_fixture():
    note = MagicMock()
    note.body = "LGTM"
    return note


@pytest.fixture(name="diffs")
def diffs_fixture():
    return [PromptDiff("chat/explain_code/system", "1.0.0", "1.1.0", "-old\n+new\n")]


def test_sync_comment_creates_note_on_first_run(unrelated_note, diffs):
    mr = MagicMock()
    mr.notes.list.return_value = [unrelated_note]

    sync_comment(mr, diffs)

    mr.notes.create.assert_called_once_with({"body": build_comment_body(diffs)})
    unrelated_note.save.assert_not_called()


def test_sync_comment_updates_existing_note(existing_note, unrelated_note, diffs):
    mr = MagicMock()
    mr.notes.list.return_value = [unrelated_note, existing_note]

    sync_comment(mr, diffs)

    mr.notes.create.assert_not_called()
    assert existing_note.body == build_comment_body(diffs)
    existing_note.save.assert_called_once()


def test_sync_comment_deletes_stale_note_when_no_diffs(existing_note):
    mr = MagicMock()
    mr.notes.list.return_value = [existing_note]

    sync_comment(mr, [])

    existing_note.delete.assert_called_once()
    mr.notes.create.assert_not_called()


def test_sync_comment_noop_when_no_diffs_and_no_note(unrelated_note):
    mr = MagicMock()
    mr.notes.list.return_value = [unrelated_note]

    sync_comment(mr, [])

    mr.notes.create.assert_not_called()
    unrelated_note.delete.assert_not_called()


def test_run_end_to_end(project, repo_root):
    mr = MagicMock()
    mr.manager.gitlab.http_list.return_value = [
        {"new_path": f"{SYSTEM_DIR}/1.1.0.jinja", "new_file": True},
        {"new_path": "ai_gateway/prompts/registry.py", "new_file": False},
    ]
    mr.notes.list.return_value = []

    diffs = run(project, mr, "main", repo_root)

    assert [(d.key, d.old_version, d.new_version) for d in diffs] == [
        ("chat/explain_code/system", "1.0.1", "1.1.0")
    ]
    mr.notes.create.assert_called_once()
    body = mr.notes.create.call_args.args[0]["body"]
    assert SENTINEL in body
    assert "-You are a helpful assistant." in body
    assert "+You are a very helpful assistant." in body


def test_run_chains_two_versions_added_in_one_mr(project, repo_root):
    (repo_root / SYSTEM_DIR / "1.2.0.jinja").write_text("You are the best assistant.\n")
    mr = MagicMock()
    mr.manager.gitlab.http_list.return_value = [
        {"new_path": f"{SYSTEM_DIR}/1.2.0.jinja", "new_file": True},
        {"new_path": f"{SYSTEM_DIR}/1.1.0.jinja", "new_file": True},
    ]
    mr.notes.list.return_value = []

    diffs = run(project, mr, "main", repo_root)

    assert [(d.old_version, d.new_version) for d in diffs] == [
        ("1.0.1", "1.1.0"),
        ("1.1.0", "1.2.0"),
    ]
    assert "-You are a very helpful assistant." in diffs[1].diff
    project.files.raw.assert_called_once_with(
        file_path=f"{SYSTEM_DIR}/1.0.1.jinja", ref="main"
    )


def test_run_with_no_prompt_changes_removes_stale_note(project, existing_note):
    mr = MagicMock()
    mr.manager.gitlab.http_list.return_value = [
        {"new_path": "ai_gateway/prompts/registry.py", "new_file": False}
    ]
    mr.notes.list.return_value = [existing_note]

    assert run(project, mr, "main", Path("/nonexistent")) == []
    existing_note.delete.assert_called_once()
    project.repository_tree.assert_not_called()
