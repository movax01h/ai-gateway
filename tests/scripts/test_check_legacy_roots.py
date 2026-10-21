"""Tests for the legacy-roots lint."""

import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import check_legacy_roots
from scripts.check_legacy_roots import (
    ALLOWLIST,
    LEGACY_ROOTS,
    check,
    directories,
    feature_of,
    layout_b_target,
    load_allowlist,
    load_allowlist_at,
    main,
    nested_prompt_domains,
    parse_allowlist,
    scan,
)

PROMPTS_ROOT = "ai_gateway/prompts/definitions"
FLOWS_ROOT = "duo_workflow_service/agent_platform/v1/flows/configs"


def test_roots_under_test_are_the_configured_legacy_roots():
    assert set(LEGACY_ROOTS) == {PROMPTS_ROOT, FLOWS_ROOT}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _touch(repo: Path, rel: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture(name="repo")
def repo_fixture() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp).resolve()
        _git(repo, "init", "-q")
        yield repo


class TestScan:
    def test_lists_tracked_and_untracked_files_under_both_roots(self, repo):
        _touch(repo, f"{PROMPTS_ROOT}/foo/base/1.0.0.yml")
        _touch(repo, f"{PROMPTS_ROOT}/foo/labels.xml")
        _touch(repo, "ai/features/x/foo/prompts/base/1.0.0.yml")
        _git(repo, "add", "-A")
        _touch(repo, f"{FLOWS_ROOT}/bar/1.0.0.yml")

        assert scan(repo) == {
            f"{PROMPTS_ROOT}/foo/base/1.0.0.yml",
            f"{PROMPTS_ROOT}/foo/labels.xml",
            f"{FLOWS_ROOT}/bar/1.0.0.yml",
        }

    def test_skips_gitignored_files(self, repo):
        (repo / ".gitignore").write_text(".DS_Store\n__pycache__/\n")
        _touch(repo, f"{PROMPTS_ROOT}/.DS_Store")
        _touch(repo, f"{FLOWS_ROOT}/__pycache__/__init__.cpython-312.pyc")

        assert scan(repo) == set()

    def test_skips_tracked_files_deleted_from_disk(self, repo):
        path = f"{FLOWS_ROOT}/bar/1.0.0.yml"
        _touch(repo, path)
        _git(repo, "add", "-A")
        (repo / path).unlink()

        assert scan(repo) == set()

    def test_reports_symlinked_directory_as_one_entry(self, repo):
        _touch(repo, "elsewhere/base/1.0.0.yml")
        (repo / PROMPTS_ROOT).mkdir(parents=True)
        (repo / PROMPTS_ROOT / "linked").symlink_to(repo / "elsewhere")

        assert scan(repo) == {f"{PROMPTS_ROOT}/linked"}


def test_directories_groups_files_by_parent():
    files = {
        f"{PROMPTS_ROOT}/foo/base/1.0.1.yml",
        f"{PROMPTS_ROOT}/foo/base/1.0.0.yml",
        f"{FLOWS_ROOT}/__init__.py",
    }

    assert directories(files) == {
        f"{PROMPTS_ROOT}/foo/base": [
            f"{PROMPTS_ROOT}/foo/base/1.0.0.yml",
            f"{PROMPTS_ROOT}/foo/base/1.0.1.yml",
        ],
        FLOWS_ROOT: [f"{FLOWS_ROOT}/__init__.py"],
    }


def test_nested_prompt_domains_come_from_two_part_version_files():
    files = {
        f"{PROMPTS_ROOT}/chat/react/base/1.0.0.yml",
        f"{PROMPTS_ROOT}/foo/base/1.0.0.yml",
        f"{PROMPTS_ROOT}/common/developer/system/1.0.0.jinja",
        f"{FLOWS_ROOT}/bar/baz/qux/1.0.0.yml",
    }

    assert nested_prompt_domains(files) == {"chat"}


class TestAllowlist:
    def test_parse_skips_blank_lines_comments_and_trailing_slashes(self):
        assert parse_allowlist("# header\n\n  a/b/  \n# c\nd/e\n") == {"a/b", "d/e"}

    def test_load_reads_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = Path(tmp) / "allow.txt"
            allowlist.write_text("a/b\n")

            assert load_allowlist(allowlist) == {"a/b"}

    def test_load_at_ref_reads_committed_version(self, repo):
        (repo / ALLOWLIST).parent.mkdir(parents=True)
        (repo / ALLOWLIST).write_text("a/b\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")
        (repo / ALLOWLIST).write_text("a/b\nc/d\n")

        assert load_allowlist_at(repo, "HEAD") == {"a/b"}

    def test_load_at_ref_returns_none_when_file_absent(self, repo):
        _touch(repo, "other.txt")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")

        assert load_allowlist_at(repo, "HEAD") is None


class TestLayoutBTarget:
    @pytest.mark.parametrize(
        ("legacy", "feature", "target"),
        [
            (
                f"{PROMPTS_ROOT}/foo/base",
                "foo",
                "ai/features/<domain>/foo/prompts/base",
            ),
            (
                f"{PROMPTS_ROOT}/foo",
                "foo",
                "ai/features/<domain>/foo/prompts",
            ),
            (
                f"{PROMPTS_ROOT}/chat/react/partials/x",
                "chat",
                "ai/features/<domain>/chat/prompts/react/partials/x",
            ),
            (
                PROMPTS_ROOT,
                "<feature>",
                "ai/features/<domain>/<feature>/prompts",
            ),
            (
                f"{FLOWS_ROOT}/bar",
                "bar",
                "ai/features/<domain>/bar/config",
            ),
        ],
    )
    def test_maps_legacy_directory(self, legacy, feature, target):
        assert feature_of(legacy) == feature
        assert layout_b_target(legacy) == target

    @pytest.mark.parametrize(
        ("legacy", "feature", "target"),
        [
            (
                f"{PROMPTS_ROOT}/chat/react/partials/x",
                "chat/react",
                "ai/features/chat/react/prompts/partials/x",
            ),
            (
                f"{PROMPTS_ROOT}/chat/react",
                "chat/react",
                "ai/features/chat/react/prompts",
            ),
            (
                f"{PROMPTS_ROOT}/foo/base",
                "foo",
                "ai/features/<domain>/foo/prompts/base",
            ),
            (
                f"{FLOWS_ROOT}/chat/1.0.0",
                "chat",
                "ai/features/<domain>/chat/config/1.0.0",
            ),
        ],
    )
    def test_maps_directory_in_nested_domain(self, legacy, feature, target):
        nested = frozenset({"chat"})

        assert feature_of(legacy, nested) == feature
        assert layout_b_target(legacy, nested) == target

    def test_rejects_path_outside_legacy_roots(self):
        with pytest.raises(ValueError, match="not under a legacy root"):
            layout_b_target("ai/features/x/foo/prompts/base")


class TestCheck:
    def test_new_file_in_listed_directory_passes(self):
        directory = f"{PROMPTS_ROOT}/chat/fix_code/base"
        on_disk = {directory: [f"{directory}/1.0.0.yml", f"{directory}/3.0.0.yml"]}

        assert check(on_disk, {directory}) == []

    def test_new_directory_names_files_feature_and_target(self):
        directory = f"{FLOWS_ROOT}/foo"
        on_disk = {directory: [f"{directory}/1.0.0.yml", f"{directory}/1.1.0.yml"]}

        (message,) = check(on_disk, set())

        assert message.startswith(f"{directory}/: new directories are not allowed")
        assert "(added: 1.0.0.yml, 1.1.0.yml)" in message
        assert "Move the whole feature 'foo'" in message
        assert "ai/features/<domain>/foo/config/" in message
        assert "or goes to ai/shared/ when several features use it" in message
        assert str(ALLOWLIST) in message
        assert "docs/adding_and_moving_features.md" in message

    def test_new_directory_in_nested_domain_names_two_part_id(self):
        directory = f"{PROMPTS_ROOT}/chat/new_tool/base"
        on_disk = {directory: [f"{directory}/1.0.0.yml"]}

        (message,) = check(on_disk, set(), nested=frozenset({"chat"}))

        assert "Move the whole feature 'chat/new_tool'" in message
        assert "ai/features/chat/new_tool/prompts/base/" in message

    def test_stale_entry_names_allowlist(self):
        stale = f"{PROMPTS_ROOT}/foo/base"

        (message,) = check({}, {stale})

        assert message == (
            f"{stale}/: listed in {ALLOWLIST} but holds no files of its own "
            f"(subdirectories are listed separately). Remove that line."
        )

    def test_root_entry_is_stale_once_its_own_files_are_gone(self):
        on_disk = {f"{FLOWS_ROOT}/bar": [f"{FLOWS_ROOT}/bar/1.0.0.yml"]}

        (message,) = check(on_disk, {FLOWS_ROOT, f"{FLOWS_ROOT}/bar"})

        assert message.startswith(f"{FLOWS_ROOT}/: listed in {ALLOWLIST}")

    def test_messages_are_sorted_new_then_stale(self):
        new = [f"{FLOWS_ROOT}/b", f"{FLOWS_ROOT}/a"]
        stale = [f"{FLOWS_ROOT}/d", f"{FLOWS_ROOT}/c"]

        messages = check({d: [f"{d}/1.0.0.yml"] for d in new}, set(stale))

        assert [m.split("/:")[0] for m in messages] == sorted(new) + sorted(stale)

    def test_added_allowlist_line_fails_against_base(self):
        old = f"{PROMPTS_ROOT}/foo/base"
        added = f"{PROMPTS_ROOT}/foo/system"
        on_disk = {old: [f"{old}/1.0.0.yml"], added: [f"{added}/1.0.0.jinja"]}

        (message,) = check(on_disk, {old, added}, base={old})

        assert message.startswith(f"{added}/: added to {ALLOWLIST}")

    def test_removed_allowlist_line_is_allowed_against_base(self):
        kept = f"{PROMPTS_ROOT}/foo/base"
        gone = f"{PROMPTS_ROOT}/bar/base"

        assert check({kept: [f"{kept}/1.0.0.yml"]}, {kept}, base={kept, gone}) == []

    def test_lines_for_a_root_new_to_the_base_are_allowed(self):
        prompt = f"{PROMPTS_ROOT}/foo/base"
        on_disk = {prompt: [f"{prompt}/1.0.0.yml"], FLOWS_ROOT: [f"{FLOWS_ROOT}/x.py"]}

        assert check(on_disk, {prompt, FLOWS_ROOT}, base={prompt}) == []


class TestMain:
    def _run(self, repo, monkeypatch, capsys, files, allowlist, argv=None):
        for rel in files:
            _touch(repo, rel)
        (repo / ALLOWLIST).parent.mkdir(parents=True, exist_ok=True)
        (repo / ALLOWLIST).write_text("\n".join(allowlist) + "\n")
        monkeypatch.setattr(check_legacy_roots, "REPO_ROOT", repo)

        code = main(argv or [])

        return code, capsys.readouterr().err

    def test_returns_zero_when_tree_matches(self, repo, monkeypatch, capsys):
        directory = f"{FLOWS_ROOT}/bar"

        code, err = self._run(
            repo, monkeypatch, capsys, [f"{directory}/1.0.0.yml"], [directory]
        )

        assert code == 0
        assert err == ""

    def test_returns_one_and_prints_findings(self, repo, monkeypatch, capsys):
        new = f"{PROMPTS_ROOT}/foo/base"
        stale = f"{FLOWS_ROOT}/gone"

        code, err = self._run(repo, monkeypatch, capsys, [f"{new}/1.0.0.yml"], [stale])

        assert code == 1
        assert f"{new}/:" in err
        assert f"{stale}/:" in err

    def test_base_flag_reports_added_lines(self, repo, monkeypatch, capsys):
        old = f"{PROMPTS_ROOT}/foo/base"
        added = f"{PROMPTS_ROOT}/foo/system"
        self._run(repo, monkeypatch, capsys, [f"{old}/1.0.0.yml"], [old])
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")

        code, err = self._run(
            repo,
            monkeypatch,
            capsys,
            [f"{added}/1.0.0.jinja"],
            [old, added],
            ["--base", "HEAD"],
        )

        assert code == 1
        assert f"{added}/: added to {ALLOWLIST}" in err


def test_shipped_allowlist_matches_disk():
    repo_root = check_legacy_roots.REPO_ROOT

    assert (
        check(directories(scan(repo_root)), load_allowlist(repo_root / ALLOWLIST)) == []
    )
