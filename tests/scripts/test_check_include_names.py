"""Tests for the include-names lint."""

import subprocess
from pathlib import Path

import pytest

from scripts import check_include_names
from scripts.check_include_names import (
    ALIASES_FILE,
    ALIASES_NAME,
    RETIRED_NAME,
    check,
    include_names,
    parse_aliases,
    parse_constant,
    parse_retired,
    scan,
    scan_at,
    served_names,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _write(repo: Path, rel: str, text: str = "x") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_constants(repo: Path, aliases: str, retired: str) -> None:
    _write(
        repo,
        str(ALIASES_FILE),
        f"{ALIASES_NAME}: dict[str, str] = {aliases}\n{RETIRED_NAME}: frozenset[str] = {retired}\n",
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


class TestIncludeNames:
    @pytest.mark.parametrize(
        ("path", "names"),
        [
            (
                "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja",
                {"common/naming/1.0.0.jinja"},
            ),
            (
                "ai/features/code/developer/prompts/common/naming/1.0.0.jinja",
                {
                    "developer/common/naming/1.0.0.jinja",
                    "code/developer/common/naming/1.0.0.jinja",
                },
            ),
            (
                "ai/shared/common/prompts/security/1.0.0.jinja",
                {"common/security/1.0.0.jinja"},
            ),
        ],
    )
    def test_maps_each_root_like_the_loaders(self, path: str, names: set[str]):
        assert include_names(path) == names

    @pytest.mark.parametrize(
        "path",
        [
            "ai_gateway/prompts/definitions/common/naming/1.0.0.yml",
            "ai/features/code/developer/config/1.0.0.yml",
            "ai/features/code/developer/prompts/AGENTS.md",
            "ai/features/code/developer/x.jinja",
            "ai/shared/common/x.jinja",
            "docs/x.jinja",
        ],
    )
    def test_non_template_paths_serve_nothing(self, path: str):
        assert include_names(path) == set()

    def test_served_names_unions_paths(self):
        paths = [
            "ai_gateway/prompts/definitions/a/1.0.0.jinja",
            "ai/shared/b/prompts/c/1.0.0.jinja",
        ]
        assert served_names(paths) == {"a/1.0.0.jinja", "b/c/1.0.0.jinja"}


class TestScan:
    def test_lists_tracked_and_untracked_files_under_the_roots(self, repo: Path):
        _write(repo, "ai/features/code/dev/prompts/system/1.0.0.jinja")
        _write(repo, "ai/shared/common/prompts/x/1.0.0.jinja")
        _write(repo, "docs/x.jinja")

        assert scan(repo) == {
            "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja",
            "ai/features/code/dev/prompts/system/1.0.0.jinja",
            "ai/shared/common/prompts/x/1.0.0.jinja",
        }

    def test_skips_tracked_files_deleted_from_disk(self, repo: Path):
        (repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja").unlink()
        assert scan(repo) == set()

    def test_scan_at_reads_the_committed_tree(self, repo: Path):
        (repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja").unlink()
        assert scan_at(repo, "HEAD") == {
            "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja"
        }

    def test_scan_at_returns_none_for_unknown_ref(self, repo: Path):
        assert scan_at(repo, "no-such-ref") is None


class TestParse:
    def test_parse_aliases_reads_annotated_dict_literal(self):
        source = (
            "X = 1\n"
            f"{ALIASES_NAME}: dict[str, str] = {{\n"
            '    "old/a.jinja": "new/a.jinja",\n'
            "}\n"
        )
        assert parse_aliases(source) == {"old/a.jinja": "new/a.jinja"}

    def test_parse_aliases_reads_plain_assignment(self):
        assert parse_aliases(f'{ALIASES_NAME} = {{"o": "n"}}') == {"o": "n"}

    def test_parse_constant_raises_when_absent(self):
        with pytest.raises(ValueError, match=ALIASES_NAME):
            parse_constant("OTHER = {}", ALIASES_NAME)

    def test_parse_aliases_rejects_non_dict(self):
        with pytest.raises(ValueError, match="dict"):
            parse_aliases(f"{ALIASES_NAME} = []")

    @pytest.mark.parametrize(
        ("literal", "expected"),
        [
            ("frozenset()", frozenset()),
            ('frozenset({"a.jinja"})', frozenset({"a.jinja"})),
            ('frozenset(["a.jinja", "b.jinja"])', frozenset({"a.jinja", "b.jinja"})),
            ('{"a.jinja"}', frozenset({"a.jinja"})),
        ],
    )
    def test_parse_retired_reads_frozenset_forms(
        self, literal: str, expected: frozenset[str]
    ):
        assert parse_retired(f"{RETIRED_NAME}: frozenset[str] = {literal}") == expected

    def test_parse_retired_rejects_other_calls(self):
        with pytest.raises(ValueError):
            parse_retired(f"{RETIRED_NAME} = list()")

    def test_real_file_defines_both_constants(self):
        source = (check_include_names.REPO_ROOT / ALIASES_FILE).read_text()
        aliases, retired = parse_aliases(source), parse_retired(source)
        assert all(
            k.endswith(".jinja") and v.endswith(".jinja") for k, v in aliases.items()
        )
        assert all(name.endswith(".jinja") for name in retired)


class TestCheck:
    def test_unchanged_tree_passes(self):
        current = {"common/naming/1.0.0.jinja"}
        assert check(current, {}, frozenset(), base=current) == []

    def test_removed_name_without_alias_fails(self):
        messages = check(
            {"dev/common/naming/1.0.0.jinja"},
            {},
            set(),
            base={"common/naming/1.0.0.jinja"},
        )
        assert len(messages) == 1
        assert messages[0].startswith("common/naming/1.0.0.jinja: no template serves")
        assert ALIASES_NAME in messages[0]
        assert RETIRED_NAME in messages[0]

    def test_removed_name_with_alias_passes(self):
        current = {"dev/common/naming/1.0.0.jinja"}
        aliases = {"common/naming/1.0.0.jinja": "dev/common/naming/1.0.0.jinja"}
        assert (
            check(current, aliases, frozenset(), base={"common/naming/1.0.0.jinja"})
            == []
        )

    def test_removed_name_listed_as_retired_passes(self):
        retired = frozenset({"common/naming/1.0.0.jinja"})
        assert check(set(), {}, retired, base={"common/naming/1.0.0.jinja"}) == []

    def test_added_name_passes(self):
        assert check({"new/1.0.0.jinja"}, {}, frozenset(), base=set()) == []

    def test_no_base_skips_removal_check(self):
        assert check(set(), {}, frozenset(), base=None) == []

    def test_alias_target_must_be_served(self):
        messages = check(set(), {"old.jinja": "gone.jinja"}, frozenset())
        assert len(messages) == 1
        assert "maps it to 'gone.jinja'" in messages[0]

    def test_alias_for_a_served_name_is_dead(self):
        messages = check(
            {"old.jinja", "new.jinja"}, {"old.jinja": "new.jinja"}, frozenset()
        )
        assert len(messages) == 1
        assert "the alias is dead" in messages[0]

    def test_name_in_both_lists_fails(self):
        messages = check(
            {"new.jinja"}, {"old.jinja": "new.jinja"}, frozenset({"old.jinja"})
        )
        assert len(messages) == 1
        assert "Keep one" in messages[0]

    def test_retired_name_still_served_fails(self):
        messages = check({"old.jinja"}, {}, frozenset({"old.jinja"}))
        assert len(messages) == 1
        assert messages[0].startswith(f"old.jinja: listed in {RETIRED_NAME}")


class TestMain:
    @pytest.fixture
    def lint_repo(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        _write_constants(repo, "{}", "frozenset()")
        monkeypatch.setattr(check_include_names, "REPO_ROOT", repo)
        return repo

    def test_passes_when_nothing_moved(self, lint_repo: Path, capsys):
        assert check_include_names.main(["--base", "HEAD"]) == 0
        assert capsys.readouterr().err == ""

    def test_fails_on_a_move_without_alias(self, lint_repo: Path, capsys):
        old = lint_repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja"
        old.unlink()
        _write(lint_repo, "ai/features/code/dev/prompts/common/naming/1.0.0.jinja")

        assert check_include_names.main(["--base", "HEAD"]) == 1
        assert (
            "common/naming/1.0.0.jinja: no template serves" in capsys.readouterr().err
        )

    def test_passes_on_a_move_with_alias(self, lint_repo: Path):
        (
            lint_repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja"
        ).unlink()
        _write(lint_repo, "ai/features/code/dev/prompts/common/naming/1.0.0.jinja")
        _write_constants(
            lint_repo,
            '{"common/naming/1.0.0.jinja": "dev/common/naming/1.0.0.jinja"}',
            "frozenset()",
        )

        assert check_include_names.main(["--base", "HEAD"]) == 0

    def test_passes_on_a_retired_name(self, lint_repo: Path):
        (
            lint_repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja"
        ).unlink()
        _write_constants(lint_repo, "{}", 'frozenset({"common/naming/1.0.0.jinja"})')

        assert check_include_names.main(["--base", "HEAD"]) == 0

    def test_unknown_base_only_warns(self, lint_repo: Path, capsys):
        (
            lint_repo / "ai_gateway/prompts/definitions/common/naming/1.0.0.jinja"
        ).unlink()

        assert check_include_names.main(["--base", "no-such-ref"]) == 0
        assert "unknown git ref" in capsys.readouterr().err
