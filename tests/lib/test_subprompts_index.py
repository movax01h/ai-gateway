from pathlib import Path

import pytest
import yaml

from lib.subprompts_index import (
    SUBPROMPTS_INDEX,
    parse_subprompts_index,
    read_subprompts_index,
)


def _write_version(prompt_dir: Path) -> None:
    (prompt_dir / "base").mkdir(parents=True)
    (prompt_dir / "base" / "1.0.0.yml").write_text("name: x\n")


class TestParseSubpromptsIndex:
    def test_reads_entries_in_file_order(self):
        text = (
            "# the sub-prompts of my_flow\n"
            "---\n"
            "my_flow_summarize: summarize\n"
            "\n"
            "my_flow_fetch:   fetch  # trailing comment\n"
        )

        entries = parse_subprompts_index(text, "index.yml")

        assert list(entries.items()) == [
            ("my_flow_summarize", "summarize"),
            ("my_flow_fetch", "fetch"),
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "a: b\n",
            "---\na: b\n",
            "# comment\n---\na: b\n",
            "a: b # comment\nc:  d  \n",
            "a: b\r\nc: d\r\n",
        ],
    )
    def test_reads_what_a_yaml_library_reads(self, text: str):
        assert parse_subprompts_index(text, "index.yml") == (yaml.safe_load(text) or {})

    @pytest.mark.parametrize(
        "line",
        [
            "my_flow/fetch: fetch",
            "my_flow_fetch: ../other_feature",
            "my_flow_fetch: nested/dir",
            'my_flow_fetch: "fetch"',
            "my_flow_fetch:",
            "my_flow_fetch:fetch",
            "my_flow_fetch: fetch#not_a_comment",
            "my_flow_fetch:\tfetch",
            "my_flow_fetch: fetch\t# comment",
            "my_flow_fetch: fetch\t",
            "  my_flow_fetch: fetch",
            "my_flow_fetch: {dir: fetch}",
            "- my_flow_fetch",
            "---",
        ],
    )
    def test_rejects_other_syntax(self, line: str):
        with pytest.raises(ValueError, match=r"index\.yml:2: expected"):
            parse_subprompts_index(f"ok: ok\n{line}\n", "index.yml")

    def test_rejects_a_second_document_start(self):
        with pytest.raises(ValueError, match=r"index\.yml:2: expected"):
            parse_subprompts_index("---\n---\na: b\n", "index.yml")

    def test_rejects_a_repeated_prompt_id(self):
        with pytest.raises(ValueError, match="'a' occurs more than once"):
            parse_subprompts_index("a: x\na: y\n", "index.yml")


class TestReadSubpromptsIndex:
    def test_maps_each_prompt_id_to_its_directory(self, tmp_path: Path):
        _write_version(tmp_path / "fetch")
        (tmp_path / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")

        assert read_subprompts_index(tmp_path) == {"my_flow_fetch": tmp_path / "fetch"}

    def test_missing_index_declares_nothing(self, tmp_path: Path):
        _write_version(tmp_path / "fetch")

        assert read_subprompts_index(tmp_path) == {}

    def test_missing_directory_raises(self, tmp_path: Path):
        (tmp_path / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")

        with pytest.raises(ValueError, match="'my_flow_fetch' lists .*fetch"):
            read_subprompts_index(tmp_path)

    def test_directory_without_base_raises(self, tmp_path: Path):
        (tmp_path / "fetch" / "system").mkdir(parents=True)
        (tmp_path / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")

        with pytest.raises(ValueError, match=r"has no base/<version>\.yml file"):
            read_subprompts_index(tmp_path)

    def test_base_without_a_version_file_raises(self, tmp_path: Path):
        (tmp_path / "fetch" / "base").mkdir(parents=True)
        (tmp_path / "fetch" / "base" / "1.0.0.jinja").write_text("include only")
        (tmp_path / SUBPROMPTS_INDEX).write_text("my_flow_fetch: fetch\n")

        with pytest.raises(ValueError, match=r"has no base/<version>\.yml file"):
            read_subprompts_index(tmp_path)
