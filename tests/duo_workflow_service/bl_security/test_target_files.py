"""Unit tests for parsing the scoped-scan file list."""

import pytest

from duo_workflow_service.bl_security.target_files import resolve_target_files


@pytest.mark.parametrize(
    "miss", [None, "", "   ", "\n", ",", " , ; ", True, False, [], ()]
)
def test_every_miss_is_no_scoped_scan(miss):
    assert resolve_target_files(miss) == []


class TestResolveTargetFiles:
    """A CI variable, a YAML literal and a real list must all land on the same path list."""

    def test_single_path_string(self):
        assert resolve_target_files("app/models/user.rb") == ["app/models/user.rb"]

    def test_separators_all_split(self):
        expected = ["a/b.rb", "c/d.go", "e/f.ex"]
        for text in (
            "a/b.rb,c/d.go,e/f.ex",
            "a/b.rb, c/d.go, e/f.ex",
            "a/b.rb\nc/d.go\ne/f.ex",
            "a/b.rb c/d.go e/f.ex",
            "a/b.rb;c/d.go;e/f.ex",
            "  a/b.rb ,\n c/d.go\t;e/f.ex  ",
        ):
            assert resolve_target_files(text) == expected, text

    def test_list_and_tuple_inputs(self):
        assert resolve_target_files(["a/b.rb", "c/d.go"]) == ["a/b.rb", "c/d.go"]
        assert resolve_target_files(("a/b.rb",)) == ["a/b.rb"]

    def test_json_array_string_is_decoded_not_split(self):
        # Splitting it would yield paths that match nothing: a silent clean scan.
        assert resolve_target_files('["a/b.rb","c/d.go"]') == [
            "a/b.rb",
            "c/d.go",
        ]
        assert resolve_target_files('[ "a/b.rb" ]') == ["a/b.rb"]
        assert resolve_target_files("[]") == []

    def test_malformed_json_falls_back_to_splitting(self):
        # A near-miss must not throw; it degrades to the separator path.
        assert resolve_target_files('["a/b.rb", "c/d.go"') == ["a/b.rb", "c/d.go"]

    def test_list_items_are_themselves_split(self):
        assert resolve_target_files(["a/b.rb,c/d.go", "e/f.ex"]) == [
            "a/b.rb",
            "c/d.go",
            "e/f.ex",
        ]

    def test_leading_dot_slash_and_slash_are_stripped(self):
        assert resolve_target_files("./a/b.rb") == ["a/b.rb"]
        assert resolve_target_files("/a/b.rb") == ["a/b.rb"]
        assert resolve_target_files(".//./a/b.rb") == ["a/b.rb"]

    @pytest.mark.parametrize(
        "path",
        [
            "[id].tsx",
            "pages/[id]",
            "app/[slug]/page.tsx",
            "routes/[...slug].ts",
            "[id]",
        ],
    )
    def test_brackets_that_belong_to_the_path_are_kept(self, path):
        # Next.js/Nuxt/SvelteKit dynamic routes put brackets in real file names.
        assert resolve_target_files(path) == [path]
        assert resolve_target_files([path]) == [path]
        assert resolve_target_files(f"a/b.rb,{path}") == ["a/b.rb", path]

    def test_unquoted_near_array_loses_only_its_outer_brackets(self):
        assert resolve_target_files("[a/b.rb, c/d.go]") == ["a/b.rb", "c/d.go"]

    def test_surrounding_quotes_are_stripped(self):
        assert resolve_target_files("'a/b.rb',\"c/d.go\"") == ["a/b.rb", "c/d.go"]

    def test_dedupes_preserving_caller_order(self):
        assert resolve_target_files("z/z.rb,a/a.rb,z/z.rb,./z/z.rb") == [
            "z/z.rb",
            "a/a.rb",
        ]
