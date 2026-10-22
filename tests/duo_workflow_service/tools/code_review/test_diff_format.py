from duo_workflow_service.tools.code_review.diff_format import (
    format_diff_lines,
    format_file_diff,
    format_file_diffs,
    format_renamed_files,
    is_binary_diff,
    split_unified_diff,
)


def test_format_diff_lines():
    """Raw diffs are parsed into the structured line format."""
    raw_diff = """@@ -1,3 +1,4 @@ class Calculator
def add(a, b)
-  a + b
+  a - b
end"""

    result = format_diff_lines(raw_diff)

    assert "<chunk_header>@@ -1,3 +1,4 @@ class Calculator</chunk_header>" in result
    assert (
        '<line type="context" old_line="1" new_line="1">def add(a, b)</line>' in result
    )
    assert '<line type="deleted" old_line="2" new_line="">  a + b</line>' in result
    assert '<line type="added" old_line="" new_line="2">  a - b</line>' in result
    assert '<line type="context" old_line="3" new_line="3">end</line>' in result


def test_format_diff_lines_with_special_characters():
    """Angle brackets and ampersands in code are passed through verbatim."""
    raw_diff = """@@ -1,1 +1,1 @@
-if x < 5 && y > 3:
+if x < 10 && y > 5:"""

    result = format_diff_lines(raw_diff)

    assert "<" in result
    assert ">" in result
    assert "&&" in result
    assert '<line type="deleted"' in result
    assert '<line type="added"' in result


def test_format_diff_lines_with_empty_lines():
    """A removed blank line still produces a line element."""
    raw_diff = """@@ -1,4 +1,4 @@
class Calculator
-
+  # New comment
end"""

    result = format_diff_lines(raw_diff)

    assert (
        '<line type="context" old_line="1" new_line="1">class Calculator</line>'
        in result
    )
    assert '<line type="deleted" old_line="2" new_line=""></line>' in result
    assert (
        '<line type="added" old_line="" new_line="2">  # New comment</line>' in result
    )


def test_format_diff_lines_binary_file():
    assert format_diff_lines("Binary files differ") == ""


def test_format_diff_lines_no_newline_at_end():
    raw_diff = """@@ -1,2 +1,2 @@
line 1
-line 2
\\ No newline at end of file
+line 2"""

    result = format_diff_lines(raw_diff)

    assert '<line type="context"' in result
    assert '<line type="deleted"' in result
    assert '<line type="nonewline"' in result
    assert "No newline at end of file" in result
    assert '<line type="added"' in result


def test_format_diff_lines_keeps_dashed_content_lines():
    """`---` inside a hunk is content (a YAML document separator), not file metadata."""
    raw_diff = """diff --git a/doc.yaml b/doc.yaml
index 1111111..2222222 100644
--- a/doc.yaml
+++ b/doc.yaml
@@ -1,4 +1,2 @@
 ---
-key: 1
----
-other: 2
+key: 2"""

    result = format_diff_lines(raw_diff)

    assert '<line type="context" old_line="1" new_line="1">---</line>' in result
    assert '<line type="deleted" old_line="3" new_line="">---</line>' in result
    assert "+++ b/doc.yaml" not in result
    assert "diff --git" not in result
    assert "index 1111111" not in result


def test_is_binary_diff_only_matches_git_markers():
    assert is_binary_diff("Binary files a/x.bin and b/x.bin differ")
    assert is_binary_diff("GIT binary patch\nliteral 0")
    assert not is_binary_diff('@@ -1,1 +1,1 @@\n+comment = "Binary files are skipped"')


def test_format_diff_lines_does_not_drop_content_for_binary_mention():
    """A diff whose content merely mentions the binary marker is still formatted."""
    raw_diff = '@@ -1,1 +1,1 @@\n+comment = "Binary files are skipped"'

    assert '<line type="added"' in format_diff_lines(raw_diff)


def test_format_file_diffs_wraps_each_file():
    result = format_file_diffs({"a.rb": "@@ -1,1 +1,1 @@\n+x = 1"})

    assert result.startswith('<file_diff filename="a.rb">')
    assert result.endswith("</file_diff>")
    assert '<line type="added" old_line="" new_line="1">x = 1</line>' in result


def test_format_renamed_files():
    assert format_renamed_files({}) == ""
    assert format_renamed_files({"new.rb": "old.rb"}) == (
        "<renamed_files>\n"
        '<file old_path="old.rb" new_path="new.rb"></file>\n'
        "</renamed_files>"
    )


def test_split_unified_diff_separates_files():
    raw = """diff --git a/a.rb b/a.rb
index 111..222 100644
--- a/a.rb
+++ b/a.rb
@@ -1,1 +1,1 @@
-a = 1
+a = 2
diff --git a/b.rb b/b.rb
index 333..444 100644
--- a/b.rb
+++ b/b.rb
@@ -1,1 +1,1 @@
-b = 1
+b = 2"""

    diffs, renamed = split_unified_diff(raw)

    assert sorted(diffs) == ["a.rb", "b.rb"]
    assert "-a = 1" in diffs["a.rb"]
    assert "-b = 1" in diffs["b.rb"]
    assert renamed == {}


def test_split_unified_diff_detects_renames_with_spaces_in_paths():
    raw = """diff --git a/old name.txt b/new name.txt
similarity index 100%
rename from old name.txt
rename to new name.txt"""

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["new name.txt"]
    assert renamed == {"new name.txt": "old name.txt"}


def test_split_unified_diff_decodes_quoted_utf8_paths():
    """`core.quotePath` octal-escapes the raw UTF-8 bytes of non-ASCII paths."""
    raw = r"""diff --git "a/na\303\257ve.py" "b/na\303\257ve.py"
index 111..222 100644
--- "a/na\303\257ve.py"
+++ "b/na\303\257ve.py"
@@ -1,1 +1,1 @@
-a = 1
+a = 2"""

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["naïve.py"]
    assert renamed == {}


def test_split_unified_diff_decodes_quoted_rename_paths():
    raw = r'''diff --git "a/na\303\257ve old.py" "b/na\303\257ve new.py"
similarity index 100%
rename from "na\303\257ve old.py"
rename to "na\303\257ve new.py"'''

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["naïve new.py"]
    assert renamed == {"naïve new.py": "naïve old.py"}


def test_split_unified_diff_handles_added_and_deleted_files():
    raw = """diff --git a/added.rb b/added.rb
new file mode 100644
index 0000000..111
--- /dev/null
+++ b/added.rb
@@ -0,0 +1,1 @@
+new = 1
diff --git a/gone.rb b/gone.rb
deleted file mode 100644
index 111..0000000
--- a/gone.rb
+++ /dev/null
@@ -1,1 +0,0 @@
-old = 1"""

    diffs, renamed = split_unified_diff(raw)

    assert sorted(diffs) == ["added.rb", "gone.rb"]
    assert renamed == {}


def test_split_unified_diff_handles_binary_files_without_path_lines():
    raw = """diff --git a/blob.bin b/blob.bin
index 111..222 100644
Binary files a/blob.bin and b/blob.bin differ"""

    diffs, _ = split_unified_diff(raw)

    assert list(diffs) == ["blob.bin"]
    assert is_binary_diff(diffs["blob.bin"])


def test_split_unified_diff_ignores_preamble():
    diffs, _ = split_unified_diff("warning: something\n")

    assert diffs == {}


def test_split_unified_diff_skips_a_chunk_with_no_recoverable_path():
    """A file whose path cannot be read is dropped, not keyed on an empty string."""
    raw = "diff --git \n@@ -1,1 +1,1 @@\n-a\n+b\n"

    diffs, renamed = split_unified_diff(raw)

    assert diffs == {}
    assert renamed == {}


def test_chunk_paths_stops_at_the_first_hunk():
    """`rename from` after a hunk is content, not metadata, so the scan has to stop at `@@`."""
    raw = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-rename from decoy.py\n"
        "+rename to other.py\n"
    )

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["a.py"]
    assert renamed == {}


def test_quoted_path_that_is_not_valid_utf8_is_left_escaped():
    """`\\377` is not a UTF-8 sequence; the escaped form beats raising on the user's filename."""
    raw = 'diff --git "a/\\377.py" "b/\\377.py"\n@@ -1,1 +1,1 @@\n-a\n+b\n'

    diffs, _ = split_unified_diff(raw)

    assert list(diffs) == ["\xff.py"]


def test_quoted_header_without_a_second_path_yields_nothing():
    raw = 'diff --git "a/only.py"\n@@ -1,1 +1,1 @@\n-a\n+b\n'

    diffs, renamed = split_unified_diff(raw)

    assert diffs == {}
    assert renamed == {}


def test_mode_change_without_path_lines_recovers_the_identical_path():
    """A pure mode change has no `---`/`+++`, so the path comes from the header alone."""
    raw = "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["run.sh"]
    assert renamed == {}


def test_header_with_differing_unprefixed_paths_falls_back_to_the_new_path():
    """The identical-path split cannot read a rename here, so only `b/` is recoverable."""
    raw = "diff --git a/old name.py b/new name.py\nold mode 100644\nnew mode 100755\n"

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["new name.py"]
    # No old path means no rename can be reported, rather than a wrong one.
    assert renamed == {}


def test_path_cannot_close_its_own_file_diff_element():
    """A directory `<` holding a file `file_diff>` is a legal path spelling `</file_diff>`.

    Unescaped it would close the element early and let the rest be read as a forged one. The
    prompt-security scanner does not cover it: it encodes `<goal>` and `<system>`, not the tags
    this format defines.
    """
    rendered = format_file_diff("</file_diff>", "@@ -1,1 +1,1 @@\n-a\n+b\n")

    assert rendered.count("</file_diff>") == 1
    assert 'filename="&lt;/file_diff>"' in rendered


def test_path_quotes_cannot_break_out_of_the_attribute():
    rendered = format_file_diff('a" onload="x', "@@ -1,1 +1,1 @@\n-a\n+b\n")

    assert 'filename="a&quot; onload=&quot;x"' in rendered


def test_characters_that_cannot_end_an_attribute_are_left_alone():
    """Nothing downstream decodes entities, so an escaped path is one the reviewer gets wrong.

    `&` and `'` are common in real paths and neither can close a double-quoted attribute.
    """
    rendered = format_file_diff("R&D/don't.md", "@@ -1 +1 @@\n+x")

    assert 'filename="R&D/don\'t.md"' in rendered


def test_renamed_file_paths_are_escaped_on_both_sides():
    rendered = format_renamed_files({"new<>.py": "old&.py"})

    assert 'old_path="old&.py"' in rendered
    assert 'new_path="new&lt;>.py"' in rendered


def test_ordinary_paths_are_untouched():
    """Escaping must be a no-op for every real path, since findings copy this value verbatim."""
    rendered = format_file_diff(
        "app/models/user_profile-v2.rb", "@@ -1,1 +1,1 @@\n-a\n+b\n"
    )

    assert 'filename="app/models/user_profile-v2.rb"' in rendered


def test_a_trailing_space_in_a_filename_survives():
    """Git appends a tab as the delimiter rather than quoting; stripping whitespace eats both."""
    raw = (
        "diff --git a/report.py  b/report.py \n"
        "--- a/report.py \t\n+++ b/report.py \t\n@@ -1 +1 @@\n-v1\n+v2\n"
    )

    diffs, _ = split_unified_diff(raw)

    assert list(diffs) == ["report.py "]


def test_a_leading_space_in_a_filename_survives():
    raw = (
        "diff --git a/ lead.py b/ lead.py\n"
        "--- a/ lead.py\t\n+++ b/ lead.py\t\n@@ -1 +1 @@\n-v1\n+v2\n"
    )

    diffs, _ = split_unified_diff(raw)

    assert list(diffs) == [" lead.py"]


def test_names_differing_only_by_trailing_space_do_not_collide():
    """Collapsing them would serve one file's patch under the other's name."""
    spaced = (
        "diff --git a/report.py  b/report.py \n"
        "--- a/report.py \t\n+++ b/report.py \t\n@@ -1 +1 @@\n-spaced v1\n+spaced v2\n"
    )
    plain = (
        "diff --git a/report.py b/report.py\n"
        "--- a/report.py\n+++ b/report.py\n@@ -1 +1 @@\n-plain v1\n+plain v2\n"
    )

    diffs, _ = split_unified_diff(spaced + plain)

    assert sorted(diffs) == ["report.py", "report.py "]
    assert "spaced v2" in diffs["report.py "]
    assert "plain v2" in diffs["report.py"]


def test_a_rename_keeps_whitespace_on_both_paths():
    """`rename from`/`rename to` run to end of line, so git appends no delimiter at all."""
    raw = (
        "diff --git a/old name.py  b/new name.py \n"
        "similarity index 100%\n"
        "rename from old name.py \n"
        "rename to new name.py \n"
    )

    diffs, renamed = split_unified_diff(raw)

    assert list(diffs) == ["new name.py "]
    assert renamed == {"new name.py ": "old name.py "}


def test_an_ordinary_path_is_unaffected():
    raw = "diff --git a/app/a.rb b/app/a.rb\n--- a/app/a.rb\n+++ b/app/a.rb\n@@ -1 +1 @@\n-a\n+b\n"

    diffs, _ = split_unified_diff(raw)

    assert list(diffs) == ["app/a.rb"]
