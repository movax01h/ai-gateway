from duo_workflow_service.tools.code_review.diff_format import (
    format_diff_lines,
    format_file_diffs,
    format_renamed_files,
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
