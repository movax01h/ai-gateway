import base64
import json
from unittest.mock import AsyncMock, _Call, patch

import pytest
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool

from duo_workflow_service.agent_platform.v1.components.deterministic_step.validation import (
    validate_against_schema,
)
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.security.prompt_security import PromptSecurity
from duo_workflow_service.tools.code_review.build_review_merge_request_context import (
    BuildReviewMergeRequestContext,
    BuildReviewMergeRequestContextInput,
)


@pytest.fixture(name="mr_data")
def mr_data_fixture():
    return {
        "id": 123,
        "title": "Implement calculator method",
        "description": "Add subtract method to calculator",
        "target_branch": "main",
        "source_branch": "feature",
    }


@pytest.fixture(name="diffs_data")
def diffs_data_fixture():
    return [
        {
            "old_path": "calculator.rb",
            "new_path": "calculator.rb",
            "new_file": False,
            "generated_file": False,
            "renamed_file": False,
            "diff": "@@ -4,7 +4,7 @@ class Calculator\n   end\n \n   def subtract(a, b)\n-    # TODO: Implement\n+    a + b\n   end\n end",
        },
        {
            "old_path": "app.log",
            "new_path": "app.log",
            "new_file": False,
            "generated_file": False,
            "renamed_file": False,
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        },
        {
            "old_path": "generated.js",
            "new_path": "generated.js",
            "new_file": False,
            "generated_file": True,
            "renamed_file": False,
            "diff": "@@ -1,3 +1,3 @@\n-old\n+new",
        },
    ]


@pytest.fixture(name="diffs_data_with_renames")
def diffs_data_with_renames_fixture():
    return [
        {
            "old_path": "todo.rb",
            "new_path": "calculator.rb",
            "new_file": False,
            "generated_file": False,
            "renamed_file": True,
            "diff": "@@ -4,7 +4,7 @@ class Calculator\n   end\n \n   def subtract(a, b)\n-    # TODO: Implement\n+    a + b\n   end\n end",
        },
        {
            "old_path": "app.log",
            "new_path": "app.log",
            "new_file": False,
            "generated_file": False,
            "renamed_file": False,
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        },
        {
            "old_path": "generated.js",
            "new_path": "generated.js",
            "new_file": False,
            "generated_file": True,
            "renamed_file": False,
            "diff": "@@ -1,3 +1,3 @@\n-old\n+new",
        },
        {
            "old_path": "README.md",
            "new_path": "Calculator.md",
            "new_file": False,
            "generated_file": True,
            "renamed_file": True,
            "diff": "",  # This is a renamed file that has no content change
        },
    ]


@pytest.fixture(name="custom_instructions_yaml")
def custom_instructions_yaml_fixture():
    yaml_content = """---
instructions:
    - name: Ruby Code Quality
        fileFilters:
            - "**/*.rb"
        instructions: |
            1. Ensure proper error handling
            2. Follow Ruby naming conventions
"""
    return {"content": base64.b64encode(yaml_content.encode("utf-8")).decode("utf-8")}


def test_parse_and_format_diff(metadata):
    """Test that raw diffs are correctly parsed into structured format."""
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    raw_diff = """@@ -1,3 +1,4 @@ class Calculator
def add(a, b)
-  a + b
+  a - b
end"""

    result = tool._parse_and_format_diff(raw_diff)

    # Check chunk header
    assert "<chunk_header>@@ -1,3 +1,4 @@ class Calculator</chunk_header>" in result

    # Check context line
    assert (
        '<line type="context" old_line="1" new_line="1">def add(a, b)</line>' in result
    )

    # Check deleted line
    assert '<line type="deleted" old_line="2" new_line="">  a + b</line>' in result

    # Check added line
    assert '<line type="added" old_line="" new_line="2">  a - b</line>' in result

    # Check context line
    assert '<line type="context" old_line="3" new_line="3">end</line>' in result


def test_parse_and_format_diff_with_special_characters(metadata):
    """Test that special XML characters are properly escaped."""
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    raw_diff = """@@ -1,1 +1,1 @@
-if x < 5 && y > 3:
+if x < 10 && y > 5:"""

    result = tool._parse_and_format_diff(raw_diff)

    # Check that < > & are escaped
    assert "<" in result
    assert ">" in result
    assert "&&" in result
    assert '<line type="deleted"' in result
    assert '<line type="added"' in result


def test_parse_and_format_diff_with_empty_lines(metadata):
    """Test that empty lines are properly handled."""
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    raw_diff = """@@ -1,4 +1,4 @@
class Calculator
-
+  # New comment
end"""

    result = tool._parse_and_format_diff(raw_diff)

    # Check that empty lines are included
    assert (
        '<line type="context" old_line="1" new_line="1">class Calculator</line>'
        in result
    )
    assert '<line type="deleted" old_line="2" new_line=""></line>' in result
    assert (
        '<line type="added" old_line="" new_line="2">  # New comment</line>' in result
    )


def test_parse_and_format_diff_binary_file(metadata):
    """Test that binary files return empty string."""
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    raw_diff = "Binary files differ"

    result = tool._parse_and_format_diff(raw_diff)

    assert result == ""


def test_parse_and_format_diff_no_newline_at_end(metadata):
    """Test handling of 'No newline at end of file' marker."""
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    raw_diff = """@@ -1,2 +1,2 @@
line 1
-line 2
\\ No newline at end of file
+line 2"""

    result = tool._parse_and_format_diff(raw_diff)

    assert '<line type="context"' in result
    assert '<line type="deleted"' in result
    assert '<line type="nonewline"' in result
    assert "No newline at end of file" in result
    assert '<line type="added"' in result


@pytest.mark.asyncio
async def test_build_review_context_fetches_all_diff_pages(
    gitlab_client_mock,
    metadata,
    mr_data,
):
    """Diffs spanning multiple API pages are all included in the review context."""
    diffs_page_1 = [
        {
            "old_path": "file_a.rb",
            "new_path": "file_a.rb",
            "new_file": False,
            "generated_file": False,
            "renamed_file": False,
            "diff": "@@ -1 +1 @@\n-old\n+new",
        }
    ]
    diffs_page_2 = [
        {
            "old_path": "file_b.rb",
            "new_path": "file_b.rb",
            "new_file": False,
            "generated_file": False,
            "renamed_file": False,
            "diff": "@@ -1 +1 @@\n-old\n+new",
        }
    ]
    file_content = {"content": base64.b64encode(b"content").decode("utf-8")}
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(diffs_page_1),
                headers={"X-Next-Page": "2"},
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_page_2)),
            GitLabHttpResponse(status_code=200, body=json.dumps({"instructions": []})),
            GitLabHttpResponse(status_code=200, body=json.dumps(file_content)),
            GitLabHttpResponse(status_code=200, body=json.dumps(file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert '<file_diff filename="file_a.rb">' in response
    assert '<file_diff filename="file_b.rb">' in response


@pytest.mark.asyncio
async def test_build_review_context_basic_success(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    original_file_content = {
        "content": base64.b64encode(
            b"class Calculator\n  def subtract(a, b)\n    # TODO: Implement\n  end\nend"
        ).decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "Here are the merge request details for you to review:" in response
    assert "<input>" in response
    assert "<mr_title>" in response
    assert "Implement calculator method" in response
    assert "<mr_description>" in response
    assert "Add subtract method to calculator" in response
    assert "<git_diffs>" in response

    # Check new structured diff format
    assert '<file_diff filename="calculator.rb">' in response
    assert "<chunk_header>" in response
    assert '<line type="context"' in response
    assert '<line type="deleted"' in response
    assert '<line type="added"' in response
    assert "</file_diff>" in response

    # Verify excluded files are not present
    assert "app.log" not in response
    assert "generated.js" not in response

    assert "<original_files>" in response
    assert "</input>" in response


@pytest.mark.asyncio
async def test_build_review_context_with_renames(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data_with_renames,
):
    original_file_content = {
        "content": base64.b64encode(
            b"class Calculator\n  def subtract(a, b)\n    # TODO: Implement\n  end\nend"
        ).decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(diffs_data_with_renames)
            ),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "Here are the merge request details for you to review:" in response
    assert "<input>" in response
    assert "<mr_title>" in response
    assert "Implement calculator method" in response
    assert "<mr_description>" in response
    assert "Add subtract method to calculator" in response
    assert "<git_diffs>" in response

    # Check new structured diff format
    assert '<file_diff filename="calculator.rb">' in response
    assert "<chunk_header>" in response
    assert '<line type="context"' in response
    assert '<line type="deleted"' in response
    assert '<line type="added"' in response
    assert "</file_diff>" in response

    assert "<renamed_files>" in response
    assert "</renamed_files>" in response

    # Check renamed files
    assert '<file old_path="README.md" new_path="Calculator.md"></file>' in response
    assert '<file old_path="todo.rb" new_path="calculator.rb"></file>' in response

    # Verify excluded files are not present
    assert "app.log" not in response
    assert "generated.js" not in response

    assert "<original_files>" in response
    assert "</input>" in response


@pytest.mark.asyncio
async def test_build_review_context_only_diffs(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=404, body=json.dumps({"message": "404 Not Found"})
            ),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id="test%2Fproject", merge_request_iid=123, only_diffs=True
    )

    assert "Here are the merge request details for you to review:" in response
    assert "<input>" in response
    assert "<mr_title>" in response
    assert "Implement calculator method" in response
    assert "<git_diffs>" in response

    # Check structured format
    assert '<file_diff filename="calculator.rb">' in response
    assert "<line type=" in response

    assert "<original_files>" not in response
    assert "<custom_instructions>" not in response
    assert gitlab_client_mock.aget.call_count == 3


@pytest.mark.asyncio
@patch("yaml.safe_load")
async def test_build_review_context_only_diffs_with_custom_instructions(
    mock_yaml_load,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
):
    """Test that custom instructions are included when only_diffs=True."""
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "Ruby Code Quality",
                "fileFilters": ["*.rb"],
                "instructions": "1. Ensure proper error handling\n2. Follow Ruby naming conventions",
            }
        ]
    }

    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
        ]
    )

    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id="test%2Fproject", merge_request_iid=123, only_diffs=True
    )

    # Should include custom instructions
    assert "<custom_instructions>" in response
    assert "Ruby Code Quality" in response

    # Should NOT include original files
    assert "<original_files>" not in response

    # Should include diffs
    assert '<file_diff filename="calculator.rb">' in response

    # Verify only 3 API calls (MR data, diffs, custom instructions)
    assert gitlab_client_mock.aget.call_count == 3


@pytest.mark.asyncio
async def test_build_review_context_skips_large_files(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    large_file_content = "\n".join([f"line {i}" for i in range(10001)])
    large_file_encoded = {
        "content": base64.b64encode(large_file_content.encode("utf-8")).decode("utf-8")
    }

    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(large_file_encoded)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<original_files>" not in response
    assert '<file_diff filename="calculator.rb">' in response


@pytest.mark.asyncio
@patch("yaml.safe_load")
async def test_build_review_context_with_custom_instructions(
    mock_yaml_load,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
):
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "Ruby Code Quality",
                "fileFilters": ["*.rb"],
                "instructions": "1. Ensure proper error handling\n2. Follow Ruby naming conventions",
            }
        ]
    }
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "custom_instructions" in response
    assert "Ruby Code Quality" in response
    assert "Apply these additional review instructions to matching files:" in response
    assert "According to custom instructions in" in response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag_kwargs,hint_expected",
    [
        ({}, True),
        ({"include_instruction_format_hint": False}, False),
        # A flow config can only pass a string literal, so the args schema has to
        # coerce it. Invoked through ainvoke to exercise that coercion.
        ({"include_instruction_format_hint": "false"}, False),
    ],
)
@patch(
    "duo_workflow_service.tools.code_review.build_review_merge_request_context.yaml.safe_load"
)
async def test_build_review_context_instruction_format_hint(
    mock_yaml_load,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
    flag_kwargs,
    hint_expected,
):
    """A flow that renders the attribution itself disables the format hint."""
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "Ruby Code Quality",
                "fileFilters": ["*.rb"],
                "instructions": "1. Ensure proper error handling",
            }
        ]
    }
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool.ainvoke(
        {
            "project_id": "test%2Fproject",
            "merge_request_iid": 123,
            **flag_kwargs,
        }
    )

    assert "Ruby Code Quality" in response
    assert "Apply these additional review instructions to matching files:" in response
    assert ("According to custom instructions in" in response) is hint_expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag_kwargs,expected",
    [
        ({}, False),
        ({"include_changed_files_list": True}, True),
        ({"include_changed_files_list": False}, False),
        ({"include_changed_files_list": "true"}, True),
    ],
)
async def test_build_review_context_changed_files_list(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    flag_kwargs,
    expected,
):
    """The checklist lists only reviewable paths, and stays out of the default output."""
    original_file_content = {
        "content": base64.b64encode(b"original content").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool.ainvoke(
        {"project_id": "test%2Fproject", "merge_request_iid": 123, **flag_kwargs}
    )

    if expected:
        assert "<changed_files>\n- calculator.rb\n</changed_files>" in response
    else:
        assert "<changed_files>" not in response


@pytest.mark.asyncio
async def test_build_review_context_changed_files_list_includes_pure_renames(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data_with_renames,
):
    """A pure rename has no reviewable diff but is still a changed file; a rename with changes is listed once."""
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(diffs_data_with_renames)
            ),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id="test%2Fproject",
        merge_request_iid=123,
        include_changed_files_list=True,
    )

    assert (
        "<changed_files>\n- calculator.rb\n- Calculator.md\n</changed_files>"
        in response
    )


@pytest.mark.asyncio
async def test_build_review_context_with_url(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    original_file_content = {
        "content": base64.b64encode(b"original content").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            Exception("Custom instructions not found"),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/merge_requests/123"
    )

    assert "Implement calculator method" in response
    assert "<input>" in response
    assert "<file_diff filename=" in response


@pytest.mark.asyncio
@patch("yaml.safe_load")
async def test_build_review_context_no_matching_custom_instructions(
    mock_yaml_load,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
):
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "JavaScript Rules",
                "fileFilters": ["*.js", "**/*.ts"],
                "instructions": "JavaScript and TypeScript specific rules",
            }
        ]
    }
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<custom_instructions>" not in response


@pytest.mark.asyncio
@patch("yaml.safe_load")
async def test_build_review_context_nested_vs_root_patterns(
    mock_yaml_load,
    gitlab_client_mock,
    metadata,
    mr_data,
    custom_instructions_yaml,
):
    nested_diffs_data = [
        {
            "diff": "@@ -1,3 +1,4 @@ class Calculator",
            "new_path": "calculator.rb",
            "old_path": "calculator.rb",
            "new_file": False,
            "generated_file": False,
        },
        {
            "diff": "@@ -1,3 +1,4 @@ module Models",
            "new_path": "app/models/user.rb",
            "old_path": "app/models/user.rb",
            "new_file": False,
            "generated_file": False,
        },
    ]
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "Nested Ruby Files Only",
                "fileFilters": ["**/*.rb"],
                "instructions": "Rules for nested Ruby files",
            },
            {
                "name": "All Ruby Files",
                "fileFilters": ["*.rb", "**/*.rb"],
                "instructions": "Rules for all Ruby files",
            },
        ]
    }
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(nested_diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "Nested Ruby Files Only" in response
    assert "All Ruby Files" in response
    assert "<custom_instructions>" in response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            BuildReviewMergeRequestContextInput(project_id=42, merge_request_iid=123),
            "Build review context for merge request !123 in project 42",
        ),
        (
            BuildReviewMergeRequestContextInput(
                project_id=42, merge_request_iid=123, only_diffs=True
            ),
            "Build review context for merge request !123 in project 42 (diffs only)",
        ),
        (
            BuildReviewMergeRequestContextInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Build review context for merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
        (
            BuildReviewMergeRequestContextInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                only_diffs=True,
            ),
            "Build review context for merge request https://gitlab.com/namespace/project/-/merge_requests/42 (diffs only)",
        ),
        (
            BuildReviewMergeRequestContextInput(
                project_id=42, merge_request_iid=123, lightweight=True
            ),
            "Build review context for merge request !123 in project 42 (lightweight)",
        ),
    ],
)
def test_build_review_context_format_display_message(input_data, expected_message):
    tool = BuildReviewMergeRequestContext(description="Build review context")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_build_review_context_validation_error(gitlab_client_mock, metadata):
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    with pytest.raises(ToolException):
        await tool._arun()
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_build_review_context_exception(gitlab_client_mock, metadata):
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    with pytest.raises(Exception, match=error_message):
        await tool._arun(project_id="test", merge_request_iid=123)


@pytest.mark.asyncio
async def test_build_review_context_no_files_content(
    gitlab_client_mock,
    metadata,
    mr_data,
):
    new_files_diffs = [
        {
            "old_path": "",
            "new_path": "new_file.rb",
            "new_file": True,
            "generated_file": False,
            "diff": "@@ -0,0 +1,3 @@\n+class NewFile\n+end",
        }
    ]

    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(new_files_diffs)),
            Exception("Custom instructions not found"),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<original_files>" not in response
    assert '<file_diff filename="new_file.rb">' in response
    assert '<line type="added"' in response


def _aget_path(call: _Call) -> str:
    """Return the request path of a gitlab_client.aget call (positional or kwarg)."""
    if call.args:
        return call.args[0]
    return call.kwargs.get("path", "")


@pytest.mark.asyncio
@patch("yaml.safe_load")
async def test_build_review_context_reads_instructions_from_default_branch(
    mock_yaml_load,
    gitlab_client_mock,
    project_mock,
    metadata,
    diffs_data,
    custom_instructions_yaml,
):
    """Custom instructions must be read from the project's default branch, not the attacker-controllable MR target
    branch (security regression for #601482).

    Original file contents stay on the target branch (data under review).
    """
    mock_yaml_load.return_value = {
        "instructions": [
            {
                "name": "Ruby Code Quality",
                "fileFilters": ["*.rb"],
                "instructions": "Follow Ruby naming conventions",
            }
        ]
    }
    mr_data = {
        "id": 123,
        "title": "Title",
        "description": "Description",
        "target_branch": "not-main",
        "source_branch": "feature",
    }
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=200, body=json.dumps(custom_instructions_yaml)
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "Ruby Code Quality" in response

    instruction_calls = [
        call
        for call in gitlab_client_mock.aget.call_args_list
        if "mr-review-instructions" in _aget_path(call)
    ]
    assert len(instruction_calls) == 1
    # Read from the default branch, NOT the MR target branch ("not-main").
    assert instruction_calls[0].kwargs["params"] == {"ref": "main"}

    original_file_calls = [
        call
        for call in gitlab_client_mock.aget.call_args_list
        if "repository/files" in _aget_path(call)
        and "mr-review-instructions" not in _aget_path(call)
    ]
    assert original_file_calls
    # Original files are still fetched from the target branch (data under review).
    assert all(
        call.kwargs["params"] == {"ref": "not-main"} for call in original_file_calls
    )


@pytest.mark.asyncio
async def test_build_review_context_no_default_branch_skips_instructions(
    gitlab_client_mock,
    project_mock,
    metadata,
    mr_data,
    diffs_data,
):
    """When the project has no resolvable default branch, no instructions file is fetched and custom instructions
    resolve to empty."""
    project_mock["default_branch"] = None
    original_file_content = {
        "content": base64.b64encode(b"class Calculator\nend").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<custom_instructions>" not in response
    # No attempt to fetch the instructions file when default branch is unknown.
    assert not [
        call
        for call in gitlab_client_mock.aget.call_args_list
        if "mr-review-instructions" in _aget_path(call)
    ]


class TestFormatDiffLinks:
    WEB_URL = "https://gitlab.com/group/project/-/merge_requests/1"

    def test_returns_empty_when_no_web_url(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        result = tool._format_diff_links(
            {"web_url": "", "diff_refs": {"head_sha": "abc123"}}, {"app.rb": "diff"}
        )
        assert result == ""

    def test_returns_empty_when_web_url_missing(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        result = tool._format_diff_links({}, {"app.rb": "diff"})
        assert result == ""

    def test_returns_empty_when_no_head_sha(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        result = tool._format_diff_links({"web_url": self.WEB_URL}, {"app.rb": "diff"})
        assert result == ""

    def test_returns_empty_when_diffs_empty(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        result = tool._format_diff_links(
            {"web_url": self.WEB_URL, "diff_refs": {"head_sha": "abc123"}}, {}
        )
        assert result == ""

    def test_single_file_pins_to_head_sha(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        mr_data = {"web_url": self.WEB_URL, "diff_refs": {"head_sha": "abc123def"}}
        diffs = {"app/models/user.rb": "diff content"}

        result = tool._format_diff_links(mr_data, diffs)

        assert result.startswith("<diff_links>")
        assert result.endswith("</diff_links>")
        assert 'path="app/models/user.rb"' in result
        assert (
            'url="https://gitlab.com/group/project/-/blob/abc123def/'
            'app/models/user.rb"' in result
        )
        # No anchor-hash replication of monolith internals.
        assert "diffs#" not in result
        assert "hash=" not in result

    def test_falls_back_to_top_level_sha(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        mr_data = {"web_url": self.WEB_URL, "sha": "deadbeef"}
        diffs = {"app.rb": "diff content"}

        result = tool._format_diff_links(mr_data, diffs)

        assert 'url="https://gitlab.com/group/project/-/blob/deadbeef/app.rb"' in result

    def test_head_sha_takes_precedence_over_sha(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        mr_data = {
            "web_url": self.WEB_URL,
            "sha": "deadbeef",
            "diff_refs": {"head_sha": "abc123def"},
        }
        diffs = {"app.rb": "diff content"}

        result = tool._format_diff_links(mr_data, diffs)

        assert "/blob/abc123def/" in result
        assert "deadbeef" not in result

    def test_encodes_special_chars_in_path(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        mr_data = {"web_url": self.WEB_URL, "diff_refs": {"head_sha": "abc123"}}
        diffs = {"dir/a file.rb": "diff content"}

        result = tool._format_diff_links(mr_data, diffs)

        # Slashes are preserved, spaces are percent-encoded.
        assert "/blob/abc123/dir/a%20file.rb" in result
        assert 'path="dir/a file.rb"' in result

    def test_multiple_files(self, metadata):
        tool = BuildReviewMergeRequestContext(metadata=metadata)
        mr_data = {"web_url": self.WEB_URL, "diff_refs": {"head_sha": "abc123"}}
        diffs = {"file_a.py": "diff", "file_b.py": "diff"}

        result = tool._format_diff_links(mr_data, diffs)

        assert result.count("<file ") == 2
        for path in diffs:
            assert f'path="{path}"' in result


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.tools.code_review.build_review_merge_request_context"
    ".supports_group_level_custom_instructions",
    return_value=True,
)
async def test_group_level_custom_instructions_decodes_encoded_project_path(
    _mock_supports,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    original_file_content = {
        "content": base64.b64encode(b"original content").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps({"instructions": []})),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    instructions_calls = [
        call
        for call in gitlab_client_mock.aget.call_args_list
        if "/custom_instructions" in _aget_path(call)
    ]
    assert len(instructions_calls) == 1
    instructions_call = instructions_calls[0]
    assert (
        _aget_path(instructions_call)
        == "/api/v4/ai/duo_workflows/code_review/custom_instructions"
    )
    assert instructions_call.kwargs["params"]["project_id"] == "test/project"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.tools.code_review.build_review_merge_request_context"
    ".supports_group_level_custom_instructions",
    return_value=True,
)
async def test_group_level_custom_instructions_failure_is_not_fatal(
    _mock_supports,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    original_file_content = {
        "content": base64.b64encode(b"original content").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
            GitLabHttpResponse(
                status_code=404, body='{"message":"404 Project Not Found"}'
            ),
            GitLabHttpResponse(status_code=200, body=json.dumps(original_file_content)),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert '<file_diff filename="calculator.rb">' in response
    assert "Tool runtime exception" not in response


# --- Incremental diff annotation -------------------------------------------------
#
# The merge request diff and the compare against the baseline are both taken against
# the current head, so the two are matched on (path, new_line).
#
# The invariant these tests protect: only the "delta" state may narrow the model's
# focus. Every other state must leave every line unmarked. Reporting "delta" over an
# empty or incomplete marker set would quietly cut review coverage in production
# without failing anything a human would notice.

BASELINE_SHA = "aaaa1111"
HEAD_SHA = "bbbb2222"

# calculator.rb in the diffs_data fixture adds "    a + b" at new_line 7.
CALCULATOR_COMPARE_DIFF = {
    "old_path": "calculator.rb",
    "new_path": "calculator.rb",
    "diff": "@@ -4,7 +4,7 @@ class Calculator\n   end\n \n   def subtract(a, b)\n-    # TODO: Implement\n+    a + b\n   end\n end",
}


@pytest.fixture(name="mr_data_with_refs")
def mr_data_with_refs_fixture(mr_data):
    return {**mr_data, "diff_refs": {"head_sha": HEAD_SHA}, "source_project_id": 1}


@pytest.fixture(name="two_file_diffs")
def two_file_diffs_fixture():
    """Two reviewable files that both add a line at new_line 7.

    The shared line number is what exposes matching on new_line alone.
    """
    hunk = "@@ -4,7 +4,7 @@\n   end\n \n   def thing(a, b)\n-    # TODO\n+    a + b\n   end\n end"
    return [
        {"old_path": path, "new_path": path, "generated_file": False, "diff": hunk}
        for path in ("alpha.rb", "beta.rb")
    ]


@pytest.fixture(name="two_hunk_diffs")
def two_hunk_diffs_fixture():
    """One file adding a line in each of two hunks, at new_line 7 and new_line 82."""
    return [
        {
            "old_path": "calculator.rb",
            "new_path": "calculator.rb",
            "generated_file": False,
            "diff": (
                "@@ -4,7 +4,7 @@ class Calculator\n   end\n \n   def subtract(a, b)\n-    # TODO: Implement\n+    a + b\n   end\n end\n"
                "@@ -80,3 +80,4 @@ class Calculator\n   def divide(a, b)\n     a / b\n+    raise if b.zero?\n   end"
            ),
        }
    ]


def _aget_responses(mr_data, diffs_data, *extra):
    """Build the aget side effects in the order _build_context issues them.

    The order is load-bearing: merge request metadata, then diffs, then custom
    instructions, then the compare (only when a baseline is supplied), then the
    original file contents. If the compare ever moves earlier, these tests hand it
    the custom-instructions exception and report "failed" for the wrong reason.
    """
    return [
        GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
        GitLabHttpResponse(status_code=200, body=json.dumps(diffs_data)),
        Exception("Custom instructions not found"),
        *extra,
    ]


def _file_content():
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            {"content": base64.b64encode(b"class Calculator\nend").decode()}
        ),
    )


def _compare_calls(gitlab_client_mock):
    return [
        str(call)
        for call in gitlab_client_mock.aget.call_args_list
        if "repository/compare" in str(call)
    ]


def _marker_count(response):
    """Count marked diff lines, not the review_scope prose naming the attribute."""
    return response.count('since_last_review="true">')


NO_COMPARE = object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        # Five other flows call this tool and none of their prompts describe the
        # block, so without a baseline it must be absent, not present and inert.
        {},
        # Lightweight mode returns file paths only, so there is nothing to mark.
        {"lightweight": True, "baseline_sha": BASELINE_SHA},
    ],
)
async def test_review_scope_omits_the_block_entirely(
    gitlab_client_mock, metadata, mr_data_with_refs, diffs_data, kwargs
):
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(mr_data_with_refs, diffs_data, _file_content())
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id=1, merge_request_iid=123, **kwargs)

    assert "review_scope" not in response
    assert _marker_count(response) == 0
    assert _compare_calls(gitlab_client_mock) == []


@pytest.mark.asyncio
async def test_review_scope_marks_only_the_hunk_that_changed(
    gitlab_client_mock, metadata, mr_data_with_refs, two_hunk_diffs
):
    # The canonical second-review shape: an older hunk the model already saw, and a
    # newer one it has not. Only the newer hunk may be marked.
    compare = {
        "diffs": [
            {
                "new_path": "calculator.rb",
                "diff": "@@ -80,3 +80,4 @@ class Calculator\n   def divide(a, b)\n     a / b\n+    raise if b.zero?\n   end",
            }
        ]
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            mr_data_with_refs,
            two_hunk_diffs,
            GitLabHttpResponse(status_code=200, body=json.dumps(compare)),
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id=1, merge_request_iid=123, baseline_sha=BASELINE_SHA
    )

    assert '<review_scope state="delta">' in response
    assert _marker_count(response) == 1
    assert (
        '<line type="added" old_line="" new_line="82" since_last_review="true">    raise if b.zero?</line>'
        in response
    )
    assert '<line type="added" old_line="" new_line="7">    a + b</line>' in response
    # One compare for the whole merge request, not one per changed file.
    assert len(_compare_calls(gitlab_client_mock)) == 1


@pytest.mark.asyncio
async def test_review_scope_matches_on_file_path_as_well_as_line(
    gitlab_client_mock, metadata, mr_data_with_refs, two_file_diffs
):
    # Both files add a line at new_line 7. Only alpha.rb is in the compare, so
    # matching on the line number alone would wrongly mark beta.rb too.
    compare = {"diffs": [{"new_path": "alpha.rb", "diff": two_file_diffs[0]["diff"]}]}
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            mr_data_with_refs,
            two_file_diffs,
            GitLabHttpResponse(status_code=200, body=json.dumps(compare)),
            _file_content(),
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id=1, merge_request_iid=123, baseline_sha=BASELINE_SHA
    )

    assert _marker_count(response) == 1
    alpha = response.split('<file_diff filename="alpha.rb">')[1].split("</file_diff>")[
        0
    ]
    beta = response.split('<file_diff filename="beta.rb">')[1].split("</file_diff>")[0]
    assert 'new_line="7" since_last_review="true"' in alpha
    assert "since_last_review" not in beta


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_head_sha,baseline,compare,expected",
    [
        # Nothing to compare against.
        (False, BASELINE_SHA, NO_COMPARE, "failed"),
        # The head has not moved since the last review.
        (True, HEAD_SHA, NO_COMPARE, "no_new_lines"),
        # A force push orphans the baseline commit and the compare 404s.
        (
            True,
            BASELINE_SHA,
            GitLabHttpResponse(
                status_code=404, body='{"message":"404 Commit Not Found"}'
            ),
            "failed",
        ),
        # The client raises rather than returning a response.
        (True, BASELINE_SHA, RuntimeError("connection reset"), "failed"),
        # A timed-out compare returns only part of what changed.
        (
            True,
            BASELINE_SHA,
            {"diffs": [CALCULATOR_COMPARE_DIFF], "compare_timeout": True},
            "truncated",
        ),
        # A patch dropped for exceeding the size limit.
        (
            True,
            BASELINE_SHA,
            {
                "diffs": [
                    CALCULATOR_COMPARE_DIFF,
                    {"old_path": "huge.rb", "new_path": "huge.rb", "diff": ""},
                ]
            },
            "truncated",
        ),
        # A merge from the target branch, plus files this review filters out.
        # Neither is reviewable here, so this must not be reported as a delta.
        (
            True,
            BASELINE_SHA,
            {
                "diffs": [
                    {
                        "new_path": "unrelated/other.rb",
                        "diff": "@@ -1,1 +1,2 @@\n ctx\n+brand new",
                    },
                    {"new_path": "app.log", "diff": "@@ -1,1 +1,2 @@\n ctx\n+log line"},
                ]
            },
            "no_new_lines",
        ),
    ],
)
async def test_review_scope_falls_back_to_a_full_review(
    gitlab_client_mock,
    metadata,
    mr_data,
    mr_data_with_refs,
    diffs_data,
    has_head_sha,
    baseline,
    compare,
    expected,
):
    # Only "delta" may narrow the model's focus. Every other state has to leave the
    # whole diff unmarked, so an absent marker is never read as "already reviewed".
    extra = []
    if compare is not NO_COMPARE:
        extra.append(
            compare
            if isinstance(compare, (GitLabHttpResponse, Exception))
            else GitLabHttpResponse(status_code=200, body=json.dumps(compare))
        )

    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            mr_data_with_refs if has_head_sha else mr_data,
            diffs_data,
            *extra,
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id=1, merge_request_iid=123, baseline_sha=baseline
    )

    assert f'<review_scope state="{expected}">' in response
    assert _marker_count(response) == 0
    # The review still runs over the full diff rather than failing the step.
    assert '<file_diff filename="calculator.rb">' in response
    if compare is NO_COMPARE:
        assert _compare_calls(gitlab_client_mock) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_project_id,expected_project",
    [
        # For a fork, both commits live in the source project.
        (99, 99),
        # Older API responses omit source_project_id. Sending the compare to
        # /projects/None/ would 404 and silently disable the feature on every review.
        (None, 7),
    ],
)
async def test_review_scope_compares_in_the_project_holding_both_commits(
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    source_project_id,
    expected_project,
):
    data = {**mr_data, "diff_refs": {"head_sha": HEAD_SHA}}
    if source_project_id:
        data["source_project_id"] = source_project_id

    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            data,
            diffs_data,
            GitLabHttpResponse(
                status_code=200, body=json.dumps({"diffs": [CALCULATOR_COMPARE_DIFF]})
            ),
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    await tool._arun(project_id=7, merge_request_iid=123, baseline_sha=BASELINE_SHA)

    calls = _compare_calls(gitlab_client_mock)
    assert len(calls) == 1
    assert f"/projects/{expected_project}/repository/compare" in calls[0]
    assert f"from={BASELINE_SHA}" in calls[0]
    assert f"to={HEAD_SHA}" in calls[0]
    # straight=true keeps the compare to the two commits. The merge base default
    # would, after a rebase, return the whole merge request and mark every line.
    assert "straight=true" in calls[0]


@pytest.mark.asyncio
async def test_review_scope_block_survives_tool_response_security(
    gitlab_client_mock, metadata, mr_data_with_refs, diffs_data
):
    # This tool's responses run through PromptSecurity, whose encode_dangerous_tags
    # rewrites any tag named in DANGEROUS_TAGS. It only matches a tag with no
    # attributes, so adding review_scope there mangles the closing tag alone and the
    # block spills into the rest of the input. Nothing else covers that path.
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            mr_data_with_refs,
            diffs_data,
            GitLabHttpResponse(
                status_code=200, body=json.dumps({"diffs": [CALCULATOR_COMPARE_DIFF]})
            ),
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id=1, merge_request_iid=123, baseline_sha=BASELINE_SHA
    )

    secured = PromptSecurity.apply_security_to_tool_response(
        response, "build_review_merge_request_context"
    )

    assert '<review_scope state="delta">' in secured
    assert "</review_scope>" in secured
    assert "&lt;/review_scope&gt;" not in secured
    assert _marker_count(secured) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bogus",
    ["HEAD~1", "the previous review", "main", "aaa", "zzzz1111", ""],
)
async def test_review_scope_ignores_a_baseline_that_is_not_a_sha(
    gitlab_client_mock, metadata, mr_data_with_refs, diffs_data, bogus
):
    # security_review and fix_pipeline expose this tool in an agent toolset, so the
    # baseline can come from a model. A value that is not a commit SHA must be dropped
    # before it costs a compare and renders a block into a prompt that never
    # describes one, rather than 404ing its way to state="failed".
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(mr_data_with_refs, diffs_data, _file_content())
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id=1, merge_request_iid=123, baseline_sha=bogus)

    assert "review_scope" not in response
    assert _marker_count(response) == 0
    assert _compare_calls(gitlab_client_mock) == []


@pytest.mark.asyncio
async def test_review_scope_marks_a_line_after_added_content_starting_with_plus_plus(
    gitlab_client_mock, metadata, mr_data_with_refs
):
    # An added line whose own text starts with "++" reaches the walker as "+++...".
    # Skipping it as file metadata drops the line and leaves every later counter in
    # the hunk one short. The two diffs then disagree about new_line, the
    # intersection empties, and a genuinely new line goes unmarked while the model is
    # told to comment on marked lines only.
    mr_diffs = [
        {
            "old_path": "bump.c",
            "new_path": "bump.c",
            "generated_file": False,
            "diff": "@@ -1,2 +1,4 @@\n int main(void) {\n+++counter;\n+  doSomething();\n }",
        }
    ]
    # The compare covers the newer commit only, so its hunk starts past the "++" line
    # and carries it as context. Its counters were never skewed.
    compare = {
        "diffs": [
            {
                "new_path": "bump.c",
                "diff": "@@ -2,1 +2,2 @@\n ++counter;\n+  doSomething();\n",
            }
        ]
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=_aget_responses(
            mr_data_with_refs,
            mr_diffs,
            GitLabHttpResponse(status_code=200, body=json.dumps(compare)),
            _file_content(),
        )
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        project_id=1, merge_request_iid=123, baseline_sha=BASELINE_SHA
    )

    assert '<review_scope state="delta">' in response
    assert _marker_count(response) == 1
    assert (
        '<line type="added" old_line="" new_line="3" since_last_review="true">  doSomething();</line>'
        in response
    )
    # The "++" line is an added line at new_line 2, not metadata. Its number is what
    # a comment on it would anchor to.
    assert '<line type="added" old_line="" new_line="2">++counter;</line>' in response


def test_baseline_sha_is_hidden_from_the_model_but_open_to_the_flow(metadata):
    # security_review and fix_pipeline expose this tool in an agent toolset. The
    # baseline is meaningless to both, so the schema they show the model must be the
    # one they had before the field existed, while the code_review flow config can
    # still set it through inputs.
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    model_facing = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert "baseline_sha" not in model_facing
    # The rest of the schema is untouched, so this is not hiding the wrong field.
    assert "merge_request_iid" in model_facing

    # DeterministicStepComponent calls ainvoke with a plain dict, which validates
    # against args_schema. The injected field has to survive that.
    assert "baseline_sha" in tool.args_schema.model_json_schema()["properties"]
    parsed = tool._parse_input(
        {"project_id": 1, "merge_request_iid": 123, "baseline_sha": BASELINE_SHA}, None
    )
    assert parsed["baseline_sha"] == BASELINE_SHA

    # And the flow config's inputs: list still validates against that schema.
    validate_against_schema(
        tool.args_schema, {"project_id", "merge_request_iid", "baseline_sha"}
    )


def test_output_flags_are_hidden_from_the_model_but_open_to_the_flow(metadata):
    # Same reason as baseline_sha: security_review and fix_pipeline expose this tool in
    # an agent toolset, and neither prompt describes these flags, so the model must not
    # be able to reshape the output while a flow config still can.
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    model_facing = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert "include_instruction_format_hint" not in model_facing
    assert "include_changed_files_list" not in model_facing
    # Flags a model may legitimately choose are still its to choose.
    assert "lightweight" in model_facing
    assert "only_diffs" in model_facing

    # DeterministicStepComponent calls ainvoke with a plain dict, which validates
    # against args_schema. The injected fields have to survive that.
    parsed = tool._parse_input(
        {
            "project_id": 1,
            "merge_request_iid": 123,
            "include_instruction_format_hint": "false",
            "include_changed_files_list": "true",
        },
        None,
    )
    assert parsed["include_instruction_format_hint"] is False
    assert parsed["include_changed_files_list"] is True

    # And the flow config's inputs: list still validates against that schema.
    validate_against_schema(
        tool.args_schema,
        {
            "project_id",
            "merge_request_iid",
            "include_instruction_format_hint",
            "include_changed_files_list",
        },
    )


def test_walk_diff_lines_keeps_metadata_skipping_outside_a_hunk(metadata):
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    raw = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,1 +1,2 @@\n ctx\n+added\n"
    )

    walked = [(kind, new, text) for kind, _, new, text in tool._walk_diff_lines(raw)]

    assert walked == [
        ("chunk_header", 1, "@@ -1,1 +1,2 @@"),
        ("context", 1, "ctx"),
        ("added", 2, "added"),
    ]


def test_added_line_keys_ignores_pairs_without_a_path_or_diff(metadata):
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    pairs = [
        (None, "@@ -1,1 +1,2 @@\n+x\n"),
        ("b.py", None),
        ("c.py", "@@ -1,1 +1,2 @@\n ctx\n+y\n"),
    ]

    assert tool._added_line_keys(pairs) == {("c.py", 2)}


def test_has_collapsed_diff_distinguishes_a_dropped_patch_from_a_patchless_change(
    metadata,
):
    tool = BuildReviewMergeRequestContext(metadata=metadata)

    assert tool._has_collapsed_diff([{"new_path": "huge.rb", "diff": ""}]) is True
    # A new file is entirely added lines, so a patchless one is a dropped patch.
    assert (
        tool._has_collapsed_diff([{"new_path": "c.rb", "diff": "", "new_file": True}])
        is True
    )
    # A pure rename and a deletion have no added line to lose.
    assert (
        tool._has_collapsed_diff(
            [{"new_path": "b.md", "diff": "", "renamed_file": True}]
        )
        is False
    )
    assert (
        tool._has_collapsed_diff(
            [{"old_path": "d.rb", "new_path": None, "diff": "", "deleted_file": True}]
        )
        is False
    )
