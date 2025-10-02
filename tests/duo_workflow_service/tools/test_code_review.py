import base64
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.code_review import (
    BuildReviewMergeRequestContext,
    BuildReviewMergeRequestContextInput,
    PostDuoCodeReview,
    PostDuoCodeReviewInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    """Fixture for mock project with exclusion rules."""
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=["**/*.log", "/secrets/**", "**/node_modules/**"],
    )


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project_mock,
    }


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
            "diff": "@@ -4,7 +4,7 @@ class Calculator\n   end\n \n   def subtract(a, b)\n-    # TODO: Implement\n+    a + b\n   end\n end",
        },
        {
            "old_path": "app.log",
            "new_path": "app.log",
            "new_file": False,
            "generated_file": False,
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        },
        {
            "old_path": "generated.js",
            "new_path": "generated.js",
            "new_file": False,
            "generated_file": True,
            "diff": "@@ -1,3 +1,3 @@\n-old\n+new",
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


@pytest.mark.asyncio
async def test_post_duo_code_review(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={"message": "Comments added successfully"}
    )
    tool = PostDuoCodeReview(metadata=metadata)
    response = await tool._arun(
        project_id="123", merge_request_iid=45, review_output="<review>test</review>"
    )
    expected_response = json.dumps(
        {"status": "success", "message": "Review posted to MR !45"}
    )
    assert response == expected_response
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/code_review/add_comments",
        body=json.dumps(
            {
                "project_id": "123",
                "merge_request_iid": 45,
                "review_output": "<review>test</review>",
            }
        ),
    )


@pytest.mark.asyncio
async def test_post_duo_code_review_exception(gitlab_client_mock, metadata):
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))
    tool = PostDuoCodeReview(metadata=metadata)
    response = await tool._arun(
        project_id=123, merge_request_iid=45, review_output="<review>test</review>"
    )
    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            PostDuoCodeReviewInput(
                project_id=42,
                merge_request_iid=123,
                review_output="<review>test</review>",
            ),
            "Post Duo Code Review to merge request !123 in project 42",
        ),
    ],
)
def test_post_duo_code_review_format_display_message(input_data, expected_message):
    tool = PostDuoCodeReview(description="Post Duo Code Review")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
async def test_build_review_context_basic_success(
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True
    original_file_content = {
        "content": base64.b64encode(
            b"class Calculator\n  def subtract(a, b)\n    # TODO: Implement\n  end\nend"
        ).decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            json.dumps(mr_data),
            json.dumps(diffs_data),
            json.dumps(original_file_content),
            Exception("Custom instructions not found"),
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
    assert "calculator.rb" in response
    assert "app.log" not in response
    assert "generated.js" not in response
    assert "<original_files>" in response
    assert "</input>" in response


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
@patch("yaml.safe_load")
async def test_build_review_context_with_custom_instructions(
    mock_yaml_load,
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True
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
            json.dumps(mr_data),
            json.dumps(diffs_data),
            json.dumps(original_file_content),
            json.dumps(custom_instructions_yaml),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "custom_instructions" in response
    assert "Ruby Code Quality" in response
    assert "Apply these additional review instructions to matching files:" in response
    assert "According to custom instructions in" in response


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
async def test_build_review_context_with_url(
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True
    original_file_content = {
        "content": base64.b64encode(b"original content").decode("utf-8")
    }
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            json.dumps(mr_data),
            json.dumps(diffs_data),
            json.dumps(original_file_content),
            Exception("Custom instructions not found"),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/merge_requests/123"
    )

    assert "Implement calculator method" in response
    assert "<input>" in response


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
@patch("yaml.safe_load")
async def test_build_review_context_no_matching_custom_instructions(
    mock_yaml_load,
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
    diffs_data,
    custom_instructions_yaml,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True
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
            json.dumps(mr_data),
            json.dumps(diffs_data),
            json.dumps(original_file_content),
            json.dumps(custom_instructions_yaml),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<custom_instructions>" not in response


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
@patch("yaml.safe_load")
async def test_build_review_context_nested_vs_root_patterns(
    mock_yaml_load,
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
    custom_instructions_yaml,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True
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
            json.dumps(mr_data),
            json.dumps(nested_diffs_data),
            json.dumps(original_file_content),
            json.dumps(original_file_content),
            json.dumps(custom_instructions_yaml),
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
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Build review context for merge request https://gitlab.com/namespace/project/-/merge_requests/42",
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
    response = await tool._arun()
    result = json.loads(response)
    assert "error" in result
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_build_review_context_exception(gitlab_client_mock, metadata):
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test", merge_request_iid=123)
    result = json.loads(response)
    assert "error" in result
    assert error_message in result["error"]


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
async def test_build_review_context_no_files_content(
    mock_diff_feature_flag,
    mock_file_feature_flag,
    gitlab_client_mock,
    metadata,
    mr_data,
):
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True

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
            json.dumps(mr_data),
            json.dumps(new_files_diffs),
            Exception("Custom instructions not found"),
        ]
    )
    tool = BuildReviewMergeRequestContext(metadata=metadata)
    response = await tool._arun(project_id="test%2Fproject", merge_request_iid=123)

    assert "<original_files>" not in response
    assert "new_file.rb" in response
