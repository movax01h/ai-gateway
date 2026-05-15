# pylint: disable=import-outside-toplevel
"""Tests for the secret_redaction module."""

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from duo_workflow_service.security.secret_redaction import (
    REDACTED_PLACEHOLDER,
    redact_secrets,
    redact_secrets_for_ui,
)


class TestRedactSecretsStrings:
    """Tests for string inputs to redact_secrets."""

    @pytest.fixture
    def jwt_token(self):
        return (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWUsImlhdCI6MTUxNjIzOTAyMn0"
            ".KMUFsIDTnFmyG3nMiGM6H9FNFUROf3wh7SmqJp-QV30"
        )

    def test_plain_text_unchanged(self):
        """Normal text with no secrets must pass through unmodified."""
        assert (
            redact_secrets("the project has 3 issues", tool_name="test")
            == "the project has 3 issues"
        )

    def test_plain_text_with_numbers_unchanged(self):
        assert redact_secrets("processed 5 records successfully", tool_name="test") == (
            "processed 5 records successfully"
        )

    def test_gitlab_personal_access_token_redacted(self):
        # GitLab PAT: "glpat-" followed by 20-50 alphanumeric/dash/underscore chars
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"
        text = f"token: {token}"
        result = redact_secrets(text, tool_name="test")
        assert token not in result
        assert REDACTED_PLACEHOLDER in result

    def test_gitlab_deploy_token_redacted(self):
        # Deploy token: "gldt-" followed by 20+ chars
        token = "gldt-AAAAABBBBCCCCDDDDEEEE"
        text = f"{token} rest of text"
        result = redact_secrets(text, tool_name="test")
        assert token not in result
        assert REDACTED_PLACEHOLDER in result

    def test_gitlab_runner_token_redacted(self):
        text = "runner_token=GR1348941ABCDEFGHIJKLMNOPQRSTUVXa"
        result = redact_secrets(text, tool_name="test")
        assert "GR1348941ABCDEFGHIJKLMNOPQRSTUVXa" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_jwt_token_redacted(self, jwt_token):
        text = f"Authorization: Bearer {jwt_token}"
        result = redact_secrets(text, tool_name="test")
        assert jwt_token not in result
        assert REDACTED_PLACEHOLDER in result

    def test_jwt_job_token_redacted(self, jwt_token):
        text = f"glcbt-{jwt_token}"
        result = redact_secrets(text, tool_name="test")
        assert jwt_token not in result
        assert REDACTED_PLACEHOLDER in result

    def test_aws_access_key_redacted(self):
        text = "api_key: AKIAIOSFODNN7EXAMPLE"
        result = redact_secrets(text, tool_name="test")
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_slack_token_redacted(self):
        text = "slack_token=xoxb-123456789-abcdefghijklmnop"
        result = redact_secrets(text, tool_name="test")
        assert "xoxb-123456789-abcdefghijklmnop" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_private_key_header_redacted(self):
        text = "-----BEGIN RSA PRIVATE KEY-----"
        result = redact_secrets(text, tool_name="test")
        assert "BEGIN RSA PRIVATE KEY" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_stripe_secret_key_redacted(self):
        text = "stripe_key=sk_live_abcdefghijklmnopqrstuvwx"
        result = redact_secrets(text, tool_name="test")
        assert "sk_live_abcdefghijklmnopqrstuvwx" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_basic_auth_url_redacted(self):
        text = "http://user:p4ssw0rd@example.com/api"
        result = redact_secrets(text, tool_name="test")
        assert "p4ssw0rd" not in result
        assert REDACTED_PLACEHOLDER in result

    def test_git_clone_url_with_oauth_token_redacted(self):
        """OAuth token embedded in a git clone URL must be redacted.

        Example: a repository URL of the form
        https://oauth:<token>@gitlab.com/group/project.git that can appear in
        tool output when git operations expose the remote URL.
        """
        token = "a24afa37cd88bcf3241cda777a2acd36b4abe4672859814eac0f1d515f78g14c"
        text = f"https://oauth:{token}@gitlab.com/.git"
        result = redact_secrets(text, tool_name="test")
        assert token not in result
        assert REDACTED_PLACEHOLDER in result

    def test_empty_string_unchanged(self):
        assert redact_secrets("", tool_name="test") == ""

    def test_non_secret_bearer_token_word_unchanged(self):
        """'Bearer token authentication' should not be modified (no actual token)."""
        text = "Bearer token authentication"
        assert redact_secrets(text, tool_name="test") == text


class TestRedactSecretsDictAndList:
    """Tests for dict and list inputs to redact_secrets."""

    def test_dict_with_secret_value_redacted(self):
        data = {"message": "token: glpat-xxxxxxxxxxxxxxxxxxxx", "count": 5}
        result = redact_secrets(data, tool_name="test")
        assert isinstance(result, dict)
        assert "glpat-xxxxxxxxxxxxxxxxxxxx" not in result["message"]
        assert REDACTED_PLACEHOLDER in result["message"]
        # Non-string values are untouched
        assert result["count"] == 5

    def test_dict_with_no_secrets_unchanged(self):
        data = {"message": "everything is fine", "count": 3}
        result = redact_secrets(data, tool_name="test")
        assert result == data

    def test_list_with_secret_entry_redacted(self):
        data = ["normal text", "token: glpat-xxxxxxxxxxxxxxxxxxxx", "more text"]
        result = redact_secrets(data, tool_name="test")
        assert isinstance(result, list)
        assert result[0] == "normal text"
        assert "glpat-xxxxxxxxxxxxxxxxxxxx" not in result[1]
        assert REDACTED_PLACEHOLDER in result[1]
        assert result[2] == "more text"

    def test_nested_structure_redacted(self):
        data = {
            "notes": [
                {"body": "see glpat-xxxxxxxxxxxxxxxxxxxx for access"},
                {"body": "no secrets here"},
            ]
        }
        result = redact_secrets(data, tool_name="test")
        assert "glpat-xxxxxxxxxxxxxxxxxxxx" not in result["notes"][0]["body"]
        assert REDACTED_PLACEHOLDER in result["notes"][0]["body"]
        assert result["notes"][1]["body"] == "no secrets here"

    def test_scalar_types_returned_unchanged(self):
        assert redact_secrets(42, tool_name="test") == 42
        assert redact_secrets(3.14, tool_name="test") == 3.14
        assert redact_secrets(True, tool_name="test") is True
        assert redact_secrets(None, tool_name="test") is None

    def test_empty_dict_unchanged(self):
        assert redact_secrets({}, tool_name="test") == {}

    def test_empty_list_unchanged(self):
        assert redact_secrets([], tool_name="test") == []


class TestFalsePositives:
    """Verify that legitimate tool output values are never redacted.

    Entropy-based detectors (``Base64HighEntropyString``, ``HexHighEntropyString``) are
    intentionally excluded from the detector list because they produce an unacceptable
    rate of false positives on ordinary GitLab API payloads (JSON keys, git SHAs, UUIDs,
    base64 file content, etc.).  These tests act as a regression guard to prevent those
    detectors from being re-added.
    """

    def test_gitlab_issue_json_keys_unchanged(self):
        """JSON keys from a GitLab issue response must not be redacted.

        Reproduces the bug where Base64HighEntropyString matched short common words like 'id', 'status', 'count' inside
        a serialised JSON string, producing output like '\"[REDACTED]\": null' and '\"i[REDACTED]\": 1'.
        """
        import json

        issue = {
            "id": None,
            "iid": 1,
            "status": "open",
            "confidential": False,
            "discussion_locked": None,
            "task_completion_status": {"count": 0, "completed_count": 0},
            "blocking_issues_count": 0,
            "task_status": "0 of 0 checklist items completed",
            "moved_to_id": None,
            "duplicated_to": "open",
            "duplicated_iid": 1,
            "health_status": None,
        }
        # Tools often return JSON as a string; make sure none of the keys are touched.
        text = json.dumps(issue)
        assert redact_secrets(text, tool_name="test") == text

    def test_git_commit_sha_unchanged(self):
        """40-char git SHAs appear constantly in tool output and must not be redacted."""
        text = "commit 511c3bcec4034598b3f0fb234b80cdc81d06f5c\nAuthor: Dev"
        assert redact_secrets(text, tool_name="test") == text

    def test_short_git_sha_unchanged(self):
        """Abbreviated SHAs (7–12 chars) used in log summaries must not be redacted."""
        text = "de7d233b0 Merge branch 'main'"
        assert redact_secrets(text, tool_name="test") == text

    def test_sha256_digest_unchanged(self):
        """SHA-256 digests appear in checksums / integrity fields and must not be redacted."""
        digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        text = f"sha256:{digest}"
        assert redact_secrets(text, tool_name="test") == text

    def test_md5_hash_unchanged(self):
        """MD5 hashes in file metadata or API responses must not be redacted."""
        text = "Content-MD5: d41d8cd98f00b204e9800998ecf8427e"
        assert redact_secrets(text, tool_name="test") == text

    def test_hex_color_codes_unchanged(self):
        """CSS / design-system color values must not be redacted."""
        text = "background-color: #1a2b3c; color: #ff6600;"
        assert redact_secrets(text, tool_name="test") == text

    def test_base64_encoded_file_content_unchanged(self):
        """Base64 file blobs returned by GitLab API content endpoints must not be redacted."""
        b64 = "SGVsbG8sIFdvcmxkIQo="
        text = f'{{"content": "{b64}", "encoding": "base64"}}'
        assert redact_secrets(text, tool_name="test") == text

    def test_longer_base64_blob_unchanged(self):
        """Longer base64 content (e.g. small file returned by read_file tool) must not be redacted."""
        b64 = "dGhpcyBpcyBub3QgYSBzZWNyZXQsIGp1c3QgYSBub3JtYWwgZmlsZSBib2R5Cg=="
        text = f"file contents (base64): {b64}"
        assert redact_secrets(text, tool_name="test") == text

    def test_uuid_unchanged(self):
        """UUIDs are high-entropy hex strings used as identifiers and must not be redacted."""
        text = "request_id=550e8400-e29b-41d4-a716-446655440000"
        assert redact_secrets(text, tool_name="test") == text

    def test_docker_image_digest_unchanged(self):
        """Docker image digests (sha256:...) in CI/CD output must not be redacted."""
        digest = (
            "sha256:a948904f2f0f479b8f936065f3aa7b9a516558c7d948065ae59acc2a5e0e9fbc"
        )
        text = f"Digest: {digest}"
        assert redact_secrets(text, tool_name="test") == text

    def test_diff_patch_line_unchanged(self):
        """Patch index lines in git diff output contain hex object ids and must not be redacted."""
        text = "index 83db48f..f735c75 100644"
        assert redact_secrets(text, tool_name="test") == text


class TestRedactSecretsForUi:
    """Tests for redact_secrets_for_ui – the UI / UiChatLog path.

    This function adds entropy-based detectors (Base64 limit=4.0, Hex limit=3.7)
    on top of the structured detectors.  Tests here focus on the *differences*
    from ``redact_secrets``: what extra secrets the entropy detectors catch, and
    what benign high-entropy values must still pass through unredacted.
    """

    # --- Extra coverage from entropy detectors ---

    def test_azure_storage_key_redacted(self):
        """High-entropy Base64 string matching an Azure storage key format must be redacted."""
        key = "lJzRc1YdsnuBA9nfdAJe2dgA+jfJpe7S8BpSCzM5s0ME="
        text = f'"AccountKey={key}"'
        result = redact_secrets_for_ui(text, tool_name="test")
        assert key not in result
        assert REDACTED_PLACEHOLDER in result

    def test_generic_high_entropy_base64_blob_redacted(self):
        """A quoted high-entropy base64 string (e.g. a generic API secret) must be redacted."""
        # entropy > 4.0 for Base64 charset
        blob = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo="
        text = f'secret: "{blob}"'
        result = redact_secrets_for_ui(text, tool_name="test")
        assert blob not in result
        assert REDACTED_PLACEHOLDER in result

    # --- Structured secrets still caught ---

    def test_gitlab_token_still_redacted(self):
        """Structured secrets must still be redacted by redact_secrets_for_ui."""
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"
        result = redact_secrets_for_ui(f"token: {token}", tool_name="test")
        assert token not in result
        assert REDACTED_PLACEHOLDER in result

    # --- Entropy false-positive guard: values that must NOT be redacted ---

    def test_gitlab_issue_json_keys_unchanged(self):
        """GitLab API JSON field names must not be redacted even by the UI path."""
        import json

        issue = {
            "id": None,
            "iid": 1,
            "status": "open",
            "confidential": False,
            "discussion_locked": None,
            "task_completion_status": {"count": 0, "completed_count": 0},
            "blocking_issues_count": 0,
            "task_status": "0 of 0 checklist items completed",
            "moved_to_id": None,
            "blocking_discussions_resolved": True,
        }
        text = json.dumps(issue)
        assert redact_secrets_for_ui(text, tool_name="test") == text

    def test_git_sha_unchanged(self):
        """40-char git SHAs (hex entropy ≤ 3.61) must not be redacted."""
        text = '"511c3bcec4034598b3f0fb234b80cdc81d06f5c"'
        assert redact_secrets_for_ui(text, tool_name="test") == text

    def test_sha256_digest_unchanged(self):
        """SHA-256 digests (hex entropy ≤ 3.67) must not be redacted."""
        digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        text = f'"sha256:{digest}"'
        assert redact_secrets_for_ui(text, tool_name="test") == text

    def test_uuid_unchanged(self):
        """UUIDs must not be redacted by the UI path."""
        text = '"550e8400-e29b-41d4-a716-446655440000"'
        assert redact_secrets_for_ui(text, tool_name="test") == text

    # --- Verify the two functions handle the same scalars / structures ---

    def test_scalar_unchanged(self):
        assert redact_secrets_for_ui(42, tool_name="test") == 42
        assert redact_secrets_for_ui(None, tool_name="test") is None

    def test_dict_values_redacted(self):
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"
        result = redact_secrets_for_ui({"body": f"see {token}"}, tool_name="test")
        assert token not in result["body"]
        assert REDACTED_PLACEHOLDER in result["body"]

    # --- Known false positives: values that ARE redacted on the UI path
    #     but must NOT be redacted on the LLM path (redact_secrets).
    #
    #     These tests document an intentional design trade-off:
    #
    #     The UI path uses entropy detectors (Base64 ≥ 4.0, Hex ≥ 3.7) to catch
    #     high-entropy blobs that have no structured prefix (e.g. Azure storage
    #     keys, generic API tokens). The same detectors have a Shannon-entropy
    #     overlap with certain legitimate engineering artefacts – specifically
    #     40-char commit SHAs quoted as JSON string values (entropy 3.86) and
    #     npm/yarn integrity hashes (entropy > 4.0).
    #
    #     Applying entropy detectors to the LLM context path (redact_secrets)
    #     would silently corrupt the data the LLM reasons over:
    #       - A workflow that reads a source file and updates a pinned commit SHA
    #         would see [REDACTED] instead of the SHA and produce a broken patch.
    #       - A workflow that audits package-lock.json integrity hashes would be
    #         unable to read or reproduce them, defeating the security audit.
    #
    #     The UI path accepts this trade-off because humans reading a chat log do
    #     not need to act on the literal value – they only need to know a tool ran.
    # ---

    def test_hardcoded_commit_sha_in_source_file_is_redacted_for_ui(self):
        """A commit SHA assigned as a quoted string constant in source code is redacted on the UI path by
        HexHighEntropyString (entropy 3.86 > threshold 3.7).

        This is a known false positive.  The same input is intentionally left
        unchanged by redact_secrets (the LLM path) so that a workflow reading
        this file can still extract and act on the SHA value.

        Example source line:
            DEPLOY_SHA = "6104942438c14ec7bd21c6cd5bd995272b3faff6"
        """
        sha = "6104942438c14ec7bd21c6cd5bd995272b3faff6"
        source_line = f'DEPLOY_SHA = "{sha}"'

        # UI path: entropy detector fires → SHA is redacted
        ui_result = redact_secrets_for_ui(source_line, tool_name="test")
        assert sha not in ui_result
        assert REDACTED_PLACEHOLDER in ui_result

        # LLM path: structured detectors only → SHA is preserved
        llm_result = redact_secrets(source_line, tool_name="test")
        assert llm_result == source_line

    def test_npm_integrity_hash_in_package_lock_is_redacted_for_ui(self):
        """An npm/yarn integrity hash (sha512, base64-encoded) is redacted on the UI path by Base64HighEntropyString
        (entropy > 4.0).

        This is a known false positive.  The same input is intentionally left
        unchanged by redact_secrets (the LLM path) so that a workflow auditing
        dependency integrity or reproducing a lockfile can read the values.

        Example package-lock.json field:
            "integrity": "sha512-oZhbN9cA..."
        """
        integrity = (
            "sha512-v2kDEe57lecTulaDIuNTPy3Ry4gLGJ6Z1O3vE1krgXZNrsQ"
            "+LFTGHVxVjcXPs17LhbZkGAtkkfnF/8FGMN3xw=="
        )
        package_lock_entry = f'"integrity": "{integrity}"'

        # UI path: entropy detector fires → hash is redacted
        ui_result = redact_secrets_for_ui(package_lock_entry, tool_name="test")
        assert integrity not in ui_result
        assert REDACTED_PLACEHOLDER in ui_result

        # LLM path: structured detectors only → hash is preserved
        llm_result = redact_secrets(package_lock_entry, tool_name="test")
        assert llm_result == package_lock_entry


class TestObjectsWithContentAttribute:
    """Tests for duck-typed handling of objects that carry a ``content`` attribute.

    The redactor must work with any object that exposes ``content`` -- not just
    ``ToolMessage`` -- so that callers never need to unwrap the object before
    passing it in.  Pydantic models are copied via ``model_copy``; plain objects
    via ``copy.copy`` + attribute assignment.
    """

    # --- ToolMessage (Pydantic model) ---

    def test_tool_message_with_clean_content_returned_unchanged(self):
        """A ToolMessage whose content contains no secrets must be returned as-is."""
        tm = ToolMessage(content="no secrets here", tool_call_id="x", name="t")
        result = redact_secrets(tm, tool_name="test")
        assert result is tm

    def test_tool_message_secret_in_content_is_redacted(self):
        """A secret inside ToolMessage.content must be replaced with REDACTED_PLACEHOLDER."""
        tm = ToolMessage(
            content="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            tool_call_id="x",
            name="t",
        )
        result = redact_secrets(tm, tool_name="test")

        assert isinstance(result, ToolMessage)
        assert result is not tm
        assert REDACTED_PLACEHOLDER in result.content
        # Non-content fields must be preserved
        assert result.tool_call_id == tm.tool_call_id
        assert result.name == tm.name

    def test_tool_message_clean_content_unchanged(self):
        """When content has no secrets the ToolMessage content must be unchanged."""
        tm = ToolMessage(content="plain text", tool_call_id="x", name="t")
        result = redact_secrets(tm, tool_name="test")
        assert result.content == "plain text"

    # --- Plain object (non-Pydantic, duck-typed) ---

    def test_plain_object_with_content_attr_is_redacted(self):
        """A plain object with a ``content`` attribute must be handled via copy.copy."""

        class SimpleResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        obj = SimpleResponse(
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        result = redact_secrets(obj, tool_name="test")

        assert result is not obj
        assert REDACTED_PLACEHOLDER in result.content
        assert obj.content != result.content  # original untouched

    def test_plain_object_clean_content_unchanged(self):
        """A plain object whose content has no secrets must have content unchanged."""

        class SimpleResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        obj = SimpleResponse("nothing sensitive")
        result = redact_secrets(obj, tool_name="test")
        assert result.content == "nothing sensitive"

    # --- Interaction with existing types ---

    def test_tool_message_inside_list_is_redacted(self):
        """A ToolMessage nested inside a list must be redacted."""
        tm = ToolMessage(
            content="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            tool_call_id="x",
            name="t",
        )
        result = redact_secrets([tm, "plain"], tool_name="test")

        assert isinstance(result[0], ToolMessage)
        assert REDACTED_PLACEHOLDER in result[0].content
        assert result[1] == "plain"

    def test_tool_message_inside_dict_value_is_redacted(self):
        """A ToolMessage stored as a dict value must be redacted."""
        tm = ToolMessage(
            content="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            tool_call_id="x",
            name="t",
        )
        result = redact_secrets({"msg": tm}, tool_name="test")

        assert isinstance(result["msg"], ToolMessage)
        assert REDACTED_PLACEHOLDER in result["msg"].content


class TestDataclassInstances:
    """Tests for duck-typed handling of dataclass instances such as ``langgraph.types.Command``.

    ``Command`` carries tool results inside ``update`` as a nested dict of lists of
    ``ToolMessage`` objects.  The redactor must recurse through the dataclass fields
    and return a new instance via ``dataclasses.replace`` when any field changed.
    """

    JWT = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )

    def _tool_message(self, content: str) -> ToolMessage:
        return ToolMessage(content=content, tool_call_id="x", name="t")

    def test_command_with_clean_update_content_unchanged(self):
        """A Command whose nested ToolMessages contain no secrets must have content unchanged."""
        cmd = Command(
            update={
                "conversation_history": {
                    "planner": [self._tool_message("no secrets here")]
                }
            }
        )
        result = redact_secrets(cmd, tool_name="test")
        assert (
            result.update["conversation_history"]["planner"][0].content
            == "no secrets here"
        )

    def test_command_with_secret_in_tool_message_is_redacted(self):
        """A secret inside a ToolMessage nested in Command.update must be redacted."""
        cmd = Command(
            update={
                "conversation_history": {
                    "planner": [self._tool_message(f"Authorization: Bearer {self.JWT}")]
                }
            }
        )
        result = redact_secrets(cmd, tool_name="test")

        assert result is not cmd
        assert isinstance(result, Command)
        redacted_tm = result.update["conversation_history"]["planner"][0]
        assert isinstance(redacted_tm, ToolMessage)
        assert self.JWT not in redacted_tm.content
        assert REDACTED_PLACEHOLDER in redacted_tm.content

    def test_command_non_update_fields_preserved(self):
        """Fields other than ``update`` must be preserved unchanged after redaction."""
        cmd = Command(
            update={
                "conversation_history": {
                    "planner": [self._tool_message(f"Bearer {self.JWT}")]
                }
            },
            goto="some_node",
        )
        result = redact_secrets(cmd, tool_name="test")

        assert result.goto == cmd.goto
        assert result.graph == cmd.graph
        assert result.resume == cmd.resume

    def test_command_with_multiple_tool_messages_all_redacted(self):
        """All ToolMessages in a Command update list must be redacted."""
        cmd = Command(
            update={
                "conversation_history": {
                    "planner": [
                        self._tool_message(f"token: {self.JWT}"),
                        self._tool_message("clean message"),
                        self._tool_message(f"also secret: {self.JWT}"),
                    ]
                }
            }
        )
        result = redact_secrets(cmd, tool_name="test")

        messages = result.update["conversation_history"]["planner"]
        assert REDACTED_PLACEHOLDER in messages[0].content
        assert messages[1].content == "clean message"
        assert REDACTED_PLACEHOLDER in messages[2].content
