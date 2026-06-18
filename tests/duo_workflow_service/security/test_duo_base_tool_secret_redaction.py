# pylint: disable=file-naming-for-tests
"""These tests live in the security test directory intentionally: the tool test
directory's conftest replaces ``_arun`` with a stripped version that skips
security processing, which would make these tests meaningless.  The security
directory has no such fixture so the real ``_arun`` pipeline is exercised.
"""

from typing import Any
from unittest.mock import patch

import pytest

import duo_workflow_service.tools.duo_base_tool as duo_base_tool_module
from duo_workflow_service.security.secret_redaction import REDACTED_PLACEHOLDER
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig


class SecretReturningTool(DuoBaseTool):
    """Minimal tool whose ``_execute`` returns a caller-controlled value."""

    name: str = "secret_returning_tool"
    description: str = "Tool that returns a pre-set response for testing"
    _response: Any = None

    def set_response(self, response: Any) -> None:
        self._response = response

    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


@pytest.mark.asyncio
async def test_arun_redacts_gitlab_token_in_string_response():
    """_arun must redact a GitLab PAT embedded in a plain string response."""
    token = "glpat-AAAAABBBBCCCCDDDDEEEE"
    tool = SecretReturningTool(metadata={})
    tool.set_response(f"see {token} for access")

    result = await tool._arun()

    assert token not in result
    assert REDACTED_PLACEHOLDER in result


@pytest.mark.asyncio
async def test_arun_does_not_modify_normal_string_response():
    """_arun must leave non-secret string responses untouched."""
    tool = SecretReturningTool(metadata={})
    tool.set_response("the project has 3 issues")

    result = await tool._arun()

    assert result == "the project has 3 issues"


@pytest.mark.asyncio
async def test_arun_redacts_secret_in_dict_response():
    """_arun must redact secrets nested inside a dict response."""
    token = "glpat-AAAAABBBBCCCCDDDDEEEE"
    tool = SecretReturningTool(metadata={})
    tool.set_response({"body": f"token: {token}", "id": 1})

    result = await tool._arun()

    assert isinstance(result, dict)
    assert token not in result["body"]
    assert REDACTED_PLACEHOLDER in result["body"]
    assert result["id"] == 1


@pytest.mark.asyncio
async def test_arun_redacts_secret_in_list_response():
    """_arun must redact secrets inside list entries."""
    token = "glpat-AAAAABBBBCCCCDDDDEEEE"
    tool = SecretReturningTool(metadata={})
    tool.set_response(["normal", f"token: {token}"])

    result = await tool._arun()

    assert isinstance(result, list)
    assert result[0] == "normal"
    assert token not in result[1]
    assert REDACTED_PLACEHOLDER in result[1]


@pytest.mark.asyncio
async def test_arun_redacts_aws_key_in_response():
    """_arun must redact AWS access keys."""
    tool = SecretReturningTool(metadata={})
    tool.set_response("key=AKIAIOSFODNN7EXAMPLE")

    result = await tool._arun()

    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert REDACTED_PLACEHOLDER in result


@pytest.mark.asyncio
async def test_arun_passes_through_scalar_response():
    """_arun must return non-string scalars unchanged (e.g. integers)."""
    tool = SecretReturningTool(metadata={})
    tool.set_response(42)

    result = await tool._arun()

    assert result == 42


# ---------------------------------------------------------------------------
# Tests for redact → truncate ordering in _arun
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_redact_truncate_ordering():
    """_arun must call redact_secrets before truncate_tool_response.

    The ordering redact → truncate ensures that full secrets are caught before truncation can split them.
    """
    call_order = []

    original_redact = duo_base_tool_module.redact_secrets
    original_truncate = duo_base_tool_module.truncate_tool_response

    def mock_redact(response, tool_name):
        call_order.append("redact")
        return original_redact(response, tool_name)

    def mock_truncate(tool_response, tool_name, truncation_config):
        call_order.append("truncate")
        return original_truncate(tool_response, tool_name, truncation_config)

    tool = SecretReturningTool(metadata={})
    tool.set_response("some tool output")

    with (
        patch.object(duo_base_tool_module, "redact_secrets", side_effect=mock_redact),
        patch.object(
            duo_base_tool_module, "truncate_tool_response", side_effect=mock_truncate
        ),
    ):
        await tool._arun()

    assert call_order == ["redact", "truncate"], (
        f"Expected ['redact', 'truncate'] but got {call_order}. "
        "The _arun method must redact before truncation (to catch full secrets)."
    )


@pytest.mark.asyncio
async def test_arun_secret_at_truncation_boundary_is_redacted():
    """A secret straddling the truncation boundary must still be redacted.

    The token is placed so that its prefix (``glpat-``) falls within the kept
    region and its suffix is cut off by truncation.  The correct redact → truncate
    order catches the full token in the redact pass, before truncation can split it.
    """
    truncated_size = 100 * 1024  # 100 KiB kept after truncation
    max_bytes = 200 * 1024  # 200 KiB threshold that triggers truncation

    # A synthetic GitLab PAT: "glpat-" + 20 alphanumeric chars (26 bytes total).
    # This matches the GitLab token pattern and will be redacted by redact_secrets.
    # Constructed dynamically to avoid triggering secret push protection on the literal.
    token = "glpat-" + "abcdefghijklmnopqrst"

    # Place the token so it straddles the truncation boundary:
    # the first 10 bytes of the token fall inside the kept region,
    # the remaining 16 bytes are cut off by truncation.
    # The suffix must begin with a non-word character so that the GitLab token
    # regex (which has a ``(?!\w)`` negative lookahead) can match the full token
    # in the redact pass before truncation splits it.
    split_point = 10
    prefix_padding = "x" * (truncated_size - split_point)
    suffix_padding = " " + "y" * (
        max_bytes - truncated_size
    )  # space ensures regex matches

    response = prefix_padding + token + suffix_padding

    tool = SecretReturningTool(
        metadata={},
        truncation_config=TruncationConfig(
            max_bytes=max_bytes, truncated_size=truncated_size
        ),
    )
    tool.set_response(response)

    result = await tool._arun()

    assert isinstance(result, str)
    assert token not in result, (
        "The full token must not appear in the output; it should have been redacted "
        "by the redact pass before truncation could split it."
    )
    assert REDACTED_PLACEHOLDER in result


@pytest.mark.asyncio
async def test_arun_redact_receives_original_response():
    """The redact_secrets call must receive the original (pre-truncation) response.

    This ensures full secrets are caught before truncation can split them.
    """
    original_redact = duo_base_tool_module.redact_secrets
    redact_inputs = []

    def mock_redact(response, tool_name):
        redact_inputs.append(response)
        return original_redact(response, tool_name)

    # Build a large response that will trigger truncation
    large_response = "a" * (300 * 1024)  # 300 KiB

    tool = SecretReturningTool(
        metadata={},
        truncation_config=TruncationConfig(
            max_bytes=200 * 1024, truncated_size=100 * 1024
        ),
    )
    tool.set_response(large_response)

    with patch.object(duo_base_tool_module, "redact_secrets", side_effect=mock_redact):
        await tool._arun()

    assert len(redact_inputs) == 1, (
        f"Expected redact_secrets to be called exactly once, got {len(redact_inputs)}"
    )
    assert redact_inputs[0] == large_response, (
        "The redact_secrets call must receive the original (pre-truncation) response "
        "so that full secrets are caught before truncation can split them."
    )
