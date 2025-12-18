"""Shared fixtures for tool tests."""

import pytest

from contract import contract_pb2
from duo_workflow_service.security.tool_output_security import ToolTrustLevel


@pytest.fixture(autouse=True)
def mock_tool_trust_level(monkeypatch):
    """Mock trust_level to TRUSTED_INTERNAL for all tool tests.

    This bypasses security wrapping so tool logic tests can verify raw output without wrapper interference. Security
    wrapping behavior is tested separately in tests/duo_workflow_service/security/.
    """
    original_arun = None

    # Import here to avoid circular imports
    from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

    original_arun = DuoBaseTool._arun

    async def _arun_without_wrapping(self, *args, **kwargs):
        """Execute tool without security wrapping."""
        from duo_workflow_service.tools.tool_output_manager import (
            truncate_tool_response,
        )

        tool_result = await self._execute(*args, **kwargs)
        return truncate_tool_response(
            tool_response=tool_result,
            tool_name=self.name,
            truncation_config=self.truncation_config,
        )

    monkeypatch.setattr(DuoBaseTool, "_arun", _arun_without_wrapping)

    yield

    # Restore original (monkeypatch handles this automatically)


@pytest.fixture(name="mock_action_response")
def mock_action_response_fixture():
    """Create a mock ActionResponse with proper structure."""
    mock_action_response = contract_pb2.ActionResponse()
    mock_action_response.requestID = "test-request-id"
    mock_action_response.plainTextResponse.response = "test response"
    mock_action_response.plainTextResponse.error = ""  # No error
    return mock_action_response


@pytest.fixture(name="mock_client_event")
def mock_client_event_fixture(mock_action_response):
    """Create a mock ClientEvent containing the ActionResponse."""
    mock_client_event = contract_pb2.ClientEvent()
    mock_client_event.actionResponse.CopyFrom(mock_action_response)
    return mock_client_event


@pytest.fixture(name="mock_success_action_response")
def mock_success_action_response_fixture():
    """Create a mock ActionResponse for successful operations."""
    mock_action_response = contract_pb2.ActionResponse()
    mock_action_response.requestID = "test-request-id"
    mock_action_response.plainTextResponse.response = "done"
    mock_action_response.plainTextResponse.error = ""  # No error
    return mock_action_response


@pytest.fixture(name="mock_success_client_event")
def mock_success_client_event_fixture(mock_success_action_response):
    """Create a mock ClientEvent for successful operations."""
    mock_client_event = contract_pb2.ClientEvent()
    mock_client_event.actionResponse.CopyFrom(mock_success_action_response)
    return mock_client_event


def create_mock_client_event_with_response(response_text: str):
    """Helper function to create a mock ClientEvent with custom response text."""
    mock_action_response = contract_pb2.ActionResponse()
    mock_action_response.requestID = "test-request-id"
    mock_action_response.plainTextResponse.response = response_text
    mock_action_response.plainTextResponse.error = ""  # No error

    mock_client_event = contract_pb2.ClientEvent()
    mock_client_event.actionResponse.CopyFrom(mock_action_response)
    return mock_client_event
