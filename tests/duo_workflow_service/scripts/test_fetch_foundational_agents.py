import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest
from requests import HTTPError

from duo_workflow_service.scripts.fetch_foundational_agents import (
    FETCH_AGENT_OPERATION_NAME,
    FETCH_AGENT_QUERY,
    FETCH_FLOW_DEFINITION,
    FETCH_FLOW_OPERATION_NAME,
    fetch_agents,
    fetch_foundational_agent,
    graphql_request,
    parse_arguments,
    save_workflow_to_file,
)


class TestGraphQLRequest:
    """Test cases for graphql_request function."""

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.request")
    def test_successful_request(self, mock_request):
        """Test successful GraphQL request."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"test": "value"}}
        mock_request.return_value = mock_response

        result = graphql_request(
            "http://test.com/graphql",
            "test-token",
            "query { test }",
            FETCH_AGENT_OPERATION_NAME,
            {"var": "value"},
        )

        assert result == {"data": {"test": "value"}}
        mock_request.assert_called_once_with(
            "POST",
            "http://test.com/graphql",
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
            },
            json={
                "query": "query { test }",
                "variables": {"var": "value"},
                "operationName": FETCH_AGENT_OPERATION_NAME,
            },
            timeout=30,
        )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.request")
    def test_request_with_no_variables(self, mock_request):
        """Test GraphQL request without variables."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"test": "value"}}
        mock_request.return_value = mock_response

        graphql_request(
            "http://test.com/graphql",
            "test-token",
            "query { test }",
            FETCH_AGENT_OPERATION_NAME,
        )

        mock_request.assert_called_once_with(
            "POST",
            "http://test.com/graphql",
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
            },
            json={
                "query": "query { test }",
                "variables": None,
                "operationName": FETCH_AGENT_OPERATION_NAME,
            },
            timeout=30,
        )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.request")
    def test_request_raises_http_error(self, mock_request):
        """Test GraphQL request that raises HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("HTTP 404")
        mock_request.return_value = mock_response

        with pytest.raises(HTTPError):
            graphql_request(
                "http://test.com/graphql",
                "test-token",
                "query { test }",
                FETCH_AGENT_OPERATION_NAME,
            )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.request")
    def test_request_with_json_decode_error(self, mock_request):
        """Test GraphQL request that fails to decode JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_request.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid JSON"):
            graphql_request(
                "http://test.com/graphql",
                "test-token",
                FETCH_AGENT_OPERATION_NAME,
                "query { test }",
            )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.request")
    def test_request_with_custom_timeout(self, mock_request):
        """Test GraphQL request uses default timeout."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"test": "value"}}
        mock_request.return_value = mock_response

        graphql_request(
            "http://test.com/graphql",
            "test-token",
            FETCH_AGENT_OPERATION_NAME,
            "query { test }",
        )

        # Verify timeout parameter is set to 30 seconds
        call_args = mock_request.call_args
        assert call_args[1]["timeout"] == 30


class TestFetchFoundationalAgent:
    """Test cases for fetch_foundational_agent function."""

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.graphql_request")
    def test_successful_fetch(self, mock_graphql_request):
        """Test successful agent fetch."""
        # Mock the agent response (first call)
        agent_response = {
            "data": {
                "aiCatalogItem": {
                    "name": "Pirate Translator",
                    "latestVersion": {
                        "id": "gid://gitlab/Ai::Catalog::ItemVersion/456"
                    },
                }
            }
        }

        # Mock the flow response (second call)
        flow_response = {
            "data": {
                "aiCatalogAgentFlowConfig": "version: v1\ncomponents:\n  - name: pirate_translator"
            }
        }

        mock_graphql_request.side_effect = [agent_response, flow_response]

        name, flow_config = fetch_foundational_agent(
            "http://test.com", "token", "agent_1:123"
        )

        # Verify both GraphQL calls were made
        assert mock_graphql_request.call_count == 2

        # First call should fetch agent info
        mock_graphql_request.assert_any_call(
            "http://test.com/api/graphql",
            "token",
            FETCH_AGENT_QUERY,
            FETCH_AGENT_OPERATION_NAME,
            {"id": "gid://gitlab/Ai::Catalog::Item/123"},
        )

        # Second call should fetch flow config
        mock_graphql_request.assert_any_call(
            "http://test.com/api/graphql",
            "token",
            FETCH_FLOW_DEFINITION,
            FETCH_FLOW_OPERATION_NAME,
            {"agentVersionId": "gid://gitlab/Ai::Catalog::ItemVersion/456"},
        )

        assert name == "agent_1"
        assert flow_config == "version: v1\ncomponents:\n  - name: pirate_translator"

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.graphql_request")
    def test_fetch_with_trailing_slash_url(self, mock_graphql_request):
        """Test agent fetch with URL that has trailing slash."""
        agent_response = {
            "data": {
                "aiCatalogItem": {
                    "name": "Test Agent",
                    "latestVersion": {
                        "id": "gid://gitlab/Ai::Catalog::ItemVersion/456"
                    },
                }
            }
        }

        flow_response = {
            "data": {
                "aiCatalogAgentFlowConfig": "version: v1\ncomponents:\n  - name: test_agent"
            }
        }

        mock_graphql_request.side_effect = [agent_response, flow_response]

        fetch_foundational_agent("http://test.com/", "token", "agent_1:123")

        # Should strip trailing slash for both calls
        mock_graphql_request.assert_any_call(
            "http://test.com/api/graphql",
            "token",
            FETCH_AGENT_QUERY,
            FETCH_AGENT_OPERATION_NAME,
            {"id": "gid://gitlab/Ai::Catalog::Item/123"},
        )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.graphql_request")
    def test_fetch_with_graphql_errors(self, mock_graphql_request):
        """Test agent fetch when GraphQL returns errors."""
        mock_response = {"errors": [{"message": "Agent not found"}]}
        mock_graphql_request.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            fetch_foundational_agent("http://test.com", "token", "agent_1:123")

        assert exc_info.value.args[0] == [{"message": "Agent not found"}]

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.graphql_request")
    def test_fetch_with_missing_version_id(self, mock_graphql_request):
        """Test agent fetch when version ID is missing."""
        agent_response = {
            "data": {
                "aiCatalogItem": {
                    "name": "Test Agent",
                    "latestVersion": {"id": None},
                }
            }
        }

        mock_graphql_request.return_value = agent_response

        with pytest.raises(RuntimeError, match="Version not found"):
            fetch_foundational_agent("http://test.com", "token", "agent_1:123")


class TestSaveWorkflowToFile:
    """Test cases for save_workflow_to_file function."""

    def test_save_workflow_to_file(self):
        """Test saving workflow definition to file."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock flow config model
            mock_flow_config_model = Mock()
            mock_flow_config_model.DIRECTORY_PATH = temp_dir
            mock_flow_config_model.model_validate = Mock(return_value=None)

            filepath = save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

            expected_filepath = os.path.join(temp_dir, "test_agent.yml")
            assert filepath == expected_filepath
            assert os.path.exists(filepath)

            # Verify file contents
            with open(filepath, "r") as f:
                saved_content = f.read()

            assert saved_content == flow_def
            # Verify model validation was called
            mock_flow_config_model.model_validate.assert_called_once()

    def test_save_workflow_to_file_if_file_exists(self):
        """Test saving workflow definition when file already exists."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_agent.yml")
            with open(filepath, "w") as f:
                f.write("existing content")

            # Create a mock flow config model
            mock_flow_config_model = Mock()
            mock_flow_config_model.DIRECTORY_PATH = temp_dir

            with pytest.raises(
                FileExistsError, match=f"File {filepath} already exists"
            ):
                save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

    def test_save_workflow_to_file_validates_config(self):
        """Test that save_workflow_to_file validates the flow config."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock flow config model that raises validation error
            mock_flow_config_model = Mock()
            mock_flow_config_model.DIRECTORY_PATH = temp_dir
            mock_flow_config_model.model_validate.side_effect = ValueError(
                "Invalid flow config"
            )

            with pytest.raises(ValueError, match="Invalid flow config"):
                save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)


class TestParseArguments:
    """Test cases for parse_arguments function."""

    def test_parse_required_arguments(self):
        """Test parsing with required arguments."""
        test_args = [
            "http://test.com/graphql",
            "test-token",
            "agent_1:123,agent_2:456,agent_3:789",
            "--flow-registry-version",
            "v1",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.gitlab_url == "http://test.com/graphql"
        assert args.gitlab_token == "test-token"
        assert args.foundational_agent_ids == "agent_1:123,agent_2:456,agent_3:789"
        assert args.flow_registry_version == "v1"
        assert args.dry_run is None

    def test_parse_with_experimental_version(self):
        """Test parsing with experimental flow registry version."""
        test_args = [
            "http://test.com/graphql",
            "test-token",
            "agent_1:123,agent_2:456,agent_3:789",
            "--flow-registry-version",
            "experimental",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.gitlab_url == "http://test.com/graphql"
        assert args.gitlab_token == "test-token"
        assert args.foundational_agent_ids == "agent_1:123,agent_2:456,agent_3:789"
        assert args.flow_registry_version == "experimental"

    def test_parse_with_dry_run(self):
        """Test parsing with dry-run flag."""
        test_args = [
            "http://test.com/graphql",
            "test-token",
            "agent_1:123,agent_2:456,agent_3:789",
            "--flow-registry-version",
            "v1",
            "--dry-run",
            "true",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.gitlab_url == "http://test.com/graphql"
        assert args.gitlab_token == "test-token"
        assert args.foundational_agent_ids == "agent_1:123,agent_2:456,agent_3:789"
        assert args.flow_registry_version == "v1"
        assert args.dry_run == "true"


class TestFetchAgents:
    """Test cases for fetch_agents main function."""

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.fetch_foundational_agent"
    )
    @patch("builtins.print")
    def test_fetch_agents_dry_run_output(
        self, mock_print, mock_fetch_agent, mock_parse_args
    ):
        """Test fetch_agents with dry-run output."""
        # Mock arguments
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.foundational_agent_ids = "agent_1:123,agent_2:456"
        mock_args.flow_registry_version = "v1"
        mock_args.dry_run = True
        mock_parse_args.return_value = mock_args

        # Mock workflow definitions - now returns tuples (name, flow_config)
        mock_fetch_agent.side_effect = [
            ("agent1", "version: v1\ncomponents:\n  - name: agent1"),
            ("agent2", "version: v1\ncomponents:\n  - name: agent2"),
        ]

        fetch_agents()

        # Verify fetch_foundational_agent was called for each ID
        assert mock_fetch_agent.call_count == 2
        mock_fetch_agent.assert_any_call("http://test.com", "token", "agent_1:123")
        mock_fetch_agent.assert_any_call("http://test.com", "token", "agent_2:456")

        # Verify output to stdout - should print separator, name, and flow config for each workflow
        expected_calls = [
            unittest.mock.call("-----"),
            unittest.mock.call("agent1"),
            unittest.mock.call("version: v1\ncomponents:\n  - name: agent1"),
            unittest.mock.call("-----"),
            unittest.mock.call("agent2"),
            unittest.mock.call("version: v1\ncomponents:\n  - name: agent2"),
        ]
        mock_print.assert_has_calls(expected_calls)

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.fetch_foundational_agent"
    )
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.save_workflow_to_file"
    )
    @patch("builtins.print")
    def test_fetch_agents_file_output(
        self,
        mock_print,
        mock_save_workflow,
        mock_fetch_agent,
        mock_parse_args,
    ):
        """Test fetch_agents with file output."""
        # Mock arguments
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.foundational_agent_ids = "agent_1:123"
        mock_args.flow_registry_version = "v1"
        mock_args.dry_run = None
        mock_parse_args.return_value = mock_args

        # Mock workflow definition - now returns tuple (name, flow_config)
        mock_fetch_agent.return_value = (
            "agent_1",
            "version: v1\ncomponents:\n  - name: agent1",
        )
        mock_save_workflow.return_value = "/tmp/output/agent_1.yml"

        fetch_agents()

        # Verify save_workflow_to_file was called once
        assert mock_save_workflow.call_count == 1
        call_args = mock_save_workflow.call_args
        assert call_args[0][0] == "agent_1"
        assert call_args[0][1] == "version: v1\ncomponents:\n  - name: agent1"
        # Third argument should be the V1FlowConfig class (imported as FlowConfig from v1.flows)
        assert call_args[0][2].__name__ == "FlowConfig"

        # Verify success message printed to stderr
        mock_print.assert_called_once_with(
            "Successfully saved workflow definition(s): ['/tmp/output/agent_1.yml']",
            file=sys.stderr,
        )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    def test_fetch_agents_invalid_flow_registry_version(self, mock_parse_args):
        """Test fetch_agents raises error for invalid flow registry version."""
        # Mock arguments with invalid flow_registry_version
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.foundational_agent_ids = "agent_1:123"
        mock_args.flow_registry_version = "invalid"
        mock_parse_args.return_value = mock_args

        with pytest.raises(
            ValueError, match="Flow Registry version 'invalid' is not supported"
        ):
            fetch_agents()
