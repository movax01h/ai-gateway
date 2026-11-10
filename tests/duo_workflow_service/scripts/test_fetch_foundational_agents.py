import os
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError
from requests import HTTPError

from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    DEFAULT_FLOW_VERSION,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig as V1FlowConfig,
)
from duo_workflow_service.scripts import fetch_foundational_agents as script
from duo_workflow_service.scripts.fetch_foundational_agents import (
    DEFAULT_AGENTS_FILE,
    FETCH_AGENT_OPERATION_NAME,
    FETCH_AGENT_QUERY,
    FETCH_FLOW_DEFINITION,
    FETCH_FLOW_OPERATION_NAME,
    fetch_agents,
    fetch_foundational_agent,
    graphql_request,
    load_agents,
    parse_agent_list,
    parse_arguments,
    save_workflow_to_file,
)
from lib.feature_roots import default_features_dir


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
            "http://test.com", "token", "agent_1", 123
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

        fetch_foundational_agent("http://test.com/", "token", "agent_1", 123)

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
            fetch_foundational_agent("http://test.com", "token", "agent_1", 123)

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
            fetch_foundational_agent("http://test.com", "token", "agent_1", 123)


_V1_FLOW_DEF = (
    "version: v1\nenvironment: chat\ncomponents: []\nrouters: []\n"
    'flow:\n  entry_point: "x"\n'
)


def _model(temp_dir: str) -> Mock:
    """A non-v1 flow config model (experimental layout) rooted at *temp_dir*."""
    model = Mock()
    model.DIRECTORY_PATH = temp_dir
    model.model_validate = Mock(return_value=None)
    return model


class TestSaveWorkflowToFile:
    """Test cases for save_workflow_to_file function."""

    def test_save_workflow_to_file(self):
        """Test saving workflow definition to versioned subdirectory."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_flow_config_model = _model(temp_dir)

            filepath = save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

            # Legacy layout: {DIRECTORY_PATH}/{agent_id}/{DEFAULT_FLOW_VERSION}.yml
            expected_filepath = os.path.join(
                temp_dir, "test_agent", f"{DEFAULT_FLOW_VERSION}.yml"
            )
            assert filepath == expected_filepath
            assert os.path.exists(filepath)

            # Verify file contents
            with open(filepath, "r") as f:
                saved_content = f.read()

            assert saved_content == flow_def
            # Verify model validation was called
            mock_flow_config_model.model_validate.assert_called_once()

    def test_save_workflow_to_file_creates_subdirectory(self):
        """Test that save_workflow_to_file creates the versioned subdirectory."""
        agent_id = "new_agent"
        flow_def = "version: v1\ncomponents:\n  - name: new_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_flow_config_model = _model(temp_dir)

            agent_dir = os.path.join(temp_dir, agent_id)
            assert not os.path.exists(agent_dir)

            save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

            assert os.path.isdir(agent_dir)

    def test_save_workflow_to_file_if_file_exists(self):
        """Test saving workflow definition when versioned file already exists."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Pre-create the versioned subdirectory and file
            agent_dir = os.path.join(temp_dir, agent_id)
            os.makedirs(agent_dir)
            filepath = os.path.join(agent_dir, f"{DEFAULT_FLOW_VERSION}.yml")
            with open(filepath, "w") as f:
                f.write("existing content")

            mock_flow_config_model = _model(temp_dir)

            with pytest.raises(
                FileExistsError, match=f"File {filepath} already exists"
            ):
                save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

    def test_save_workflow_to_file_validates_config(self):
        """Test that save_workflow_to_file validates the flow config."""
        agent_id = "test_agent"
        flow_def = "version: v1\ncomponents:\n  - name: test_agent"

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_flow_config_model = _model(temp_dir)
            mock_flow_config_model.model_validate.side_effect = ValueError(
                "Invalid flow config"
            )

            with pytest.raises(ValueError, match="Invalid flow config"):
                save_workflow_to_file(agent_id, flow_def, mock_flow_config_model)

    def test_v1_config_is_written_as_a_feature(self, tmp_path, monkeypatch):
        """A v1 config lands at ai/features/foundational_agents/<agent>/config/1.0.0.yml."""
        store = tmp_path / "ai" / "features" / "foundational_agents"
        monkeypatch.setattr(script, "FOUNDATIONAL_AGENTS_DIR", store)
        flow_def = _V1_FLOW_DEF

        filepath = save_workflow_to_file("agent", flow_def, V1FlowConfig)

        assert filepath == str(
            store / "agent" / "config" / f"{DEFAULT_FLOW_VERSION}.yml"
        )
        assert (
            store / "agent" / "config" / f"{DEFAULT_FLOW_VERSION}.yml"
        ).read_text() == flow_def
        # The legacy root is never written for v1.
        assert not (V1FlowConfig.DIRECTORY_PATH / "agent").exists()

    def test_v1_config_refuses_to_overwrite(self, tmp_path, monkeypatch):
        """The exists-check applies to the nested <agent>/config/ layout too."""
        store = tmp_path / "foundational_agents"
        monkeypatch.setattr(script, "FOUNDATIONAL_AGENTS_DIR", store)
        existing = store / "agent" / "config" / f"{DEFAULT_FLOW_VERSION}.yml"
        existing.parent.mkdir(parents=True)
        existing.write_text("existing content")

        with pytest.raises(FileExistsError, match="already exists"):
            save_workflow_to_file("agent", _V1_FLOW_DEF, V1FlowConfig)

        assert existing.read_text() == "existing content"

    def test_v1_config_refuses_a_name_bundled_in_the_legacy_root(
        self, tmp_path, monkeypatch
    ):
        """A legacy flow of the same name would shadow the fetched copy at load time."""
        store = tmp_path / "foundational_agents"
        monkeypatch.setattr(script, "FOUNDATIONAL_AGENTS_DIR", store)
        legacy = tmp_path / "configs"
        (legacy / "agent").mkdir(parents=True)
        monkeypatch.setattr(V1FlowConfig, "DIRECTORY_PATH", legacy)

        with pytest.raises(FileExistsError, match="would shadow"):
            save_workflow_to_file("agent", _V1_FLOW_DEF, V1FlowConfig)

        assert not (store / "agent").exists()

    def test_v1_config_validation_error_writes_nothing(self, tmp_path, monkeypatch):
        """An invalid v1 config leaves no directory behind in the store."""
        store = tmp_path / "foundational_agents"
        monkeypatch.setattr(script, "FOUNDATIONAL_AGENTS_DIR", store)

        with pytest.raises(ValidationError):
            save_workflow_to_file(
                "agent", "version: v1\ncomponents: []\n", V1FlowConfig
            )

        assert not (store / "agent").exists()

    def test_v1_store_is_the_foundational_agents_domain(self):
        """The real store is the foundational_agents domain under ai/features."""
        assert (
            script.FOUNDATIONAL_AGENTS_DIR
            == default_features_dir() / "foundational_agents"
        )
        assert script.FOUNDATIONAL_AGENTS_DIR.parent.name == "features"
        assert not script.FOUNDATIONAL_AGENTS_DIR.is_relative_to(
            V1FlowConfig.DIRECTORY_PATH
        )


_AGENTS_YAML = (
    "agents:\n"
    "  - reference: agent_1\n"
    "    catalog_id: 123\n"
    "  - reference: agent_2\n"
    "    catalog_id: 456\n"
)


class TestLoadAgents:
    """Test cases for load_agents function."""

    def test_returns_the_entries(self, tmp_path):
        """The entries come back in file order."""
        agents_file = tmp_path / "agents.yml"
        agents_file.write_text(_AGENTS_YAML)

        agents = load_agents(agents_file)

        assert [(a.reference, a.catalog_id) for a in agents] == [
            ("agent_1", 123),
            ("agent_2", 456),
        ]

    def test_bundled_file_is_valid(self):
        """The file the Docker images read must always load."""
        agents = load_agents(DEFAULT_AGENTS_FILE)

        assert agents
        assert all(a.reference and a.catalog_id > 0 for a in agents)

    def test_bundled_file_is_tracked_and_shipped(self):
        """Fetched agents are ignored under foundational_agents/, the list must not be."""
        repo_root = default_features_dir().parents[1]
        relative = DEFAULT_AGENTS_FILE.relative_to(repo_root).as_posix()
        assert (
            relative == "ai/features/foundational_agents/config/ai-catalog-agents.yml"
        )

        for ignore_file in (".gitignore", ".dockerignore"):
            assert f"!{relative}\n" in (repo_root / ignore_file).read_text(), (
                f"{ignore_file} excludes {relative}"
            )

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("agents: []\n", "at least 1 item"),
            ("agents:\n  - reference: agent_1\n", "catalog_id"),
            ("agents:\n  - catalog_id: 123\n", "reference"),
            (
                "agents:\n  - reference: agent_1\n    catalog_id: abc\n",
                "catalog_id",
            ),
            (
                "agents:\n  - reference: agent_1\n    catalog_id: 1\n    id: 2\n",
                "id",
            ),
            ("other: []\n", "agents"),
            (
                "agents:\n"
                "  - reference: a\n    catalog_id: 1\n"
                "  - reference: a\n    catalog_id: 2\n",
                "Duplicate agent references",
            ),
        ],
    )
    def test_rejects_invalid_files(self, tmp_path, content, match):
        """Schema violations raise before anything is fetched."""
        agents_file = tmp_path / "agents.yml"
        agents_file.write_text(content)

        with pytest.raises(ValidationError, match=match):
            load_agents(agents_file)

    def test_missing_file(self, tmp_path):
        """A missing file is reported as such, not as invalid."""
        with pytest.raises(FileNotFoundError):
            load_agents(tmp_path / "missing.yml")

    @pytest.mark.parametrize("dockerfile", ["Dockerfile", "docker/fips/Dockerfile"])
    def test_dockerfile_list_matches_the_bundled_file(self, dockerfile):
        """GDK reads the agent list from the Dockerfile, so it must not drift from the file."""
        repo_root = default_features_dir().parents[1]
        # Same shape GDK's support/gitlab-ai-gateway-fetch-foundational-agents matches.
        match = re.search(
            r'fetch-foundational-agents "[^"]*" "[^"]*" "([^"]*)"',
            (repo_root / dockerfile).read_text(),
        )
        assert match, f"{dockerfile} has no line GDK can read the agent list from"

        assert parse_agent_list(match.group(1)) == load_agents(DEFAULT_AGENTS_FILE)


class TestParseAgentList:
    """Test cases for parse_agent_list function."""

    def test_returns_the_pairs(self):
        """Pairs come back in the given order with integer catalog IDs."""
        agents = parse_agent_list("agent_1:123,agent_2:456")

        assert [(a.reference, a.catalog_id) for a in agents] == [
            ("agent_1", 123),
            ("agent_2", 456),
        ]

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            ("agent_1", "catalog_id"),
            ("agent_1:abc", "catalog_id"),
            (":123", "reference"),
            ("", "reference"),
            ("a:1,a:2", "Duplicate agent references"),
        ],
    )
    def test_rejects_invalid_lists(self, value, match):
        """The list is validated with the same schema as the agents file."""
        with pytest.raises(ValidationError, match=match):
            parse_agent_list(value)


class TestParseArguments:
    """Test cases for parse_arguments function."""

    def test_parse_required_arguments(self):
        """Test parsing with required arguments."""
        test_args = [
            "http://test.com",
            "test-token",
            "--flow-registry-version",
            "v1",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.gitlab_url == "http://test.com"
        assert args.gitlab_token == "test-token"
        assert args.flow_registry_version == "v1"
        assert args.agents is None
        assert args.agents_file == str(DEFAULT_AGENTS_FILE)
        assert args.dry_run is None

    def test_parse_with_experimental_version(self):
        """Test parsing with experimental flow registry version."""
        test_args = [
            "http://test.com",
            "test-token",
            "--flow-registry-version",
            "experimental",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.flow_registry_version == "experimental"

    def test_parse_with_agents_file_and_dry_run(self):
        """Test parsing with an explicit agents file and the dry-run flag."""
        test_args = [
            "http://test.com",
            "test-token",
            "--flow-registry-version",
            "v1",
            "--agents-file",
            "my_agents.yml",
            "--dry-run",
            "true",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.agents_file == "my_agents.yml"
        assert args.dry_run == "true"

    def test_parse_with_positional_agent_list(self):
        """The `reference:id` list is still accepted as a third positional argument."""
        test_args = [
            "http://test.com",
            "test-token",
            "agent_1:123,agent_2:456",
            "--flow-registry-version",
            "v1",
        ]

        with patch("sys.argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.agents == "agent_1:123,agent_2:456"
        assert args.agents_file == str(DEFAULT_AGENTS_FILE)


class TestFetchAgents:
    """Test cases for fetch_agents main function."""

    @pytest.fixture
    def agents_file(self, tmp_path):
        """A two-entry agents file on disk."""
        path = tmp_path / "agents.yml"
        path.write_text(_AGENTS_YAML)
        return str(path)

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.fetch_foundational_agent"
    )
    @patch("builtins.print")
    def test_fetch_agents_dry_run_output(
        self, mock_print, mock_fetch_agent, mock_parse_args, agents_file
    ):
        """Test fetch_agents with dry-run output."""
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.agents_file = agents_file
        mock_args.agents = None
        mock_args.flow_registry_version = "v1"
        mock_args.dry_run = True
        mock_parse_args.return_value = mock_args

        mock_fetch_agent.side_effect = [
            ("agent_1", "version: v1\ncomponents:\n  - name: agent1"),
            ("agent_2", "version: v1\ncomponents:\n  - name: agent2"),
        ]

        fetch_agents()

        # One fetch per entry in the agents file
        assert mock_fetch_agent.call_count == 2
        mock_fetch_agent.assert_any_call("http://test.com", "token", "agent_1", 123)
        mock_fetch_agent.assert_any_call("http://test.com", "token", "agent_2", 456)

        # Verify output to stdout - should print separator, name, and flow config for each workflow
        expected_calls = [
            unittest.mock.call("-----"),
            unittest.mock.call("agent_1"),
            unittest.mock.call("version: v1\ncomponents:\n  - name: agent1"),
            unittest.mock.call("-----"),
            unittest.mock.call("agent_2"),
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
        tmp_path,
    ):
        """Test fetch_agents with file output."""
        agents_file = tmp_path / "agents.yml"
        agents_file.write_text("agents:\n  - reference: agent_1\n    catalog_id: 123\n")

        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.agents_file = str(agents_file)
        mock_args.agents = None
        mock_args.flow_registry_version = "v1"
        mock_args.dry_run = None
        mock_parse_args.return_value = mock_args

        mock_fetch_agent.return_value = (
            "agent_1",
            "version: v1\ncomponents:\n  - name: agent1",
        )
        mock_save_workflow.return_value = (
            f"/tmp/output/agent_1/{DEFAULT_FLOW_VERSION}.yml"
        )

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
            f"Successfully saved workflow definition(s): ['/tmp/output/agent_1/{DEFAULT_FLOW_VERSION}.yml']",
            file=sys.stderr,
        )

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    def test_fetch_agents_invalid_flow_registry_version(
        self, mock_parse_args, agents_file
    ):
        """Test fetch_agents raises error for invalid flow registry version."""
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.agents_file = agents_file
        mock_args.agents = None
        mock_args.flow_registry_version = "invalid"
        mock_parse_args.return_value = mock_args

        with pytest.raises(
            ValueError, match="Flow Registry version 'invalid' is not supported"
        ):
            fetch_agents()

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.fetch_foundational_agent"
    )
    def test_fetch_agents_invalid_agents_file_fetches_nothing(
        self, mock_fetch_agent, mock_parse_args, tmp_path
    ):
        """The agents file is validated before any catalog request."""
        agents_file = tmp_path / "agents.yml"
        agents_file.write_text("agents: []\n")

        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.agents_file = str(agents_file)
        mock_args.agents = None
        mock_args.flow_registry_version = "v1"
        mock_parse_args.return_value = mock_args

        with pytest.raises(ValidationError):
            fetch_agents()

        mock_fetch_agent.assert_not_called()

    @patch("duo_workflow_service.scripts.fetch_foundational_agents.parse_arguments")
    @patch(
        "duo_workflow_service.scripts.fetch_foundational_agents.fetch_foundational_agent"
    )
    def test_fetch_agents_positional_list_replaces_the_file(
        self, mock_fetch_agent, mock_parse_args, agents_file
    ):
        """A positional list is fetched and the agents file is not read."""
        mock_args = Mock()
        mock_args.gitlab_url = "http://test.com"
        mock_args.gitlab_token = "token"
        mock_args.agents = "agent_3:789"
        mock_args.agents_file = agents_file
        mock_args.flow_registry_version = "v1"
        mock_args.dry_run = True
        mock_parse_args.return_value = mock_args
        mock_fetch_agent.return_value = ("agent_3", "flow: 3")

        fetch_agents()

        mock_fetch_agent.assert_called_once_with(
            "http://test.com", "token", "agent_3", 789
        )
