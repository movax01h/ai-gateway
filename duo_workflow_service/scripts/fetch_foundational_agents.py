#!/usr/bin/env python3
"""Script to sync foundational agents from GitLab AI Catalog.

Usage:
    python fetch_foundational_agents.py <gitlab_url> <gitlab_token> [<agents>]
    --flow-registry-version VERSION [--agents-file PATH] [--dry-run]

Arguments:
    gitlab_url: GitLab base URL (e.g., http://gdk.test:3000)
    gitlab_token: GitLab API token for authentication
    agents: Comma-separated list of `reference:catalog_id` pairs. Optional;
        when given, it is used instead of the agents file.
    --flow-registry-version: Flow Registry syntax version to fetch (experimental or v1). Required.
    --agents-file: YAML file that lists the agents to fetch. Defaults to
        ai/features/foundational_agents/config/ai-catalog-agents.yml. Ignored when `agents` is given.
    --dry-run: If provided, prints to stdout instead of saving YAML files.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from requests import request

from duo_workflow_service.agent_platform.experimental.flows import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows import FlowConfig as V1FlowConfig
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    DEFAULT_FLOW_VERSION,
)
from lib.feature_roots import default_features_dir

FETCH_AGENT_OPERATION_NAME = "aiCatalogAgent"
FETCH_AGENT_QUERY = """
query aiCatalogAgent($id: AiCatalogItemID!) {
    aiCatalogItem(id: $id) {
        name
        latestVersion {
            id
        }
    }
}
"""

FETCH_FLOW_OPERATION_NAME = "agentFlowConfig"
FETCH_FLOW_DEFINITION = """
query agentFlowConfig($agentVersionId: AiCatalogItemVersionID!) {
    aiCatalogAgentFlowConfig(agentVersionId: $agentVersionId, flowConfigType: CHAT)
}
"""


class FoundationalAgentEntry(BaseModel):
    """One entry under ``agents:`` in the agents file."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(min_length=1)
    catalog_id: int


class FoundationalAgentsFile(BaseModel):
    """Schema of the agents file, see ``DEFAULT_AGENTS_FILE``."""

    model_config = ConfigDict(extra="forbid")

    agents: List[FoundationalAgentEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_are_unique(self) -> "FoundationalAgentsFile":
        references = [agent.reference for agent in self.agents]
        duplicates = sorted({r for r in references if references.count(r) > 1})
        if duplicates:
            raise ValueError(f"Duplicate agent references: {duplicates}")
        return self


def load_agents(path: str | os.PathLike[str]) -> List[FoundationalAgentEntry]:
    """Read the agents file and return its validated entries.

    Args:
        path: Path to a YAML file with the ``FoundationalAgentsFile`` schema,
            see ``DEFAULT_AGENTS_FILE``.

    Returns:
        The entries under ``agents:``, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the file does not match the schema: no
            ``agents`` list, an empty list, a missing or unknown key, or a
            duplicate ``reference``.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return FoundationalAgentsFile.model_validate(data).agents


def parse_agent_list(value: str) -> List[FoundationalAgentEntry]:
    """Parse the positional ``reference:catalog_id,...`` agent list.

    Args:
        value: Comma-separated ``reference:catalog_id`` pairs, for example
            ``"duo_planner:348,onboarding_guide:1003979"``.

    Returns:
        One entry per pair, in the given order.

    Raises:
        pydantic.ValidationError: If a pair has no ``:``, has an empty
            ``reference``, has a non-integer ``catalog_id``, or if a
            ``reference`` appears twice.
    """
    entries = []
    for pair in value.split(","):
        reference, _, catalog_id = pair.partition(":")
        entries.append({"reference": reference, "catalog_id": catalog_id})

    return FoundationalAgentsFile.model_validate({"agents": entries}).agents


def graphql_request(
    url: str,
    token: str,
    query: str,
    operation_name: str,
    variables: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make a GraphQL request to the GitLab API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "query": query,
        "variables": variables,
        "operationName": operation_name,
    }

    response = request("POST", url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_foundational_agent(
    gitlab_url: str, gitlab_token: str, reference: str, catalog_id: int
) -> Tuple[str, str]:
    """Sync a single foundational agent and return its workflow definition."""

    agent_response: Dict[str, Any] = graphql_request(
        f"{gitlab_url.rstrip('/')}/api/graphql",
        gitlab_token,
        FETCH_AGENT_QUERY,
        FETCH_AGENT_OPERATION_NAME,
        {"id": f"gid://gitlab/Ai::Catalog::Item/{catalog_id}"},
    )

    if "errors" in agent_response:
        raise RuntimeError(agent_response["errors"])

    version_id: str = agent_response["data"]["aiCatalogItem"]["latestVersion"]["id"]

    if not version_id:
        raise RuntimeError("Version not found")

    flow_response: Dict[str, Any] = graphql_request(
        f"{gitlab_url.rstrip('/')}/api/graphql",
        gitlab_token,
        FETCH_FLOW_DEFINITION,
        FETCH_FLOW_OPERATION_NAME,
        {"agentVersionId": version_id},
    )

    flow_config: str = flow_response["data"]["aiCatalogAgentFlowConfig"]

    return reference, flow_config


# Domain under ai/features/ that holds catalog-fetched flows. Each agent is a
# regular feature dir, so flow-config discovery picks it up with no special case.
FOUNDATIONAL_AGENTS_DIR = default_features_dir() / "foundational_agents"

# The list of agents the Docker images fetch at build time. It sits next to the
# agents it describes, so `COPY ai/ ai/` ships it with the image.
DEFAULT_AGENTS_FILE = FOUNDATIONAL_AGENTS_DIR / "config" / "ai-catalog-agents.yml"


def _agent_config_dir(
    agent_id: str,
    flow_config_model: type[V1FlowConfig] | type[ExperimentalFlowConfig],
) -> str:
    """Return the directory a fetched *agent_id* is written to for this registry version."""
    if flow_config_model is V1FlowConfig:
        return str(FOUNDATIONAL_AGENTS_DIR / agent_id / "config")
    # The experimental registry has no feature roots; keep its legacy layout.
    return os.path.join(flow_config_model.DIRECTORY_PATH, agent_id)


def save_workflow_to_file(
    agent_id: str,
    flow_def: str,
    flow_config_model: type[V1FlowConfig] | type[ExperimentalFlowConfig],
) -> str:
    """Save a workflow definition to a versioned YAML file.

    A v1 config is stored as a feature under ``ai/features/foundational_agents/``::

        ai/features/foundational_agents/{agent_id}/config/{DEFAULT_FLOW_VERSION}.yml

    Flow-config discovery registers it like any bundled feature. A fetched
    agent must not share its name with a bundled feature (discovery fails at
    boot) or with a legacy flow under ``V1FlowConfig.DIRECTORY_PATH`` (the
    legacy copy wins at load time, so this function refuses to write it). An
    experimental config keeps the legacy layout
    ``{DIRECTORY_PATH}/{agent_id}/{DEFAULT_FLOW_VERSION}.yml``.

    **Current limitation — always saves as version 1.0.0**

    This function always writes the fetched flow config as ``1.0.0.yml``,
    regardless of the version declared inside the YAML itself.  Bundling
    multiple major versions side by side (e.g. ``1.0.0.yml`` and ``2.0.0.yml``)
    for self-hosted instances is not yet supported.  If you need to update an
    existing flow, delete the old file first and re-run the script, or edit the
    file in place.

    **Raises ``FileExistsError`` if the target file already exists**

    The function does *not* overwrite an existing file.  If
    the target ``1.0.0.yml`` already exists the call raises
    ``FileExistsError`` and no data is written.  This is intentional: it
    prevents accidental overwrites of hand-edited or previously synced configs.
    To update an existing flow, remove the old file manually before running the
    script.

    Args:
        agent_id: The flow name used as the subdirectory name (e.g. ``"my_flow"``).
            This is the ``reference`` of an entry in the agents file.
        flow_def: Raw YAML string fetched from the AI Catalog.  It is validated
            against ``flow_config_model`` before being written to disk.
        flow_config_model: The Pydantic model class that corresponds to the
            requested flow registry version (``V1FlowConfig`` or
            ``ExperimentalFlowConfig``).  It selects the target layout, see above.

    Returns:
        The absolute path of the file that was written.

    Raises:
        FileExistsError: If the target file already exists (no overwrite).
        yaml.YAMLError: If ``flow_def`` is not valid YAML.
        pydantic.ValidationError: If the parsed YAML does not conform to
            ``flow_config_model``.

    Example::

        # Fetch and save the agents listed in my_agents.yml using the v1 flow
        # registry schema:
        #
        #   python fetch_foundational_agents.py \\
        #       http://gdk.test:3000 $TOKEN \\
        #       --flow-registry-version v1 --agents-file my_agents.yml
        #
        # For an entry with reference "my_flow" this writes:
        #   ai/features/foundational_agents/my_flow/config/1.0.0.yml
    """
    # This always saves as the default version (1.0.0). Bundling multiple
    # major versions side by side for self-hosted instances is not yet
    # supported.
    agent_dir: str = _agent_config_dir(agent_id, flow_config_model)
    filepath: str = os.path.join(agent_dir, f"{DEFAULT_FLOW_VERSION}.yml")

    # The legacy root wins over feature roots at load time, so a same-named
    # bundled flow would silently shadow the fetched copy. Fail here instead.
    if flow_config_model is V1FlowConfig:
        legacy_dir = V1FlowConfig.DIRECTORY_PATH / agent_id
        if legacy_dir.exists():
            raise FileExistsError(
                f"Bundled flow {legacy_dir} would shadow the fetched {agent_id!r}"
            )

    if os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists")

    # parse yaml string to a dictionary
    flow_config_dict = yaml.safe_load(flow_def)
    # validate the dictionary against the pydantic model
    flow_config_model.model_validate(flow_config_dict)

    os.makedirs(agent_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(flow_def)
    return filepath


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync foundational agents from GitLab AI Catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("gitlab_url", help="GitLab base URL")
    parser.add_argument("gitlab_token", help="GitLab API token")
    parser.add_argument(
        "agents",
        nargs="?",
        help="Comma-separated reference:catalog_id pairs. Replaces --agents-file when given.",
    )
    parser.add_argument(
        "--flow-registry-version",
        required=True,
        help="Flow Registry syntax version to fetch. Allowed values: 'experimental' or 'v1'.",
    )
    parser.add_argument(
        "--agents-file",
        default=str(DEFAULT_AGENTS_FILE),
        help="YAML file that lists the agents to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        help="If provided, prints to stdout instead of saving YAML files",
    )

    return parser.parse_args()


def fetch_agents() -> None:
    """Main function to parse arguments and sync foundational agents."""
    args: argparse.Namespace = parse_arguments()

    # Validate flow registry version
    if args.flow_registry_version not in ["experimental", "v1"]:
        raise ValueError(
            f"Flow Registry version '{args.flow_registry_version}' is not supported. Allowed values: experimental, v1"
        )

    # Map version string to its corresponding class
    version_to_class: dict[str, type[ExperimentalFlowConfig] | type[V1FlowConfig]] = {
        "experimental": ExperimentalFlowConfig,
        "v1": V1FlowConfig,
    }
    flow_config_class = version_to_class[args.flow_registry_version]

    agents: List[FoundationalAgentEntry] = (
        parse_agent_list(args.agents) if args.agents else load_agents(args.agents_file)
    )

    workflow_definitions: list[tuple[str, str]] = [
        fetch_foundational_agent(
            args.gitlab_url, args.gitlab_token, agent.reference, agent.catalog_id
        )
        for agent in agents
    ]

    if args.dry_run:
        for flow_name, flow_def in workflow_definitions:
            print("-----")
            print(flow_name)
            print(flow_def)
    else:
        # Save to file
        saved_files: List[str] = [
            save_workflow_to_file(file_name, flow_definition, flow_config_class)
            for (file_name, flow_definition) in workflow_definitions
        ]

        print(
            f"Successfully saved workflow definition(s): {saved_files}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    fetch_agents()
