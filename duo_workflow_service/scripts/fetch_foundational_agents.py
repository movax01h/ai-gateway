#!/usr/bin/env python3
"""Script to sync foundational agents from GitLab AI Catalog.

Usage:
    python fetch_foundational_agents.py <gitlab_url> <gitlab_token> <foundational_agent_ids>
    [--flow-registry-version VERSION] [--dry-run]

Arguments:
    gitlab_url: GitLab GraphQL API URL (e.g., http://gdk.test:3000/api/graphql)
    gitlab_token: GitLab API token for authentication
    foundational_agent_ids: Comma-separated list of foundational agent IDs (e.g., "348,349,350")
    --flow-registry-version: Flow Registry syntax version to fetch (experimental or v1). Required.
    --dry-run: If provided, prints to stdout instead of saving YAML files.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from requests import request

from duo_workflow_service.agent_platform.experimental.flows import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows import FlowConfig as V1FlowConfig

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
    gitlab_url: str, gitlab_token: str, agent_id: str
) -> Tuple[str, str]:
    """Sync a single foundational agent and return its workflow definition."""

    file_name, catalog_id = agent_id.split(":")

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

    return file_name, flow_config


def save_workflow_to_file(
    agent_id: str,
    flow_def: str,
    flow_config_model: type[V1FlowConfig] | type[ExperimentalFlowConfig],
) -> str:
    """Save a workflow definition to a YAML file."""
    filename: str = f"{agent_id}.yml"
    filepath: str = os.path.join(flow_config_model.DIRECTORY_PATH, filename)

    if os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists")
    # parse yaml string to a dictionary

    flow_config_dict = yaml.safe_load(flow_def)
    # validate the dictionary against the pydantic model
    flow_config_model.model_validate(flow_config_dict)

    with open(filepath, "w") as f:
        f.write(flow_def)
    return filepath


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync foundational agents from GitLab AI Catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("gitlab_url", help="GitLab GraphQL API URL")
    parser.add_argument("gitlab_token", help="GitLab API token")
    parser.add_argument(
        "foundational_agent_ids", help="Comma-separated list of agent IDs"
    )
    parser.add_argument(
        "--flow-registry-version",
        required=True,
        help="Flow Registry syntax version to fetch. Allowed values: 'experimental' or 'v1'.",
    )
    parser.add_argument(
        "--dry-run",
        help="If provided, prints to stdout instead of saving YAML files",
    )

    return parser.parse_args()


def fetch_agents() -> None:
    """Main function to parse arguments and sync foundational agents."""
    args: argparse.Namespace = parse_arguments()

    agent_ids: List[str] = args.foundational_agent_ids.split(",")

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

    workflow_definitions: list[tuple[str, str]] = [
        fetch_foundational_agent(args.gitlab_url, args.gitlab_token, agent_id)
        for agent_id in agent_ids
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
