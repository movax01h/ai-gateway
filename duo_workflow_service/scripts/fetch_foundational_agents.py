#!/usr/bin/env python3
"""Script to sync foundational agents from GitLab AI Catalog.

Usage:
    python fetch_foundational_agents.py <gitlab_url> <gitlab_token> <foundational_agent_ids> [--output-path <path>]

Arguments:
    gitlab_url: GitLab GraphQL API URL (e.g., http://gdk.test:3000/api/graphql)
    gitlab_token: GitLab API token for authentication
    foundational_agent_ids: Comma-separated list of foundational agent IDs (e.g., "348,349,350")
    --output-path: Optional directory path to save YAML files. If not provided, prints to stdout.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from requests import request

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
    gitlab_url: str, gitlab_token: str, agent_id: int
) -> Tuple[str, str]:
    """Sync a single foundational agent and return its workflow definition."""

    agent_response: Dict[str, Any] = graphql_request(
        f"{gitlab_url.rstrip('/')}/api/graphql",
        gitlab_token,
        FETCH_AGENT_QUERY,
        FETCH_AGENT_OPERATION_NAME,
        {"id": f"gid://gitlab/Ai::Catalog::Item/{agent_id}"},
    )

    if "errors" in agent_response:
        raise RuntimeError(agent_response["errors"])

    version_id: str = agent_response["data"]["aiCatalogItem"]["latestVersion"]["id"]
    name: str = agent_response["data"]["aiCatalogItem"]["name"]

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

    return name.replace(" ", "_").lower(), flow_config


def save_workflow_to_file(agent_id: str, flow_def: str, output_path: str) -> str:
    """Save a workflow definition to a YAML file."""
    filename: str = f"{agent_id}.yml"
    filepath: str = os.path.join(output_path, filename)

    if os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists")

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
        "--output-path",
        help="Directory path to save YAML files. If not provided, prints to stdout.",
    )

    return parser.parse_args()


def fetch_agents() -> None:
    """Main function to parse arguments and sync foundational agents."""
    args: argparse.Namespace = parse_arguments()

    # Parse agent IDs
    try:
        agent_ids: List[int] = [
            int(id.strip()) for id in args.foundational_agent_ids.split(",")
        ]
    except ValueError:
        print(
            "Error: foundational_agent_ids must be comma-separated integers",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate output path if provided
    if args.output_path:
        if not os.path.exists(args.output_path):
            raise ValueError(
                f"Output path does not exist: {args.output_path}",
            )

    workflow_definitions: list[tuple[str, str]] = [
        fetch_foundational_agent(args.gitlab_url, args.gitlab_token, agent_id)
        for agent_id in agent_ids
    ]

    if args.output_path:
        # Save to file
        saved_files: List[str] = [
            save_workflow_to_file(file_name, flow_definition, args.output_path)
            for (file_name, flow_definition) in workflow_definitions
        ]

        print(
            f"Successfully saved workflow definition(s): {saved_files}",
            file=sys.stderr,
        )
    else:
        for flow_name, flow_def in workflow_definitions:
            print("-----")
            print(flow_name)
            print(flow_def)


if __name__ == "__main__":
    fetch_agents()
