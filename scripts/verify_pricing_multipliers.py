#!/usr/bin/env python
"""Verify all billable models have corresponding pricing multipliers in CustomersDot.

This script prevents shipping new models without billing configuration, which would
cause unbillable usage events and lost revenue.

It compares models from two sources against pricing_multipliers.yml in CustomersDot:
1. Selectable models in unit_primitives.yml (models users can select in the UI)
2. Proxy models in models.yml (models with proxy_provider set, accessible via proxy endpoint)

The script fails if any billable model is missing a pricing multiplier.

Note: Designed for GitLab CI where CI_JOB_TOKEN is automatically available.
AI Gateway must be in CustomersDot's CI/CD Job Token Allowlist.
"""

import os
import sys
from urllib.parse import quote_plus

import requests
import yaml

from ai_gateway.model_selection import ModelSelectionConfig

CUSTOMERSDOT_PROJECT = "gitlab-org/customers-gitlab-com"
PRICING_FILE_PATH = "config/billing/pricing_multipliers.yml"


def normalize_model_id(model_id: str) -> str:
    """Normalize model ID to match CustomersDot's pricing lookup format.

    Vertex models use @ in their IDs (e.g., "claude-sonnet-4-5@20250929"),
    but CustomersDot's UsagePricingRule converts @ to - when looking up pricing.

    See: customers-gitlab-com/app/models/billing/conversion/usage_pricing_rule.rb
    """
    return model_id.replace("@", "-")


def fetch_pricing_multipliers() -> dict:
    """Fetch pricing_multipliers.yml from CustomersDot via GitLab API."""
    # pylint: disable=direct-environment-variable-reference
    token = os.environ.get("CI_JOB_TOKEN")
    # pylint: enable=direct-environment-variable-reference
    if not token:
        print("WARNING: CI_JOB_TOKEN not set, API request may fail.", file=sys.stderr)

    url = (
        f"https://gitlab.com/api/v4/projects/{quote_plus(CUSTOMERSDOT_PROJECT)}"
        f"/repository/files/{quote_plus(PRICING_FILE_PATH)}/raw"
    )
    headers = {"JOB-TOKEN": token} if token else {}

    response = requests.get(url, headers=headers, params={"ref": "main"}, timeout=30)

    if response.status_code == 401:
        print(
            "ERROR: Authentication failed. Ensure CI_JOB_TOKEN is available "
            "and AI Gateway is in CustomersDot's job token allowlist.",
            file=sys.stderr,
        )
        sys.exit(1)

    if response.status_code == 404:
        print(
            f"ERROR: Could not find {PRICING_FILE_PATH} in CustomersDot.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not response.ok:
        print(
            f"ERROR: Failed to fetch pricing multipliers: {response.status_code}",
            file=sys.stderr,
        )
        sys.exit(1)

    return yaml.safe_load(response.text)


def get_selectable_models() -> dict[str, str]:
    """Get all selectable model IDs mapped to their gitlab_identifier.

    Reads models.yml and unit_primitives.yml to find all models that users can
    select in the UI. These are the models that need pricing multipliers.

    Returns:
        Dict mapping normalized model IDs to gitlab_identifiers.
        Example: {"claude-sonnet-4-5-20250929": "claude_sonnet_4_5_20250929_vertex"}
    """
    config = ModelSelectionConfig.instance()
    llm_definitions = config.get_llm_definitions()

    # Collect all gitlab_identifiers that appear in unit_primitives.yml
    # These are models users can select, so they need pricing multipliers
    selectable_gitlab_ids = set()
    for up_config in config.get_unit_primitive_config():
        # Skip code_completions - uses flat rate pricing, not model-based multipliers
        # Skip embeddings_code - pricing to be set as part of
        #   https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1985
        if up_config.feature_setting in ("code_completions", "embeddings_code"):
            continue

        selectable_gitlab_ids.add(up_config.default_model)
        selectable_gitlab_ids.update(up_config.selectable_models)
        if up_config.dev:
            selectable_gitlab_ids.update(up_config.dev.selectable_models)

    # Map each gitlab_identifier to its params.model value
    # params.model is what gets sent in billing events
    result = {}
    for gitlab_id in selectable_gitlab_ids:
        if gitlab_id not in llm_definitions:
            print(f"WARNING: '{gitlab_id}' not found in models.yml", file=sys.stderr)
            continue

        model_id = llm_definitions[gitlab_id].params.model
        if not model_id:
            print(f"WARNING: No params.model for '{gitlab_id}'", file=sys.stderr)
            continue

        normalized = normalize_model_id(model_id)
        result[normalized] = gitlab_id

    return result


def get_proxy_models() -> dict[str, str]:
    """Get all proxy model IDs mapped to their gitlab_identifier.

    Reads models.yml to find all models with proxy_provider set. These models
    are accessible via the proxy endpoint and need pricing multipliers.

    Returns:
        Dict mapping normalized model IDs to gitlab_identifiers.
        Example: {"claude-sonnet-4-5-20250929": "claude_sonnet_4_5_20250929"}
    """
    config = ModelSelectionConfig.instance()
    llm_definitions = config.get_llm_definitions()

    result = {}
    for gitlab_id, llm_def in llm_definitions.items():
        # Only include models with proxy_provider set
        if not llm_def.proxy_provider:
            continue

        model_id = llm_def.params.model
        if not model_id:
            print(
                f"WARNING: No params.model for proxy model '{gitlab_id}'",
                file=sys.stderr,
            )
            continue

        normalized = normalize_model_id(model_id)
        result[normalized] = gitlab_id

    return result


def get_pricing_keys(pricing_config: dict) -> set[str]:
    """Extract model keys that have pricing multipliers defined.

    Looks in the agent_llm_request category since that's where LLM model pricing is configured in CustomersDot.
    """
    try:
        resource_multipliers = pricing_config["default"]["agent_llm_request"][
            "resource_multipliers"
        ]
        return set(resource_multipliers.keys())
    except KeyError as e:
        print(
            f"ERROR: Unexpected pricing_multipliers.yml structure: missing {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    print("Fetching pricing multipliers from CustomersDot...")
    pricing_keys = get_pricing_keys(fetch_pricing_multipliers())

    print("Loading selectable models from unit_primitives.yml...")
    selectable_models = get_selectable_models()

    print("Loading proxy models from models.yml...")
    proxy_models = get_proxy_models()

    # Combine both sources - proxy models may overlap with selectable models
    # Use selectable_models as base, then add any proxy-only models
    all_billable_models = dict(selectable_models)
    for model_id, gitlab_id in proxy_models.items():
        if model_id not in all_billable_models:
            all_billable_models[model_id] = gitlab_id

    print(
        f"Checking {len(all_billable_models)} billable models "
        f"({len(selectable_models)} selectable, {len(proxy_models)} proxy)...\n"
    )

    # Find models without pricing multipliers
    missing = [
        (model_id, gitlab_id)
        for model_id, gitlab_id in sorted(all_billable_models.items())
        if model_id not in pricing_keys
    ]

    if missing:
        print("MISSING PRICING MULTIPLIERS:", file=sys.stderr)
        for model_id, gitlab_id in missing:
            print(f"  - {model_id} (gitlab_identifier: {gitlab_id})", file=sys.stderr)
        print(
            f"\nERROR: {len(missing)} model(s) missing pricing multipliers.",
            file=sys.stderr,
        )
        print("\nAdd them to CustomersDot BEFORE merging:", file=sys.stderr)
        print(
            f"https://gitlab.com/{CUSTOMERSDOT_PROJECT}/-/blob/main/{PRICING_FILE_PATH}",
            file=sys.stderr,
        )
        return 1

    print(f"All {len(all_billable_models)} billable models have pricing multipliers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
