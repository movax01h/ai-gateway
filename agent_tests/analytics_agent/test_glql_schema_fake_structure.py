"""Check that the fake GLQL schema still has the structure of the live one.

Not an agent test: it calls no model and needs no API key. It lives here because the fake only matters when the agent
tests run, and `make test-agents` already reruns a test that fails on a network error.
"""

import os

import pytest

from scripts.check_glql_schema_fake import (
    FAKE_PATH,
    drift_report,
    fetch_schema,
    load_json,
)


def test_fake_glql_schema_has_the_structure_of_the_live_one():
    # pylint: disable=direct-environment-variable-reference
    gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com").rstrip("/")
    live = fetch_schema(gitlab_url)
    fake = load_json(FAKE_PATH)

    messages = drift_report(live, fake)

    if messages:
        pytest.fail(
            f"FAKE GLQL SCHEMA DRIFTED from the live schema ({gitlab_url}, version {live.get('version')}; "
            f"the fake declares {fake.get('version')}). This is not an agent regression. Update the fake's version "
            f"along with its entries:\n" + "\n".join(messages)
        )
