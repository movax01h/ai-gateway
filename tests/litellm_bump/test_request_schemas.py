"""Step 1: the request litellm builds for each provider must match the schema we recorded.

The recorded schemas are in `tests/litellm_bump/schemas/`. To accept an intended change, review the diff in the
failure, then rewrite the files:

    UPDATE_LITELLM_SCHEMAS=1 poetry run pytest tests/litellm_bump/test_request_schemas.py
"""

import json
from pathlib import Path

import pytest

from tests.litellm_bump.conftest import FakeProviders
from tests.litellm_bump.scenarios import SCENARIOS
from tests.litellm_bump.shapes import request_shape

SCHEMAS = Path(__file__).parent / "schemas"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_request_matches_recorded_schema(
    name: str,
    providers: FakeProviders,
    fake_vertex_auth,
    fake_aws,
    update_schemas: bool,
):
    text = await SCENARIOS[name](providers)

    assert text == "hi", "litellm did not parse the canned response"

    actual = request_shape(providers.last)
    path = SCHEMAS / f"{name}.json"

    if update_schemas:
        path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")

    assert actual == json.loads(path.read_text(encoding="utf-8"))
