"""Step 2: the provider's own definition of the fields we send must not have changed.

Only Bedrock is covered. Its schema ships in `botocore`, so the check runs offline. For every field in our recorded
Bedrock request schema, we record its type and whether it is required in the Bedrock Converse API. The test fails if
a field is missing from the provider definition, or if its type or required flag changed. To accept an intended
change, review the diff, then rewrite the file:

    UPDATE_LITELLM_SCHEMAS=1 poetry run pytest tests/litellm_bump/test_provider_schemas.py
"""

import json
from pathlib import Path
from typing import Any

import botocore.session
import pytest
from botocore.model import Shape

SCHEMAS = Path(__file__).parent / "schemas"


def converse_input_shape() -> Shape:
    model = botocore.session.get_session().get_service_model("bedrock-runtime")
    return model.operation_model("Converse").input_shape


def describe_fields(
    body: Any, shape: Shape, prefix: str = "", required: bool = False
) -> dict[str, dict[str, Any]]:
    """Map each field path in our request body to its type and required flag in the provider definition."""
    if shape.type_name == "list":
        items = describe_fields(body[0], shape.member, prefix + "[]", False)
        return {prefix: {"type": "list", "required": required}, **items}

    fields: dict[str, dict[str, Any]] = {}
    if prefix:
        fields[prefix] = {"type": shape.type_name, "required": required}

    if isinstance(body, dict):
        assert shape.type_name == "structure", (
            f"{prefix or 'body'} is not an object in the provider definition"
        )
        required_members = set(shape.required_members)
        for key, value in body.items():
            assert key in shape.members, (
                f"Bedrock Converse does not define `{prefix + '.' if prefix else ''}{key}`. "
                f"Defined members: {sorted(shape.members)}"
            )
            path = f"{prefix}.{key}" if prefix else key
            fields.update(
                describe_fields(
                    value, shape.members[key], path, key in required_members
                )
            )
    return fields


def test_bedrock_fields_we_send_match_the_provider_definition(update_schemas: bool):
    recorded = json.loads((SCHEMAS / "bedrock_claude.json").read_text(encoding="utf-8"))
    # modelId is part of the URL path, not the body, so it is not in our recorded body.
    actual = describe_fields(recorded["body"], converse_input_shape())

    path = SCHEMAS / "bedrock_provider.json"
    if update_schemas:
        path.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_provider_check_rejects_a_field_the_provider_does_not_define():
    with pytest.raises(AssertionError, match="does not define"):
        describe_fields(
            {"inferenceConfig": {"max_tokens": "number"}}, converse_input_shape()
        )
