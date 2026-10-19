import json
from typing import ClassVar

import pytest

from ai_gateway.response_schemas.base import BaseAgentOutput


class SampleOutput(BaseAgentOutput):
    tool_title: ClassVar[str] = "sample_output"

    message: str


class TestToStringOutput:
    @pytest.mark.parametrize(
        "message",
        [
            "Plain text.",
            "Use `inline` code.",
            # A value that tries to close the fence and add Markdown after it.
            "Before\n```\n# Injected heading\n```suggestion\nafter",
        ],
    )
    def test_wraps_json_in_a_single_code_fence(self, message):
        output = SampleOutput(message=message).to_string_output()
        lines = output.split("\n")

        assert lines[0] == "```json"
        assert lines[-1] == "```"
        # String values keep newlines escaped, so no line inside can close the fence.
        assert not any(line.lstrip().startswith("```") for line in lines[1:-1])
        assert json.loads("\n".join(lines[1:-1])) == {"message": message}
