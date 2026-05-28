import json

import pytest
from pydantic import ValidationError

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.clarification_question import (
    ClarificationOption,
    ClarificationQuestionInput,
    ClarificationQuestionTool,
)


@pytest.fixture(name="tool")
def tool_fixture() -> ClarificationQuestionTool:
    return ClarificationQuestionTool()


def _make_option(suffix: str, recommended: bool = False) -> ClarificationOption:
    return ClarificationOption(
        id=f"opt_{suffix}",
        label=f"Option {suffix}",
        description=f"Description for option {suffix}.",
        recommended=recommended,
    )


@pytest.fixture(name="sample_options")
def sample_options_fixture() -> list[ClarificationOption]:
    return [
        ClarificationOption(
            id="use_jwt",
            label="Use JWT",
            description="Stateless token-based authentication.",
            recommended=True,
        ),
        ClarificationOption(
            id="use_oauth",
            label="Use OAuth",
            description="Delegated authentication via an identity provider.",
        ),
    ]


@pytest.mark.asyncio
async def test_execute_returns_question_and_options_as_json(
    tool: ClarificationQuestionTool, sample_options: list[ClarificationOption]
):
    result = await tool._execute(
        question="Which authentication should we use?",
        options=sample_options,
        unexpected_kwarg="ignored",
    )

    assert json.loads(result) == {
        "question": "Which authentication should we use?",
        "options": [
            {
                "id": "use_jwt",
                "label": "Use JWT",
                "description": "Stateless token-based authentication.",
                "recommended": True,
            },
            {
                "id": "use_oauth",
                "label": "Use OAuth",
                "description": "Delegated authentication via an identity provider.",
                "recommended": False,
            },
        ],
    }


def test_tool_metadata(tool: ClarificationQuestionTool):
    assert tool.name == "clarification_question"
    assert tool.args_schema is ClarificationQuestionInput
    assert tool.trust_level == ToolTrustLevel.TRUSTED_INTERNAL


def test_format_display_message(
    tool: ClarificationQuestionTool, sample_options: list[ClarificationOption]
):
    args = ClarificationQuestionInput(
        question="Which framework should we use?",
        options=sample_options,
    )

    assert (
        tool.format_display_message(args)
        == "Asking clarification: Which framework should we use?"
    )


@pytest.mark.parametrize("count", [2, 3])
def test_options_accepts_valid_lengths(count: int):
    options = [_make_option(str(i)) for i in range(count)]

    model = ClarificationQuestionInput(question="Q?", options=options)

    assert len(model.options) == count


@pytest.mark.parametrize("count", [0, 1, 4])
def test_options_rejects_invalid_lengths(count: int):
    options = [_make_option(str(i)) for i in range(count)]

    with pytest.raises(ValidationError):
        ClarificationQuestionInput(question="Q?", options=options)
