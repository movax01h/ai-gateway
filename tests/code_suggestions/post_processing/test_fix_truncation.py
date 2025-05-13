import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.ops import fix_truncation

# codespell:ignore-begin
RUBY_SAMPLE_1 = (
    # prefix
    "Event.new(\n",
    # truncated completion that should be trimmed back
    "    name: params[:name],\n    config: { repeat: tru",
    # suffix
    "\n)",
)
# codespell:ignore-end

RUBY_SAMPLE_1__TRIMMED_COMPLETION = "    name: params[:name],\n    config: { repeat:"

RUBY_SAMPLE_2 = (
    # prefix
    "Event.new(\n",
    # completion that would introduce new errors if trimmed back
    "    name: params[:name],\n    config: { repeat: true }",
    # suffix
    "\n)",
)

RUBY_SAMPLE_3 = (
    # prefix
    "Event.new(\n",
    # single line truncated completion that would be empty if trimmed back
    "    test.a_very_long_chain_of_methods.that_cuts_off_",
    # suffix
    "\n)",
)

JAVA_SAMPLE_1 = (
    # prefix
    "final Event event = new Event.Builder()\n",
    # truncated completion that should be trimmed back
    "        .withConfig(config)\n        .withAnotherConfigu",
    # suffix
    "",
)

JAVA_SAMPLE_1__TRIMMED_COMPLETION = "        .withConfig(config)"

PYTHON_SAMPLE_1 = (
    # prefix
    "async def event(self) -> Event:\n",
    # truncated completion that should be trimmed back
    "    name = get_event_name()\n    event = await get_event_configu",
    # suffix
    "\n return event",
)

PYTHON_SAMPLE_1__TRIMMED_COMPLETION = "    name = get_event_name()\n    event = await"

GO_SAMPLE_1 = (
    # prefix
    "for len(events) > 0 {\n",
    # truncated completion that should be trimmed back
    "     event := events[0]\n      events = events[1",
    # suffix
    "\n      event.Name = name",
)

GO_SAMPLE_1__TRIMMED_COMPLETION = "     event := events[0]\n      events ="


@pytest.mark.parametrize(
    (
        "lang_id",
        "code_sample",
        "max_output_tokens_used",
        "raw_completion",
        "completion",
        "expected_completion",
    ),
    [
        # Truncated completion that should be trimmed back
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_1,
            True,
            RUBY_SAMPLE_1[1],
            RUBY_SAMPLE_1[1],
            RUBY_SAMPLE_1__TRIMMED_COMPLETION,
        ),
        # Completion does not use exactly max output tokens
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_1,
            False,
            RUBY_SAMPLE_1[1],
            RUBY_SAMPLE_1[1],
            RUBY_SAMPLE_1[1],  # Completion is unchanged
        ),
        # Completion was modified by a previous post processor
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_1,
            True,
            RUBY_SAMPLE_1[1] + ".code_removed_from_previous_post_processor",
            RUBY_SAMPLE_1[1],
            RUBY_SAMPLE_1[1],  # Completion is unchanged
        ),
        # Completion ends with a space
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_1,
            True,
            RUBY_SAMPLE_1[1] + " ",
            RUBY_SAMPLE_1[1] + " ",
            RUBY_SAMPLE_1[1] + " ",  # Completion is unchanged
        ),
        # Completion ends with a newline
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_1,
            True,
            RUBY_SAMPLE_1[1] + "\n",
            RUBY_SAMPLE_1[1] + "\n",
            RUBY_SAMPLE_1[1] + "\n",  # Completion is unchanged
        ),
        # Completion that would introduce new errors if trimmed back
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_2,
            True,
            RUBY_SAMPLE_2[1],
            RUBY_SAMPLE_2[1],
            RUBY_SAMPLE_2[1],  # Completion is unchanged
        ),
        # Single line truncated completion that would be empty if trimmed back
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_3,
            True,
            RUBY_SAMPLE_3[1],
            RUBY_SAMPLE_3[1],
            RUBY_SAMPLE_3[1],  # Completion is unchanged
        ),
        # Truncated completion that should be trimmed back
        (
            LanguageId.JAVA,
            JAVA_SAMPLE_1,
            True,
            JAVA_SAMPLE_1[1],
            JAVA_SAMPLE_1[1],
            JAVA_SAMPLE_1__TRIMMED_COMPLETION,
        ),
        # Truncated completion that should be trimmed back
        (
            LanguageId.PYTHON,
            PYTHON_SAMPLE_1,
            True,
            PYTHON_SAMPLE_1[1],
            PYTHON_SAMPLE_1[1],
            PYTHON_SAMPLE_1__TRIMMED_COMPLETION,
        ),
        # Truncated completion that should be trimmed back
        (
            LanguageId.GO,
            GO_SAMPLE_1,
            True,
            GO_SAMPLE_1[1],
            GO_SAMPLE_1[1],
            GO_SAMPLE_1__TRIMMED_COMPLETION,
        ),
    ],
)
@pytest.mark.asyncio
async def test_fix_truncation(
    lang_id: LanguageId,
    code_sample: tuple,
    max_output_tokens_used: bool,
    raw_completion: str,
    completion: str,
    expected_completion: str,
):
    prefix, _, suffix = code_sample

    actual_completion = await fix_truncation(
        prefix,
        completion,
        suffix,
        max_output_tokens_used,
        raw_completion,
        lang_id,
    )

    assert actual_completion == expected_completion
